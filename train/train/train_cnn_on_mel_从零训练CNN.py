"""
复现 Pandeya model_best.hdf5 架构 + 10类分类头
=============================================
用 PyTorch 重建 5×Conv2D(32ch) + GlobalMaxPool + Dense50 → 加 Dense10

步骤:
  1. 加载 model_best.hdf5 的 Conv+BN+FC 权重 (Keras→PyTorch 转置)
  2. 冻结 pre-trained 部分, 只训新加的 Dense10 头
  3. 测试跨数据集
# 从零训练 CNN：Mel谱图 → MobileCatCNN 架构
"""

import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from pathlib import Path

import h5py
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from torch import nn

# ── 配置 ──
SR = 12000  # Pandeya 用的 12kHz
BASE = Path(r'/path/to/dataset')
MODEL_PATH = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X/model_best.hdf5'
DATA_DIR = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
CROSS_DIR = BASE / '原猫语分类/情绪分类'
OUTPUT_DIR = BASE / '模型代码/训练结果/hubert_v2'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 32
EPOCHS_FT = 30  # fine-tune 轮数
EPOCHS_SCRATCH = 60  # 从头训练轮数

LABELS = [
    "Angry", "Defence", "Fighting", "Happy", "HuntingMind",
    "Mating", "MotherCall", "Paining", "Resting", "Warning"
]

CN_MAP = {
    '交配（低沉嘶吼）': 'Mating', '休息': 'Resting', '快乐': 'Happy',
    '愤怒（呜哇哇哇）': 'Angry', '打架（尖锐刺耳）': 'Fighting',
    '母体呼叫（警笛）': 'MotherCall', '狩猎（短促的嗷呜）': 'HuntingMind',
    '疼痛': 'Paining', '警告（呜~）（咕噜）': 'Warning',
    '防御(响尾蛇嘶哈)': 'Defence',
}
MERGE_6 = {
    'Angry':'Aggressive', 'Defence':'Aggressive', 'Fighting':'Aggressive', 'Warning':'Aggressive',
    'Happy':'Happy', 'HuntingMind':'Happy', 'Mating':'Mating', 'MotherCall':'MotherCall',
    'Paining':'Paining', 'Resting':'Resting',
}

# =========================================================
# Mel 频谱计算 (匹配 Pandeya melgram 参数)
# =========================================================
MAX_MEL_FRAMES = 200  # ~4.3s @ 12kHz/256hop, 猫叫声一般 0.5-2s

def compute_melgram(y, orig_sr=None):
    """匹配 Pandeya: sr=12000, 96mel, 512FFT, 256hop, dB scale, per-sample norm"""
    if orig_sr is not None and orig_sr != 12000:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=12000)
    
    # 硬截断: 防止 MP3 头损坏导致超长解码 (OOM)
    max_samples = MAX_MEL_FRAMES * 256  # 51200 = 4.27s @ 12kHz
    if len(y) > max_samples:
        import warnings
        warnings.warn(f'[TRUNCATE] 音频 {len(y)/12000:.1f}s → {max_samples/12000:.1f}s (max frames={MAX_MEL_FRAMES})')
        y = y[:max_samples]
    
    mel = librosa.feature.melspectrogram(
        y=y, sr=12000, n_fft=512, hop_length=256,
        n_mels=96, fmin=0, fmax=6000, power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Per-sample normalization (匹配 Normalization2D)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    # PyTorch 格式: (C=1, F=96, T)
    mel_t = torch.FloatTensor(mel_db[np.newaxis, :, :])  # (1, 96, T)
    # 安全钳: mel 维度也硬截断
    if mel_t.shape[-1] > MAX_MEL_FRAMES:
        mel_t = mel_t[:, :, :MAX_MEL_FRAMES]
    return mel_t


# =========================================================
# Pandeya CNN 架构 (PyTorch 重建)
# =========================================================
class PandeyaCNN(nn.Module):
    """5×Conv2D(32,3×3) + BN + ELU + MaxPool → GlobalMaxPool → Dense50
    
    原始 model_best.hdf5 是特征提取器.
    支持两种特征提取模式:
      mode='pool':  输出 32 维 (GlobalMaxPool 后, Dense 前)  — 更原始的特征
      mode='dense': 输出 50 维 (Dense50+sigmoid 后)           — 压缩后的嵌入
    
    输入: (B, 1, 96, T) -> 输出: (B, 32) 或 (B, 50)
    """
    def __init__(self, mode='dense'):
        super().__init__()
        self.mode = mode
        
        def conv_block(in_ch, out_ch, pool_size):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch, momentum=0.01),
                nn.ELU(),
                nn.MaxPool2d(pool_size, ceil_mode=True),
            )
        
        self.conv1 = conv_block(1, 32, (2, 4))
        self.conv2 = conv_block(32, 32, (3, 4))
        self.conv3 = conv_block(32, 32, (2, 5))
        self.conv4 = conv_block(32, 32, (2, 4))
        self.conv5 = conv_block(32, 32, (4, 4))
        
        self.fc = nn.Linear(32, 50)   # 对应 dense_1
    
    def forward(self, x, return_features=False):
        """
        return_features=False: (B, 32) pool 特征 或 (B, 50) dense 特征
        return_features=True : dict {conv1_emb,...,conv5_emb, pool, dense}
        """
        feats = {}
        x = self.conv1(x)
        if return_features: feats['conv1'] = x
        x = self.conv2(x)
        if return_features: feats['conv2'] = x
        x = self.conv3(x)
        if return_features: feats['conv3'] = x
        x = self.conv4(x)
        if return_features: feats['conv4'] = x
        x = self.conv5(x)
        if return_features: feats['conv5'] = x
        x = F.adaptive_max_pool2d(x, (1, 1))
        x = x.flatten(1)
        if return_features: feats['pool'] = x
        x = torch.sigmoid(self.fc(x))
        if return_features: feats['dense'] = x
        
        if return_features:
            return feats
        if self.mode == 'pool':
            return feats['pool']
        return x


# 注: CustomCNN (32→64→128) 已从对比实验中移除
# 保留 PandeyaCNN + SVM (匹配论文方法)
# 加载 Pandeya 权重 (Keras hdf5 → PyTorch)
# =========================================================
def load_pandeya_weights(model, h5_path):
    """从 model_best.hdf5 加载 Conv+BN+FC 权重, Keras NHWC → PyTorch NCHW"""
    f = h5py.File(h5_path, 'r')
    wg = f['model_weights']
    
    def load_np(layer, name):
        return wg[layer][layer][name][:]
    
    # Conv layers: Keras (H,W,in,out) → PyTorch (out,in,H,W)
    for i in range(1, 6):
        conv = getattr(model, f'conv{i}')
        conv[0].weight.data.copy_(
            torch.from_numpy(np.transpose(load_np(f'conv2d_{i}', 'kernel:0'), (3, 2, 0, 1)))
        )
        conv[0].bias.data.copy_(torch.from_numpy(load_np(f'conv2d_{i}', 'bias:0')))
        
        bn = conv[1]
        bn.weight.data.copy_(torch.from_numpy(load_np(f'batch_normalization_{i}', 'gamma:0')))
        bn.bias.data.copy_(torch.from_numpy(load_np(f'batch_normalization_{i}', 'beta:0')))
        bn.running_mean.data.copy_(torch.from_numpy(load_np(f'batch_normalization_{i}', 'moving_mean:0')))
        bn.running_var.data.copy_(torch.from_numpy(load_np(f'batch_normalization_{i}', 'moving_variance:0')))
    
    # FC: Keras (in,out) → PyTorch (out,in)
    fc_w = load_np('dense_1', 'kernel:0')
    fc_b = load_np('dense_1', 'bias:0')
    model.fc.weight.data.copy_(torch.from_numpy(fc_w.T))
    model.fc.bias.data.copy_(torch.from_numpy(fc_b))
    
    f.close()
    print('  已加载 Pandeya 预训练权重 (5 Conv + 1 FC)')


# =========================================================
# 数据加载
# =========================================================
def load_catsound_v2(data_dir):
    """加载 CatSound_V2, 返回逐条样本列表 (不做配对平均)"""
    import csv
    
    csv_path = data_dir / 'Catmood_NAYA_1xAug.csv'
    samples = []  # [{path, label, cat_id}]
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 3: continue
            csv_path_str = row[1].strip()
            label_idx = int(row[2])
            label_name = LABELS[label_idx]
            filename = os.path.basename(csv_path_str)
            local_path = data_dir / label_name / filename
            if local_path.exists():
                stem = local_path.stem
                cat_id = stem.replace('_aug1(1)', '').lower()
                samples.append({
                    'path': str(local_path),
                    'label': label_name,
                    'cat_id': cat_id,
                })
    
    return samples


def precompute_all_mels(samples):
    """预计算所有样本的 Mel 频谱"""
    all_mels = {}  # path -> tensor (1, 96, T)
    for s in samples:
        if s['path'] not in all_mels:
            y, sr = librosa.load(s['path'], sr=None, mono=True)
            all_mels[s['path']] = compute_melgram(y, sr)
    return all_mels


# =========================================================
# 嵌入提取 + SVM 评估 (匹配 Pandeya 论文: 特征提取器 + RF/SVM)
# =========================================================
def evaluate_embeddings(model, samples, mel_cache, classifier_type='svm'):
    """Frozen 模型提嵌入 → 分类器 (SVM 或 RF), GroupKFold 5折"""
    model.eval()
    all_cat_ids = [s['cat_id'] for s in samples]
    all_labels = [s['label'] for s in samples]
    le = LabelEncoder()
    y = le.fit_transform(all_labels)
    
    mode_name = model.mode if hasattr(model, 'mode') else 'dense'
    feat_dim = 32 if mode_name == 'pool' else 50
    print(f'  提取 {feat_dim}-dim 嵌入 (mode={mode_name})...')
    t0 = time.time()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(samples), BATCH_SIZE):
            batch = samples[i:i+BATCH_SIZE]
            mels = [mel_cache[s['path']].unsqueeze(0) for s in batch]
            max_T = max(m.shape[-1] for m in mels)
            batch_t = torch.cat(
                [F.pad(m, (0, max_T - m.shape[-1])) for m in mels], dim=0
            ).to(DEVICE)
            emb = model(batch_t).cpu().numpy()
            embeddings.append(emb)
    X = np.concatenate(embeddings, axis=0)
    print(f'  完成 ({time.time()-t0:.1f}s), 形状: {X.shape}')
    
    # GroupKFold + 分类器
    from sklearn.ensemble import RandomForestClassifier
    gkf = GroupKFold(n_splits=5)
    fold_accs = []
    scaler = StandardScaler()
    
    for fold, (tr, te) in enumerate(gkf.split(samples, y, all_cat_ids), 1):
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])
        if classifier_type == 'rf':
            clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
        else:
            clf = LinearSVC(C=1.0, max_iter=10000, dual='auto')
        clf.fit(X_tr, y[tr])
        acc = clf.score(X_te, y[te])
        fold_accs.append(acc)
        print(f'    Fold {fold} {classifier_type.upper()}: {acc:.1%}')
    
    mean_acc, std_acc = np.mean(fold_accs), np.std(fold_accs)
    print(f'  {classifier_type.upper()} 5折 CV: {mean_acc:.1%} ± {std_acc:.1%}')
def extract_fdap_features(model, samples, mel_cache):
    """
    类似 FDAP 的多层特征提取:
    - conv3 (H=8):  4 频带 × {mean,std,max} × 32ch = 384
    - conv4 (H=4):  2 频带 × {mean,std,max} × 32ch = 192
    - pool  (32):   1  × {value}            × 32ch = 32
    总计: 608 维
    """
    model.eval()
    print('  提取 FDAP 特征 (3层, 608-dim)...')
    t0 = time.time()
    
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(samples), BATCH_SIZE):
            batch = samples[i:i+BATCH_SIZE]
            mels = [mel_cache[s['path']].unsqueeze(0) for s in batch]
            max_T = max(m.shape[-1] for m in mels)
            batch_t = torch.cat(
                [F.pad(m, (0, max_T - m.shape[-1])) for m in mels], dim=0
            ).to(DEVICE)
            
            feats = model(batch_t, return_features=True)
            batch_fdap = []
            
            # conv3: (B, 32, 8, W) → 4 bands × 2 per band
            for layer_name, n_bands in [('conv3', 4), ('conv4', 2)]:
                fmap = feats[layer_name]  # (B, C, H, W)
                B, C, H, W = fmap.shape
                band_size = H // n_bands
                for b in range(n_bands):
                    band = fmap[:, :, b*band_size:(b+1)*band_size, :]  # (B, C, band_size, W)
                    batch_fdap.append(band.mean(dim=[2, 3]))     # mean
                    batch_fdap.append(band.std(dim=[2, 3]))      # std
                    batch_fdap.append(band.max(dim=3)[0].max(dim=2)[0])  # max over H,W
            
            # pool: (B, 32) — already computed
            batch_fdap.append(feats['pool'])  # (B, 32)
            
            # 合并
            all_feats.append(torch.cat(batch_fdap, dim=1).cpu().numpy())
    
    X = np.concatenate(all_feats, axis=0)
    print(f'  完成 ({time.time()-t0:.1f}s), 形状: {X.shape}')
    return X
    """封装嵌入 + SVM 分类器用于跨数据集评估"""
    def __init__(self, model, scaler, svm, le):
        self.model = model
        self.scaler = scaler
        self.svm = svm
        self.le = le
    
    def predict(self, mel_tensor):
        self.model.eval()
        with torch.no_grad():
            emb = self.model(mel_tensor.unsqueeze(0).to(DEVICE)).cpu().numpy()
        emb_scaled = self.scaler.transform(emb)
        return self.le.inverse_transform(self.svm.predict(emb_scaled))[0]


# =========================================================
# 跨数据集评估
# =========================================================
def eval_cross(model, le, data_dir, device=DEVICE):
    """在原猫语分类 163 条上评估 (批量推理)"""
    model.eval()
    total, correct_10, correct_6 = 0, 0, 0
    
    # 收集所有文件
    all_files, all_true_10, all_true_6 = [], [], []
    for cn_dir in sorted(data_dir.iterdir()):
        if not cn_dir.is_dir(): continue
        en_label = CN_MAP.get(cn_dir.name)
        if en_label is None: continue
        merged6 = MERGE_6[en_label]
        
        files = sorted(cn_dir.glob('*.wav')) + sorted(cn_dir.glob('*.mp3'))
        for fp in files:
            all_files.append(fp)
            all_true_10.append(en_label)
            all_true_6.append(merged6)
    
    # 分批推理
    with torch.no_grad():
        for i in range(0, len(all_files), BATCH_SIZE):
            batch_files = all_files[i:i+BATCH_SIZE]
            batch_mels = []
            for fp in batch_files:
                y, sr = librosa.load(str(fp), sr=None, mono=True)
                mel = compute_melgram(y, sr)  # (1, 96, T)
                batch_mels.append(mel.unsqueeze(0))
            
            # 填充到相同长度
            max_T = max(m.shape[-1] for m in batch_mels)
            padded = []
            for m in batch_mels:
                if m.shape[-1] < max_T:
                    pad = (0, max_T - m.shape[-1])
                    padded.append(F.pad(m, pad))
                else:
                    padded.append(m)
            batch = torch.cat(padded, dim=0).to(device)
            
            logits = model(batch)
            preds = logits.argmax(1).cpu().numpy()
            
            for j in range(len(batch_files)):
                total += 1
                pred_label = le.inverse_transform([preds[j]])[0]
                pred_6 = MERGE_6.get(pred_label, pred_label)
                
                if pred_label == all_true_10[i+j]:
                    correct_10 += 1
                if pred_6 == all_true_6[i+j]:
                    correct_6 += 1
    
    print(f'    10类: {correct_10}/{total} = {correct_10/total:.1%}')
    print(f'    6类:  {correct_6}/{total} = {correct_6/total:.1%}')
    return correct_10/total, correct_6/total


def evaluate_cross_embedding(classifier, mel_cache):
    """用 EmbeddingClassifier 评估原猫语分类 163 条数据"""
    model = classifier.model
    model.eval()
    total, correct_10, correct_6 = 0, 0, 0
    
    all_files, all_true_10, all_true_6 = [], [], []
    for cn_dir in sorted(CROSS_DIR.iterdir()):
        if not cn_dir.is_dir(): continue
        en_label = CN_MAP.get(cn_dir.name)
        if en_label is None: continue
        merged6 = MERGE_6[en_label]
        for fp in sorted(cn_dir.glob('*.wav')) + sorted(cn_dir.glob('*.mp3')):
            all_files.append(fp)
            all_true_10.append(en_label)
            all_true_6.append(merged6)
    
    with torch.no_grad():
        for i in range(0, len(all_files), BATCH_SIZE):
            batch_files = all_files[i:i+BATCH_SIZE]
            mels = []
            for fp in batch_files:
                y, sr = librosa.load(str(fp), sr=None, mono=True)
                mel = compute_melgram(y, sr)
                mels.append(mel.unsqueeze(0))
            max_T = max(m.shape[-1] for m in mels)
            batch = torch.cat([F.pad(m, (0, max_T - m.shape[-1])) for m in mels], dim=0).to(DEVICE)
            embs = model(batch).cpu().numpy()
            embs_scaled = classifier.scaler.transform(embs)
            preds = classifier.svm.predict(embs_scaled)
            
            for j, pred_idx in enumerate(preds):
                total += 1
                pred_label = classifier.le.inverse_transform([pred_idx])[0]
                pred_6 = MERGE_6.get(pred_label, pred_label)
                if pred_label == all_true_10[i+j]:
                    correct_10 += 1
                if pred_6 == all_true_6[i+j]:
                    correct_6 += 1
    
    print(f'    10类: {correct_10}/{total} = {correct_10/total:.1%}')
    print(f'    6类:  {correct_6}/{total} = {correct_6/total:.1%}')
    return correct_10/total, correct_6/total


def cv_on_features(X, samples, classifier_type='rf'):
    """对预计算的特征矩阵做 GroupKFold + 分类器"""
    from sklearn.ensemble import RandomForestClassifier
    all_cat_ids = [s['cat_id'] for s in samples]
    all_labels = [s['label'] for s in samples]
    le = LabelEncoder()
    y = le.fit_transform(all_labels)
    
    gkf = GroupKFold(n_splits=5)
    accs = []
    scaler = StandardScaler()
    for tr, te in gkf.split(samples, y, all_cat_ids):
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])
        if classifier_type == 'rf':
            clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
        else:
            clf = LinearSVC(C=1.0, max_iter=10000, dual='auto')
        clf.fit(X_tr, y[tr])
        accs.append(clf.score(X_te, y[te]))
    return np.mean(accs), np.std(accs)


def main():
    print('═' * 60)
    print('  Pandeya CNN 预训练权重 — 特征 + 分类器对比')
    print('═' * 60)
    
    # 加载数据
    print('\n[1/3] 加载 CatSound_V2 逐条样本...')
    samples = load_catsound_v2(DATA_DIR)
    print(f'  共 {len(samples)} 条样本')
    
    print('\n  预计算 Mel 频谱...')
    t0 = time.time()
    mel_cache = precompute_all_mels(samples)
    print(f'  完成 ({time.time()-t0:.1f}s, {len(mel_cache)} 条)')
    
    results = []
    
    for mode in ['pool', 'dense']:
        for clf_type in ['svm', 'rf']:
            feat_name = '32-dim (pool)' if mode == 'pool' else '50-dim (dense)'
            label = f'{feat_name} + {clf_type.upper()}'
            
            print(f'\n{"="*60}')
            print(f'  {label}')
            print(f'{"="*60}')
            
            model = PandeyaCNN(mode=mode).to(DEVICE)
            load_pandeya_weights(model, MODEL_PATH)
            
            scaler, clf, le, mean_acc, _ = evaluate_embeddings(
                model, samples, mel_cache, classifier_type=clf_type)
            results.append((label, mean_acc))
    
    # FDAP 多层特征
    model_fdap = PandeyaCNN(mode='pool').to(DEVICE)
    load_pandeya_weights(model_fdap, MODEL_PATH)
    print(f'\n{"="*60}')
    print('  FDAP 特征 (conv3+conv4+pool, 608-dim) + RF')
    print(f'{"="*60}')
    X_fdap = extract_fdap_features(model_fdap, samples, mel_cache)
    fdap_mean, fdap_std = cv_on_features(X_fdap, samples, classifier_type='rf')
    print(f'  RF 5折 CV: {fdap_mean:.1%} ± {fdap_std:.1%}')
    results.append(('FDAP (608-dim) + RF', fdap_mean))
    
    # 结果汇总
    print(f'\n{"═"*60}')
    print('  结果汇总')
    print(f'{"═"*60}')
    for label, acc in sorted(results, key=lambda x: -x[1]):
        print(f'  {label:<35s} {acc:.1%}')
    print(f'{"═"*60}')
    print('  论文声称 (FDAP + RF): 87.3%')
    print(f'{"═"*60}')


if __name__ == '__main__':
    main()

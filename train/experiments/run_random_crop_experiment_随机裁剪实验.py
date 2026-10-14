"""
Random Crop 实验: CatSound_V2 每文件随机取 2s → HuBERT freeze + SVM
对比: 全文件 CV 82.2% vs Random Crop 2s CV
# 随机裁剪实验：测试不同音频时长对模型影响
"""

import os, sys, time, warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

import numpy as np
import torch
import librosa
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupKFold
from transformers import HubertModel, Wav2Vec2FeatureExtractor

SR = 16000
CROP_DUR = 2.0        # 固定窗口 2s
CROP_SAMPLES = int(SR * CROP_DUR)  # 32000
N_CROPS = 3           # 每文件随机取 3 个 2s 窗口 → 嵌入平均
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYERS = [3, 4, 5]
SEED = 42

BASE = Path(r'/path/to/dataset')
DATA_DIR = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
LABELS = [
    'Angry', 'Defence', 'Fighting', 'Happy', 'HuntingMind',
    'Mating', 'MotherCall', 'Paining', 'Resting', 'Warning'
]

np.random.seed(SEED)

print('═' * 65)
print(f'  Random Crop 2s ×{N_CROPS} + HuBERT freeze + SVM')
print('═' * 65)

# ── 1. 加载数据 + Multi-Crop ──
print(f'\n[1/3] 加载 + Random Crop 2s (每文件 {N_CROPS} 次)...')
samples = []  # [{waveforms: [3×2s], label, cat_id}]

for label in LABELS:
    dir_path = DATA_DIR / label
    if not dir_path.exists(): continue
    for fp in sorted(dir_path.glob('*.mp3')):
        try:
            y, _sr = librosa.load(str(fp), sr=SR, mono=True)
            n = len(y)
            
            crops = []
            for _ in range(N_CROPS):
                if n > CROP_SAMPLES:
                    start = np.random.randint(0, n - CROP_SAMPLES)
                    crop = y[start:start + CROP_SAMPLES]
                else:
                    pad = CROP_SAMPLES - n
                    pad_left = np.random.randint(0, pad + 1)
                    crop = np.pad(y, (pad_left, pad - pad_left))
                crops.append(crop.astype(np.float32))
            
            stem = fp.stem
            cat_id = stem.replace('_aug1(1)', '').lower()
            samples.append({
                'waveforms': crops,
                'label': label,
                'cat_id': cat_id,
            })
        except Exception as e:
            print(f'  [FAIL] {label}/{fp.name}: {e}')

print(f'  共 {len(samples)} 个文件 ({len(samples)*N_CROPS} 条 crop)')

# ── 2. HuBERT 嵌入 ──
print('\n[2/3] 提取 HuBERT 嵌入...')
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    'facebook/hubert-base-ls960', local_files_only=True)
hubert = HubertModel.from_pretrained(
    'facebook/hubert-base-ls960', local_files_only=True).to(DEVICE)
hubert.eval()

all_embs = []
all_labels = [s['label'] for s in samples]
all_cats = [s['cat_id'] for s in samples]

t0 = time.time()
with torch.no_grad():
    for i in range(0, len(samples), BATCH_SIZE):
        batch = samples[i:i+BATCH_SIZE]
        # 每文件 N_CROPS 个波形, 全部展开成独立样本
        all_waves = sum([s['waveforms'] for s in batch], [])
        
        inputs = feature_extractor(
            all_waves, sampling_rate=SR,
            return_tensors='pt', padding=True,
        ).to(DEVICE)
        
        outputs = hubert(inputs.input_values, output_hidden_states=True)
        hs = outputs.hidden_states
        crop_embs = torch.stack([hs[l] for l in LAYERS], dim=0).mean(dim=0).mean(dim=1)
        
        # 按每 N_CROPS 个 crop 平均, 恢复为 1 个嵌入/文件
        for j in range(len(batch)):
            avg_emb = crop_embs[j*N_CROPS:(j+1)*N_CROPS].mean(dim=0)
            all_embs.append(avg_emb.cpu().numpy())

X = np.stack(all_embs, axis=0)
le = LabelEncoder()
y = le.fit_transform(all_labels)
print(f'  完成 ({time.time()-t0:.1f}s), X={X.shape}')

# ── 3. GroupKFold SVM ──
print('\n[3/3] GroupKFold SVM...')
gkf = GroupKFold(n_splits=5)
fold_accs = []

for fold, (tr, te) in enumerate(gkf.split(samples, y, all_cats), 1):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr])
    X_te = scaler.transform(X[te])
    svm = LinearSVC(C=1.0, max_iter=10000, dual='auto')
    svm.fit(X_tr, y[tr])
    acc = svm.score(X_te, y[te])
    fold_accs.append(acc)
    print(f'  Fold {fold}: {acc:.1%}')

mean_acc = np.mean(fold_accs)
std_acc = np.std(fold_accs)
print(f'\n{"="*65}')
print(f'  Random Crop 2s  SVM 5折 CV: {mean_acc:.1%} ± {std_acc:.1%}')
print(f'  原文 (全文件)     SVM 5折 CV: 82.2% ± 1.3%')
print(f'  事件分割          SVM 5折 CV: 72.5% ± 0.8%')
print(f'{"="*65}')

# ── 4. 用全量数据训练最终模型, 测新猫语 40 条 ──
NEW_DIR = BASE / '新猫语分类/新情绪分类'
CN_MAP = {
    '友好': 'Happy', '吸引注意': 'Mating',
    '愤怒示威': 'Angry', '渴求': 'NULL',
    '警告驱逐': 'Warning',
}
MERGE_6 = {
    'Angry':'Aggressive','Defence':'Aggressive','Fighting':'Aggressive','Warning':'Aggressive',
    'Happy':'Happy','HuntingMind':'Happy','Mating':'Mating','MotherCall':'MotherCall',
    'Paining':'Paining','Resting':'Resting',
}

print(f'\n{"="*65}')
print(f'  测新猫语 40 条')
print(f'{"="*65}')

# 全量训练
scaler = StandardScaler()
X_all = scaler.fit_transform(X)
svm_all = LinearSVC(C=1.0, max_iter=10000, dual='auto')
svm_all.fit(X_all, y)
print(f'  全量训练完成, 模型: {svm_all}')

# 加载新猫语 40 条
np.random.seed(42 + 999)  # 跟训练用不同 seed
new_samples = []
new_labels_10 = []
new_labels_cn = []
new_fnames = []
for cn_dir in sorted(NEW_DIR.iterdir()):
    if not cn_dir.is_dir(): continue
    en_label = CN_MAP.get(cn_dir.name, cn_dir.name)
    for fp in sorted(cn_dir.glob('*')):
        try:
            y, _sr = librosa.load(str(fp), sr=SR, mono=True)
            n = len(y)
            # 同训练逻辑: 3 次 crop → 平均
            crops = []
            for _ in range(N_CROPS):
                if n > CROP_SAMPLES:
                    start = np.random.randint(0, n - CROP_SAMPLES)
                    crop = y[start:start + CROP_SAMPLES]
                else:
                    pad = CROP_SAMPLES - n
                    pad_left = np.random.randint(0, pad + 1)
                    crop = np.pad(y, (pad_left, pad - pad_left))
                crops.append(crop.astype(np.float32))
            new_samples.append(crops)
            new_labels_10.append(en_label)
            new_labels_cn.append(cn_dir.name)
            new_fnames.append(os.path.basename(str(fp))[:35])
        except Exception as e:
            print(f'  [FAIL] 新猫语/{cn_dir.name}/{fp.name}: {e}')

print(f'  加载 {len(new_samples)} 条')

# 提取嵌入
new_embs = []
with torch.no_grad():
    for i in range(0, len(new_samples), BATCH_SIZE):
        batch = new_samples[i:i+BATCH_SIZE]
        all_waves = sum(batch, [])
        inputs = feature_extractor(
            all_waves, sampling_rate=SR,
            return_tensors='pt', padding=True,
        ).to(DEVICE)
        outputs = hubert(inputs.input_values, output_hidden_states=True)
        hs = outputs.hidden_states
        crop_embs = torch.stack([hs[l] for l in LAYERS], dim=0).mean(dim=0).mean(dim=1)
        for j in range(len(batch)):
            avg_emb = crop_embs[j*N_CROPS:(j+1)*N_CROPS].mean(dim=0)
            new_embs.append(avg_emb.cpu().numpy())

X_new = np.stack(new_embs, axis=0)
X_new_scaled = scaler.transform(X_new)
pred_indices = svm_all.predict(X_new_scaled)
pred_labels = le.inverse_transform(pred_indices)

# 结果
total_10, correct_10 = 0, 0
total_6, correct_6 = 0, 0
per_class = {}

for i, pred in enumerate(pred_labels):
    true_10 = new_labels_10[i]
    true_cn = new_labels_cn[i]
    pred_6 = MERGE_6.get(pred, pred)
    true_6 = MERGE_6.get(true_10, true_10)
    
    is_correct_10 = pred == true_10
    is_correct_6 = pred_6 == true_6
    if is_correct_10: correct_10 += 1
    if is_correct_6: correct_6 += 1
    total_10 += 1
    total_6 += 1
    
    check = '✓' if is_correct_10 else '✗'
    fname = new_fnames[i]
    print(f'  {check} [{true_cn:<6s}→{pred:<12s}] {fname}')
    
    if true_10 not in per_class:
        per_class[true_10] = {'total': 0, 'corr': 0}
    per_class[true_10]['total'] += 1
    if is_correct_10:
        per_class[true_10]['corr'] += 1

print(f'\n  ─── 汇总 ───')
print(f'  10类准确率: {correct_10}/{total_10} = {correct_10/total_10:.1%}')
print(f'  6类合并:   {correct_6}/{total_6} = {correct_6/total_6:.1%}')
for label, info in sorted(per_class.items()):
    print(f'  {label:<15s} {info["corr"]}/{info["total"]} = {info["corr"]/info["total"]:.1%}')

# ── 5. 跨数据集: 原猫语分类 163 条 ──
CROSS_163 = BASE / '原猫语分类/情绪分类'
CN_MAP_163 = {
    '交配（低沉嘶吼）': 'Mating', '休息': 'Resting', '快乐': 'Happy',
    '愤怒（呜哇哇哇）': 'Angry', '打架（尖锐刺耳）': 'Fighting',
    '母体呼叫（警笛）': 'MotherCall', '狩猎（短促的嗷呜）': 'HuntingMind',
    '疼痛': 'Paining', '警告（呜~）（咕噜）': 'Warning',
    '防御(响尾蛇嘶哈)': 'Defence',
}

print(f'\n{"="*65}')
print(f'  跨数据集: 原猫语分类 163 条')
print(f'{"="*65}')

np.random.seed(42 + 888)  # 第三个 seed
cross_files = []
for cn_dir in sorted(CROSS_163.iterdir()):
    if not cn_dir.is_dir(): continue
    en_label = CN_MAP_163.get(cn_dir.name)
    if en_label is None: continue
    for fp in sorted(cn_dir.glob('*.wav')) + sorted(cn_dir.glob('*.mp3')):
        cross_files.append((str(fp), en_label))

print(f'  共 {len(cross_files)} 条')

# 加载 + crop
cross_crops = []
cross_labels = []
cross_fnames = []
for fp, en_label in cross_files:
    try:
        y, _sr = librosa.load(fp, sr=SR, mono=True)
        n = len(y)
        crops = []
        for _ in range(N_CROPS):
            if n > CROP_SAMPLES:
                start = np.random.randint(0, n - CROP_SAMPLES)
                crop = y[start:start + CROP_SAMPLES]
            else:
                pad = CROP_SAMPLES - n
                pad_left = np.random.randint(0, pad + 1)
                crop = np.pad(y, (pad_left, pad - pad_left))
            crops.append(crop.astype(np.float32))
        cross_crops.append(crops)
        cross_labels.append(en_label)
        cross_fnames.append(os.path.basename(fp)[:35])
    except Exception as e:
        print(f'  [FAIL] 跨数据集/{en_label}/{os.path.basename(fp)[:35]}: {e}')

# 提取嵌入
cross_embs = []
with torch.no_grad():
    for i in range(0, len(cross_crops), BATCH_SIZE):
        batch = cross_crops[i:i+BATCH_SIZE]
        all_waves = sum(batch, [])
        inputs = feature_extractor(
            all_waves, sampling_rate=SR,
            return_tensors='pt', padding=True,
        ).to(DEVICE)
        outputs = hubert(inputs.input_values, output_hidden_states=True)
        hs = outputs.hidden_states
        crop_embs = torch.stack([hs[l] for l in LAYERS], dim=0).mean(dim=0).mean(dim=1)
        for j in range(len(batch)):
            avg_emb = crop_embs[j*N_CROPS:(j+1)*N_CROPS].mean(dim=0)
            cross_embs.append(avg_emb.cpu().numpy())

X_cross = np.stack(cross_embs, axis=0)
X_cross_scaled = scaler.transform(X_cross)
cross_preds = le.inverse_transform(svm_all.predict(X_cross_scaled))

correct_10 = sum(p == t for p, t in zip(cross_preds, cross_labels))
total = len(cross_preds)
print(f'  10类: {correct_10}/{total} = {correct_10/total:.1%}')

# 6类合并
correct_6 = 0
for p, t in zip(cross_preds, cross_labels):
    if MERGE_6.get(p, p) == MERGE_6.get(t, t):
        correct_6 += 1
print(f'  6类:  {correct_6}/{total} = {correct_6/total:.1%}')
print(f'  (原始全文件模型: 10类=31.1%, 6类=41.5%)')

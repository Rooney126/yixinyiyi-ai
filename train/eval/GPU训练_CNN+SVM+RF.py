# GPU训练：CNN/SVM/RF 分类器对比（含数据增强）
# -*- coding: utf-8 -*-
"""
译芯怡意 — GPU 加速训练 v5 (SVM + CNN)
===============================
预处理: 归一化 → 对齐(随机裁剪/对称反射) → Mel Z-score标准化
增强: 噪声/音高偏移/时间伸缩 (训练时随机选用)
SVM: GroupKFold CV + 网格搜索 | CNN: K折CV + warmup + 早停
优化: 文件路径缓存 | Mel 频谱预计算缓存 | 异常处理
"""

import os, json, pickle, warnings
# 修复中文用户名导致的 multiprocessing 编码错误
os.environ['JOBLIB_TEMP_FOLDER'] = 'C:/temp'
os.makedirs('C:/temp', exist_ok=True)
warnings.filterwarnings('ignore')

import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, GroupKFold

# 随机种子 — 确保可复现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ── 路径 ──
USER = os.environ.get('USERNAME', os.environ.get('USER', '陈项目负责人'))
EMOTION_DIR = Path(f'C:/Users/{USER}/Desktop/猫语分类/情绪分类')
TIMBRE_DIR  = Path(f'C:/Users/{USER}/Desktop/猫语分类/音色分类')
OUTPUT_DIR  = Path(f'C:/Users/{USER}/Desktop/猫语分类/模型代码/训练结果')
TS = datetime.now().strftime('%Y%m%d_%H%M%S')

# ── 参数 ──
SR = 16000
DURATION = 2.0
TARGET_LEN = int(SR * DURATION)
N_FFT = 512
HOP_LENGTH = 256
N_MELS = 64
N_MFCC = 20
FMIN = 50
FMAX = 8000
BATCH_SIZE = 16
K_FOLDS = 3
PITCH_SHIFT = 1.0
WEIGHT_DECAY = 5e-5
WARMUP_EPOCHS = 10
MAX_EPOCHS = 100
PATIENCE = 20
LEARNING_RATE = 0.001


# =========================================================
# 音频路径索引 (一次性建立，避免重复扫描目录)
# =========================================================
def build_file_index(emotion_dir):
    """建立 {filename: full_path} 索引，替代每次 load_audio 时的目录扫描."""
    index = {}
    for cls_dir in emotion_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        for fp in cls_dir.glob('*.wav'):
            index[fp.name] = fp
    return index


def load_audio(fname, file_index):
    """通过预建索引快速加载音频，含异常校验."""
    fp = file_index.get(fname)
    if fp is None:
        raise FileNotFoundError(f'音频文件未找到: {fname}，请检查情绪分类目录结构')
    y, _ = librosa.load(str(fp), sr=SR, mono=True)
    if len(y) == 0:
        raise ValueError(f'音频文件为空或解码失败: {fp}')
    return y


# =========================================================
# 音频预处理
# =========================================================
def normalize(y):
    """峰值归一化到 [-1, 1]."""
    peak = np.max(np.abs(y))
    return y / peak if peak > 1e-10 else y


def align(y, n=TARGET_LEN, random_crop=False):
    """长度对齐：超长→随机裁剪或居中截取, 不足→对称反射填充."""
    if len(y) == n:
        return y
    if len(y) > n:
        start = np.random.randint(0, len(y) - n) if random_crop else (len(y) - n) // 2
        return y[start:start + n]
    pad_total = n - len(y)
    pad_left = pad_total // 2
    return np.pad(y, (pad_left, pad_total - pad_left), mode='reflect')


def augment(y):
    """训练时随机选择 3 种增强之一(1/4 概率不增强)."""
    aug = np.random.randint(0, 4)
    if aug == 0:
        return y                                      # 不增强，提前返回
    elif aug == 1:
        y = y + np.random.randn(len(y)) * np.random.uniform(0.003, 0.01)  # 高斯噪声
    elif aug == 2:
        y = librosa.effects.pitch_shift(y, sr=SR,
                    n_steps=np.random.uniform(-PITCH_SHIFT, PITCH_SHIFT))
        if len(y) < TARGET_LEN:
            pad = TARGET_LEN - len(y)
            y = np.pad(y, (pad // 2, pad - pad // 2), mode='reflect')
        else:
            y = y[:TARGET_LEN]
    elif aug == 3:
        rate = np.random.uniform(0.85, 1.15)
        y = librosa.effects.time_stretch(y, rate=rate)
        if len(y) < TARGET_LEN:
            pad = TARGET_LEN - len(y)
            y = np.pad(y, (pad // 2, pad - pad // 2), mode='reflect')
        else:
            start = np.random.randint(0, len(y) - TARGET_LEN + 1)
            y = y[start:start + TARGET_LEN]
    return np.clip(y, -1.0, 1.0)


# =========================================================
# 频域特征
# =========================================================
def mel_spectrogram(y):
    """Mel 频谱 (64 bands, dB scale)."""
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT,
                                          hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX)
    return librosa.power_to_db(mel, ref=1.0)


def extract_mfcc(y):
    """MFCC 统计特征 (20 coeffs × mean/std → 40维), 供 SVM 使用."""
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])


# =========================================================
# PyTorch Dataset (含 Mel 缓存，省掉非增强路径的重复计算)
# =========================================================
class CatSoundDataset(Dataset):
    def __init__(self, files, labels, mel_mean, mel_std, file_index,
                 augment_flag=False, mel_cache=None):
        self.files = files
        self.labels = labels
        self.mel_mean = mel_mean         # (n_mels, time)
        self.mel_std = mel_std           # (n_mels, time)
        self.file_index = file_index
        self.mel_cache = mel_cache if mel_cache is not None else {}
        self.augment_flag = augment_flag

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        # 非训练模式 + 缓存命中 → 直接用预计算的 Mel (训练集不需要重复算)
        if not self.augment_flag and fname in self.mel_cache:
            mel = self.mel_cache[fname].copy()
        else:
            y = load_audio(fname, self.file_index)
            y = align(normalize(y), random_crop=self.augment_flag)
            if self.augment_flag:
                y = augment(y)
            mel = mel_spectrogram(y)
        # 逐频率通道 Z-score 标准化
        mel = (mel - self.mel_mean) / self.mel_std
        return torch.FloatTensor(mel).unsqueeze(0), self.labels[idx]


# =========================================================
# CNN 模型 (FDAP: 频率分段平均池化)
# =========================================================
class FDAPMobileCatCNN(nn.Module):
    """
    轻量 CNN + 频率分段平均池化 (Frequency Division Average Pooling).
    参数 ~164K, 适合小数据集.
    """
    def __init__(self, num_classes=10, fdap_bands=4):
        super().__init__()
        self.fdap_bands = fdap_bands
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16),
                                   nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
                                   nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
                                   nn.ReLU(), nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
                                   nn.ReLU(), nn.AdaptiveAvgPool2d((8, None)))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(128 * fdap_bands, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        b, c, f, t = x.shape
        bs = f // self.fdap_bands
        fdap = []
        for i in range(self.fdap_bands):
            end = (i + 1) * bs if i < self.fdap_bands - 1 else f
            fdap.append(x[:, :, i * bs:end, :].mean(dim=[2, 3]))
        x = torch.cat(fdap, dim=1)
        return self.classifier(x)


# =========================================================
# 数据映射 + 按猫划分
# =========================================================
def build_mappings():
    """扫描音色和情绪目录，建立 猫↔文件名↔情绪 的映射."""
    file_to_cat = {}
    cat_emotion_files = defaultdict(lambda: defaultdict(list))
    for root, dirs, files in os.walk(TIMBRE_DIR):
        if root == TIMBRE_DIR:
            continue
        cat_id = os.path.relpath(root, TIMBRE_DIR)
        for f in files:
            if f.endswith('.wav'):
                file_to_cat[f] = f'0/{f}' if cat_id == '0' else cat_id
    for root, dirs, files in os.walk(EMOTION_DIR):
        if root == EMOTION_DIR:
            continue
        emotion = os.path.relpath(root, EMOTION_DIR)
        for f in files:
            if f.endswith('.wav') and f in file_to_cat:
                cat_emotion_files[file_to_cat[f]][emotion].append(f)
    return file_to_cat, cat_emotion_files


def split_data(cat_emotion_files, test_ratio=0.2):
    """按情绪分别选 20% 文件所对应的猫做测试集（同一只猫不会同时出现在训练和测试)."""
    emotion_cats = defaultdict(lambda: defaultdict(list))
    for cat_id, emotions in cat_emotion_files.items():
        for emo, files in emotions.items():
            emotion_cats[emo][cat_id].extend(files)
    test_cats = set()
    for emo in sorted(emotion_cats.keys()):
        cat_data = emotion_cats[emo]
        total = sum(len(v) for v in cat_data.values())
        target = max(1, round(total * test_ratio))
        sorted_cats = sorted(cat_data.items(), key=lambda x: -len(x[1]))
        non_zero = [(c, f) for c, f in sorted_cats if not c.startswith('0/')]
        zero_ones = [(c, f) for c, f in sorted_cats if c.startswith('0/')]
        count = 0
        test = set()
        for cid, fs in non_zero:
            if count < target:
                test.add(cid)
                count += len(fs)
        if count < target and zero_ones:
            np.random.seed(42)
            np.random.shuffle(zero_ones)
            for cid, fs in zero_ones[:target - count]:
                test.add(cid)
                count += len(fs)
        # 只要某情绪被选为测试猫，就加入全局 test_cats
        test_cats |= test
        print(f'  {emo:20s}: {total:3d} → 训练{total - count:3d} 测试{count:3d}')
    # 不在 test_cats 的所有猫都是训练猫
    train_cats = {cid for cid in cat_emotion_files if cid not in test_cats}
    return train_cats, test_cats


# =========================================================
# SVM 嵌套交叉验证 (GroupKFold + GridSearchCV)
# =========================================================
def svm_grid_cv(X, y, cat_ids):
    print(f'\nSVM 网格搜索 + {K_FOLDS}折 GroupKFold...')
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.1, 0.01, 0.001],
        'kernel': ['rbf'],
        'class_weight': ['balanced'],
    }
    gkf = GroupKFold(n_splits=K_FOLDS)
    fold_scores = []
    best_params = None
    best_score = 0
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, cat_ids)):
        X_tr = X[train_idx]
        X_val = X[val_idx]
        y_tr = y[train_idx]
        y_val = y[val_idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        grid = GridSearchCV(SVC(probability=True, random_state=42, max_iter=5000),
                             param_grid, cv=min(3, len(np.unique(y_tr))),
                             scoring='accuracy', n_jobs=-1)
        grid.fit(X_tr_s, y_tr)
        val_acc = grid.score(X_val_s, y_val)
        fold_scores.append(val_acc)
        print(f'  折{fold + 1}: val={val_acc:.3f}  params={grid.best_params_}')
        if val_acc > best_score:
            best_score = val_acc
            best_params = grid.best_params_
    print(f'  SVM CV均值: {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f}')
    print(f'  最佳参数: {best_params}')
    return best_params, fold_scores


# =========================================================
# CNN 单折训练 (warmup + 早停)
# =========================================================
def train_cnn_fold(train_files, train_labels, val_files, val_labels,
                   mel_mean, mel_std, file_index, mel_cache, n_classes, device):
    train_ds = CatSoundDataset(train_files, train_labels, mel_mean, mel_std,
                               file_index, augment_flag=True)
    val_ds = CatSoundDataset(val_files, val_labels, mel_mean, mel_std,
                             file_index, mel_cache=mel_cache)  # 验证集用缓存加速
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    model = FDAPMobileCatCNN(num_classes=n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val = 0
    best_state = None
    patience_count = 0
    for epoch in range(MAX_EPOCHS):
        # Warmup 线性增长
        if epoch < WARMUP_EPOCHS:
            for g in opt.param_groups:
                g['lr'] = LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
        model.train()
        train_correct = 0
        train_total = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            nn.CrossEntropyLoss()(model(x), y).backward()
            opt.step()
            _, pred = model(x).max(1)
            train_correct += pred.eq(y).sum().item()
            train_total += y.size(0)
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                _, pred = model(x).max(1)
                val_correct += pred.eq(y).sum().item()
                val_total += y.size(0)
        val_acc = val_correct / val_total
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= PATIENCE:
            if (epoch + 1) % 15 != 0:
                print(f'    E{epoch + 1:3d}  早停在 val={best_val:.3f}')
            break
        if (epoch + 1) % 15 == 0:
            print(f'    E{epoch + 1:3d}  train={train_correct / train_total:.3f}  val={val_acc:.3f}')
    model.load_state_dict(best_state)
    return model, best_val


# =========================================================
# 主流程
# =========================================================
def main():
    print('=' * 55)
    print('  译芯怡意 - GPU 训练 v5 (SVM + CNN)')
    print('  K折CV | Z-score标准化 | Mel缓存 | warmup+早停')
    print('=' * 55)

    # ── GPU 检测 ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n  设备: {device}')
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f'  GPU: {torch.cuda.get_device_name(0)} ({props.total_memory / 1024 ** 3:.1f}GB)')

    # ── 建立文件索引 (一次性扫描，后续所有 load_audio 免扫描) ──
    file_index = build_file_index(EMOTION_DIR)

    # ── 数据映射 + 划分 ──
    file_to_cat, cat_emotion_files = build_mappings()
    print('\n按情绪 + 按猫划分 (20%猫做测试):')
    train_cats, test_cats = split_data(cat_emotion_files)

    train_files, train_emos = [], []
    test_files, test_emos = [], []
    train_cat_labels = []
    for cat_id, emotions in cat_emotion_files.items():
        for emo, files in emotions.items():
            for f in files:
                if cat_id in test_cats:
                    test_files.append(f)
                    test_emos.append(emo)
                else:
                    train_files.append(f)
                    train_emos.append(emo)
                    train_cat_labels.append(cat_id)

    le = LabelEncoder()
    le.fit(train_emos + test_emos)
    y_train = le.transform(train_emos)
    y_test = le.transform(test_emos)
    n_classes = len(le.classes_)
    train_cat_labels = np.array(train_cat_labels)

    # 检查：测试集是否包含训练集不存在的类
    test_only = set(y_test) - set(y_train)
    if test_only:
        print('⚠ 测试集包含训练集没有的类:', [le.classes_[i] for i in test_only])

    print(f'\n训练: {len(train_files)} 样本 ({len(train_cats)} 猫)')
    print(f'测试: {len(test_files)} 样本 ({len(test_cats)} 猫)')
    for emo in le.classes_:
        tr = train_emos.count(emo)
        te = test_emos.count(emo)
        print(f'  {emo:20s}: train {tr:3d}  test {te:3d}')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── SVM 特征提取 ──
    print('\n提取 SVM 特征 (MFCC)...')
    X_train = np.array([extract_mfcc(align(normalize(load_audio(f, file_index)))) for f in train_files])
    X_test = np.array([extract_mfcc(align(normalize(load_audio(f, file_index)))) for f in test_files])

    svm_params, svm_fold_scores = svm_grid_cv(X_train, y_train, train_cat_labels)
    scaler_svm = StandardScaler()
    X_train_s = scaler_svm.fit_transform(X_train)
    X_test_s = scaler_svm.transform(X_test)
    svm_final = SVC(**svm_params, probability=True, random_state=42, max_iter=5000)
    svm_final.fit(X_train_s, y_train)
    svm_test = svm_final.score(X_test_s, y_test)
    print(f'  SVM 测试: {svm_test:.1%}')

    # ── CNN Mel 预计算 + Z-score 参数 ──
    print('\n预计算 Mel 频谱缓存 + 逐频 Z-score 参数...')
    train_mel_cache = {}
    for f in train_files:
        y = load_audio(f, file_index)
        y = normalize(y)
        y = align(y)
        train_mel_cache[f] = mel_spectrogram(y)

    all_mels = np.array([train_mel_cache[f] for f in train_files])  # (n_train, 64, time)
    mel_mean = all_mels.mean(axis=0)   # (64, time)
    mel_std  = all_mels.std(axis=0) + 1e-8
    print(f'  Mel 缓存: {len(train_mel_cache)} 条, 形状: {list(all_mels.shape)}')

    # ── CNN K折CV ──
    print(f'\nCNN K折交叉验证 (K={K_FOLDS}):')
    gkf = GroupKFold(n_splits=K_FOLDS)
    cnn_fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(train_files, y_train, train_cat_labels)):
        print(f'\n  折 {fold + 1}/{K_FOLDS}:')
        tr_f = [train_files[i] for i in train_idx]
        tr_l = y_train[train_idx]
        vl_f = [train_files[i] for i in val_idx]
        vl_l = y_train[val_idx]
        model_fold, val_acc = train_cnn_fold(
            tr_f, tr_l, vl_f, vl_l, mel_mean, mel_std,
            file_index, train_mel_cache, n_classes, device)
        cnn_fold_scores.append(val_acc)
        print(f'  折{fold + 1} val={val_acc:.3f}')

    print(f'\n  CNN CV均值: {np.mean(cnn_fold_scores):.3f} ± {np.std(cnn_fold_scores):.3f}')

    # ── CNN 全量训练 ──
    print(f'\nCNN 全量训练 (训练集 {len(train_files)} 条, {len(train_cats)} 只猫)...')
    all_train_ds = CatSoundDataset(train_files, y_train, mel_mean, mel_std,
                                   file_index, augment_flag=True)
    all_train_loader = DataLoader(all_train_ds, BATCH_SIZE, shuffle=True, num_workers=0)
    test_ds = CatSoundDataset(test_files, y_test, mel_mean, mel_std,
                              file_index, mel_cache=train_mel_cache)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    final_model = FDAPMobileCatCNN(num_classes=n_classes).to(device)
    opt = torch.optim.AdamW(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(MAX_EPOCHS):
        if epoch < WARMUP_EPOCHS:
            for g in opt.param_groups:
                g['lr'] = LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
        final_model.train()
        total_loss = 0
        for x, y in all_train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(final_model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 15 == 0:
            print(f'  E{epoch + 1:3d}  loss={total_loss:.4f}')

    final_model.eval()
    tc = 0
    tt = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            _, p = final_model(x).max(1)
            tc += p.eq(y).sum().item()
            tt += y.size(0)
    cnn_test = tc / tt
    print(f'  CNN 测试: {cnn_test:.1%}')

    # ── 保存模型 ──
    torch.save(final_model.state_dict(), OUTPUT_DIR / f'cnn_gpu_{TS}.pth')
    with open(OUTPUT_DIR / f'svm_{TS}.pkl', 'wb') as f:
        pickle.dump({
            'model': svm_final,
            'scaler': scaler_svm,
            'label_encoder': le,
            'params': svm_params,
            'mel_mean': mel_mean,    # 推理时需要 Z-score 参数
            'mel_std': mel_std,
        }, f)

    # ── 结果 ──
    print('\n' + '=' * 50)
    print('  最终结果')
    print('=' * 50)
    print(f'  SVM (网格搜索+{K_FOLDS}折CV):')
    print(f'    CV均值: {np.mean(svm_fold_scores):.1%}  |  测试: {svm_test:.1%}')
    print(f'  CNN ({K_FOLDS}折CV + warmup + 早停):')
    print(f'    CV均值: {np.mean(cnn_fold_scores):.1%} ± {np.std(cnn_fold_scores):.1%}')
    print(f'    测试: {cnn_test:.1%}')
    # 用 CV 均值选最优（测试集仅做最终报告，不做模型选择）
    svm_cv_mean = np.mean(svm_fold_scores)
    cnn_cv_mean = np.mean(cnn_fold_scores)
    winner = 'SVM' if svm_cv_mean >= cnn_cv_mean else 'CNN'
    print(f'  最佳: {winner} ({max(svm_test, cnn_test):.1%})')

    results = {
        'version': 'v5',
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'n_train': len(train_files),
        'n_test': len(test_files),
        'n_train_cats': len(train_cats),
        'n_test_cats': len(test_cats),
        'svm_cv': [float(s) for s in svm_fold_scores],
        'svm_test': float(svm_test),
        'cnn_cv': [float(s) for s in cnn_fold_scores],
        'cnn_test': float(cnn_test),
        'best_model': winner,
        'config': {
            'k_folds': K_FOLDS, 'batch_size': BATCH_SIZE,
            'warmup_epochs': WARMUP_EPOCHS, 'weight_decay': WEIGHT_DECAY,
            'pitch_shift': PITCH_SHIFT, 'learning_rate': LEARNING_RATE,
            'mel_bands': N_MELS, 'target_len_sec': DURATION,
        },
        'classes': le.classes_.tolist(),
        'timestamp': datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / f'results_v5_{TS}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'\n  保存路径: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()

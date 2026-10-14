# Wav2Vec2 嵌入提取 + MLP/SVM 训练（旧版，已废弃）
# -*- coding: utf-8 -*-
"""
译芯怡意 — Wav2Vec2 嵌入训练 v1 (SVM + RF + MLP)
===============================
原理: Wav2Vec2 自监督预训练模型提取深层嵌入 (768维)
      替代手工 MFCC/Mel 特征，保留 SVM 管线 + 增加 RF + MLP
参考文献:
  [1] Pandeya et al. (2018) — FDAP 创新、VGGish+SVM →
      预训练嵌入+分类器在猫声分类中最有效
  [2] "Scalable Animal Sound Detection" (2025) —
      Wav2Vec2 92.75% vs 手工特征 78.62% (+14%)
  [3] Baevski et al. (2020) — Wav2Vec2: 自监督语音表征
"""

import json
import os
import pickle
import warnings

os.environ['JOBLIB_TEMP_FOLDER'] = 'C:/temp'
os.makedirs('C:/temp', exist_ok=True)
warnings.filterwarnings('ignore')

import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from torch import nn

# Wav2Vec2
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# 随机种子 — 确保可复现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ── 路径 ──
USER = os.environ.get('USERNAME', os.environ.get('USER', 'user'))
EMOTION_DIR = Path(f'C:/Users/{USER}/Desktop/猫语分类/情绪分类')
TIMBRE_DIR  = Path(f'C:/Users/{USER}/Desktop/猫语分类/音色分类')
OUTPUT_DIR  = Path(f'C:/Users/{USER}/Desktop/猫语分类/模型代码/训练结果')
TS = datetime.now().strftime('%Y%m%d_%H%M%S')

# ── 全局参数 ──
# 对齐目标长度：采样率 × 时长（2.0s）
SR = 16000                      # Wav2Vec2 要求 16kHz
DURATION = 2.0                   # 保留历史
TARGET_LEN = int(SR * DURATION)  # 目标样本数
K_FOLDS = 3
WARMUP_EPOCHS = 10

# Wav2Vec2 模型选择
WAV2VEC2_MODEL = "facebook/wav2vec2-base"   # 95M 参数, 768维嵌入
# 备选: "facebook/wav2vec2-base-960h" (ASR 微调版)
# 备选: "facebook/hubert-base-ls960" (HuBERT 同等大小)


# =========================================================
# 文件索引 (一次性建立)
# =========================================================


def build_file_index(emotion_dir):
    index = {}
    for cls_dir in emotion_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        for fp in cls_dir.glob('*.wav'):
            index[fp.name] = fp
    return index


def load_audio(fname, file_index):
    fp = file_index.get(fname)
    if fp is None:
        raise FileNotFoundError(f'音频文件未找到: {fname}')
    y, _ = librosa.load(str(fp), sr=SR, mono=True)
    if len(y) == 0:
        raise ValueError(f'音频文件为空: {fp}')
    return y


# =========================================================
# 音频预处理 (统一到 16kHz / 2秒)
# =========================================================
def normalize(y):
    """峰值归一化到 [-1, 1], 含静音检测."""
    peak = np.max(np.abs(y))
    if peak < 1e-10:
        raise ValueError(f'音频疑似静音或损坏 (peak={peak:.2e})')
    return y / peak


def align_to_target(y):
    """固定对齐: 居中裁剪/对称反射填充到固定长度."""
    n = TARGET_LEN
    if len(y) == n:
        return y
    if len(y) > n:
        start = (len(y) - n) // 2
        return y[start:start + n]
    pad_total = n - len(y)
    pad_left = pad_total // 2
    return np.pad(y, (pad_left, pad_total - pad_left), mode='reflect')


# =========================================================
# Wav2Vec2 嵌入提取 (一次性, GPU 加速)
# =========================================================
class Wav2Vec2EmbeddingExtractor:
    """
    加载 Wav2Vec2 预训练模型.
    冻结所有参数, 提取最后一层隐藏状态, 时序平均 → 768维嵌入.
    """

    def __init__(self, model_name=WAV2VEC2_MODEL, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f'  加载 Wav2Vec2: {model_name} (设备: {self.device})')
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        # 冻结全部参数
        for param in self.model.parameters():
            param.requires_grad = False
        self.embed_dim = self.model.config.hidden_size  # 768 for base

    def extract(self, y_np):
        """
        输入: 16kHz 音频 numpy 数组
        返回: 768维嵌入向量
        """
        # Wav2Vec2 处理器自动处理: resample, normalize, 返回 input_values
        inputs = self.processor(y_np, sampling_rate=SR,
                                return_tensors="pt", padding=True).input_values
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs, output_hidden_states=True).hidden_states  # 13层
            # 取第2,3,4层均值 (Cauzinille 2024: 早期层 > 最终层)
            outputs = torch.stack([outputs[i] for i in [2, 3, 4]], dim=0).mean(dim=0)
        # 时序均值池化
        embedding = outputs.mean(dim=1).squeeze().cpu().numpy()  # (768,)
        return embedding


# =========================================================
# 数据映射 + 按猫划分 (同 v5 逻辑)
# =========================================================
def build_mappings():
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
        test_cats |= test
        print(f'  {emo:20s}: {total:3d} → 训练{total - count:3d} 测试{count:3d}')
    train_cats = {cid for cid in cat_emotion_files if cid not in test_cats}
    return train_cats, test_cats


# =========================================================
# SVM 网格搜索 (基于 Wav2Vec2 嵌入)
# =========================================================
def train_svm(X_train, y_train, train_cat_labels):
    print('\n── SVM 网格搜索 ──')
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

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, train_cat_labels)):
        # 每折内独立缩放，避免验证集数据泄漏
        X_tr_raw = X_train[tr_idx]
        X_val_raw = X_train[val_idx]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_val = scaler.transform(X_val_raw)
        y_tr = y_train[tr_idx]
        y_val = y_train[val_idx]

        grid = GridSearchCV(SVC(probability=True, random_state=42, max_iter=5000),
                            param_grid, cv=min(3, len(np.unique(y_tr))),
                            scoring='accuracy', n_jobs=2)
        grid.fit(X_tr, y_tr)
        val_acc = grid.score(X_val, y_val)
        fold_scores.append(val_acc)
        print(f'  折{fold + 1}: val={val_acc:.3f}  params={grid.best_params_}')
        if val_acc > best_score:
            best_score = val_acc
            best_params = grid.best_params_

    print(f'  SVM CV均值: {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f}')
    print(f'  最佳参数: {best_params}')

    # 全量训练 (用全部数据拟合最终 scaler + 模型)
    scaler_final = StandardScaler()
    X_all_scaled = scaler_final.fit_transform(X_train)
    model = SVC(**best_params, probability=True, random_state=42, max_iter=5000)
    model.fit(X_all_scaled, y_train)
    return model, scaler_final, fold_scores


# =========================================================
# 随机森林网格搜索
# =========================================================
def train_rf(X_train, y_train, train_cat_labels):
    print('\n── 随机森林网格搜索 ──')
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample'],
    }

    gkf = GroupKFold(n_splits=K_FOLDS)
    fold_scores = []
    best_params = None
    best_score = 0

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, train_cat_labels)):
        X_tr = X_train[tr_idx]
        X_val = X_train[val_idx]
        y_tr = y_train[tr_idx]
        y_val = y_train[val_idx]

        grid = GridSearchCV(RandomForestClassifier(random_state=42),
                            param_grid, cv=min(3, len(np.unique(y_tr))),
                            scoring='accuracy', n_jobs=2)
        grid.fit(X_tr, y_tr)
        val_acc = grid.score(X_val, y_val)
        fold_scores.append(val_acc)
        print(f'  折{fold + 1}: val={val_acc:.3f}  params={grid.best_params_}')
        if val_acc > best_score:
            best_score = val_acc
            best_params = grid.best_params_

    print(f'  RF CV均值: {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f}')
    print(f'  最佳参数: {best_params}')

    model = RandomForestClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)
    return model, fold_scores


# =========================================================
# MLP 分类器 (在 Wav2Vec2 嵌入上做简单分类)
# =========================================================
class EmbeddingMLP(nn.Module):
    """
    在 Wav2Vec2 768维嵌入上的轻量 MLP.
    架构 768→64→10 (~50K 参数), 适合小数据集.
    """
    def __init__(self, input_dim=768, num_classes=10, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_val, y_val, input_dim, num_classes, device):
    """单折 MLP 训练 (warmup + 早停)."""
    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.LongTensor(y_train).to(device)
    X_v = torch.FloatTensor(X_val).to(device)
    y_v = torch.LongTensor(y_val).to(device)

    model = EmbeddingMLP(input_dim=input_dim, num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val = 0
    best_state = None
    patience = 20
    patience_count = 0
    warmup = WARMUP_EPOCHS
    lr_base = 1e-3

    for epoch in range(100):
        if epoch < warmup:
            for g in opt.param_groups:
                g['lr'] = lr_base * (epoch + 1) / warmup

        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(X_tr), y_tr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(X_v).argmax(dim=1)
            val_acc = (pred == y_v).float().mean().item()

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= patience:
            break

    model.load_state_dict(best_state)
    return model, best_val


# =========================================================
# 主流程
# =========================================================
def main():
    print('=' * 55)
    print('  译芯怡意 - Wav2Vec2 嵌入训练 v1')
    print('  特征: Wav2Vec2 (768维) | 分类器: SVM + RF + MLP')
    print('  前置: pip install transformers torch librosa scikit-learn')
    print('=' * 55)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n  设备: {device}')
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f'  GPU: {torch.cuda.get_device_name(0)} ({props.total_memory / 1024 ** 3:.1f}GB)')
    print()

    # ── 文件索引 ──
    file_index = build_file_index(EMOTION_DIR)

    # ── 数据映射 + 划分 ──
    file_to_cat, cat_emotion_files = build_mappings()
    print('按情绪 + 按猫划分 (20%猫做测试):')
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

    # ── Wav2Vec2 嵌入提取 ──
    print('\n── Wav2Vec2 嵌入提取 ──')
    all_files = train_files + test_files
    cache_npy = OUTPUT_DIR / 'embedding_cache.npy'
    cache_fnames = OUTPUT_DIR / 'embedding_cache_fnames.txt'

    cached = False
    if cache_npy.exists() and cache_fnames.exists():
        cached_fnames = cache_fnames.read_text().splitlines()
        if cached_fnames == all_files:
            print(f'  [缓存命中] 加载 {cache_npy}')
            X_all = np.load(str(cache_npy))
            embed_dim = X_all.shape[1]
            X_train = X_all[:len(train_files)]
            X_test = X_all[len(train_files):]
            print(f'  嵌入维度: {embed_dim}')
            print(f'  X_train: {X_train.shape}, X_test: {X_test.shape}')
            cached = True

    if not cached:
        extractor = Wav2Vec2EmbeddingExtractor(device=device)
        all_embeddings = {}
        for i, fname in enumerate(all_files):
            y = load_audio(fname, file_index)
            y = normalize(y)
            y = align_to_target(y)
            emb = extractor.extract(y)
            all_embeddings[fname] = emb
            if (i + 1) % 50 == 0 or (i + 1) == len(all_files):
                print(f'  嵌入: {i + 1}/{len(all_files)}')

        X_train = np.array([all_embeddings[f] for f in train_files])
        X_test  = np.array([all_embeddings[f] for f in test_files])
        embed_dim = X_train.shape[1]
        print(f'  嵌入维度: {embed_dim}')
        print(f'  X_train: {X_train.shape}, X_test: {X_test.shape}')

        # 保存缓存
        X_all = np.vstack([X_train, X_test])
        np.save(str(cache_npy), X_all)
        cache_fnames.write_text('\n'.join(all_files))
        print(f'  [缓存保存] {cache_npy}')

        del extractor, all_embeddings

    # ── 可选: PCA 降维 (768 → 128, 保持核心信息 + 加速 SVM/RF) ──
    # 注: Wav2Vec2 768维对于当前数据量可能偏大, PCA 可去噪
    print('\n── 可选: PCA 降维 (768→128) ──')
    pca = PCA(n_components=min(128, len(X_train) - 1))
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f'  PCA降维后: X_train {X_train_pca.shape}, 解释方差: {pca.explained_variance_ratio_.sum():.3f}')

    # ========================================================
    # 1. SVM (在原始嵌入上做)
    # ========================================================
    print('\n' + '=' * 50)
    print('  SVM 分类器')
    print('=' * 50)
    svm_model, svm_scaler, svm_fold_scores = train_svm(X_train, y_train, train_cat_labels)
    X_test_scaled = svm_scaler.transform(X_test)
    svm_test = svm_model.score(X_test_scaled, y_test)

    print('\n  SVM 逐类表现:')
    print(classification_report(y_test, svm_model.predict(X_test_scaled),
                                 target_names=le.classes_, digits=3))

    # ========================================================
    # 2. 随机森林 (在 PCA 嵌入上做, 高维数据 RF 更友好)
    # ========================================================
    print('\n' + '=' * 50)
    print('  随机森林分类器')
    print('=' * 50)
    rf_model, rf_fold_scores = train_rf(X_train_pca, y_train, train_cat_labels)
    rf_test = rf_model.score(X_test_pca, y_test)
    print(f'  RF 测试: {rf_test:.1%}')
    print('\n  RF 逐类表现:')
    print(classification_report(y_test, rf_model.predict(X_test_pca),
                                 target_names=le.classes_, digits=3))

    # ========================================================
    # 3. MLP (在原始嵌入上, K折CV)
    # ========================================================
    print('\n' + '=' * 50)
    print('  MLP 分类器 (K折CV)')
    print('=' * 50)
    gkf = GroupKFold(n_splits=K_FOLDS)
    mlp_fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, train_cat_labels)):
        X_tr = X_train[tr_idx]
        X_vl = X_train[val_idx]
        y_tr = y_train[tr_idx]
        y_vl = y_train[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_vl_s = scaler.transform(X_vl)

        model, val_acc = train_mlp(X_tr_s, y_tr, X_vl_s, y_vl,
                                    input_dim=embed_dim, num_classes=n_classes,
                                    device=device)
        mlp_fold_scores.append(val_acc)
        print(f'  折{fold + 1}: val={val_acc:.3f}')

    print(f'  MLP CV均值: {np.mean(mlp_fold_scores):.3f} ± {np.std(mlp_fold_scores):.3f}')

    # MLP 全量训练 (所有训练数据)
    # 注: K折CV 已验证架构有效, 全量训练直接用全部数据, 不需留验证集
    #     避免 StandardScaler 数据泄漏 + n_splits 硬编码问题
    print('\n  MLP 全量训练...')
    scaler_final = StandardScaler()
    X_train_all = scaler_final.fit_transform(X_train)
    X_test_final = scaler_final.transform(X_test)

    mlp_final = EmbeddingMLP(input_dim=embed_dim, num_classes=n_classes).to(device)
    opt = torch.optim.AdamW(mlp_final.parameters(), lr=1e-3, weight_decay=1e-4)
    X_t = torch.FloatTensor(X_train_all).to(device)
    y_t = torch.LongTensor(y_train).to(device)

    for ep in range(WARMUP_EPOCHS + 20):
        lr = 1e-3 * min((ep + 1) / WARMUP_EPOCHS, 1.0)
        for g in opt.param_groups:
            g['lr'] = lr
        mlp_final.train()
        opt.zero_grad()
        loss = F.cross_entropy(mlp_final(X_t), y_t)
        loss.backward()
        opt.step()

    mlp_final.eval()
    with torch.no_grad():
        pred = mlp_final(torch.FloatTensor(X_test_final).to(device)).argmax(dim=1)
        mlp_test = (pred == torch.LongTensor(y_test).to(device)).float().mean().item()
    print(f'  MLP 测试: {mlp_test:.1%}')
    print('\n  MLP 逐类表现:')
    mlp_preds = mlp_final(torch.FloatTensor(X_test_final).to(device)).argmax(dim=1).cpu().numpy()
    print(classification_report(y_test, mlp_preds,
                                 target_names=le.classes_, digits=3))

    # ========================================================
    # 结果汇总
    # ========================================================
    scores = {
        'svm': {'cv': [float(s) for s in svm_fold_scores], 'test': float(svm_test)},
        'rf':  {'cv': [float(s) for s in rf_fold_scores],  'test': float(rf_test)},
        'mlp': {'cv': [float(s) for s in mlp_fold_scores], 'test': float(mlp_test)},
    }

    print('\n' + '=' * 50)
    print('  最终结果')
    print('=' * 50)
    for name, s in scores.items():
        cv_mean = sum(s['cv']) / len(s['cv'])
        print(f'  {name.upper():5s}: CV={cv_mean:.1%}  |  测试={s["test"]:.1%}')

    # 用 CV 均值选最优（测试集仅做最终报告，不做模型选择）
    best_model_name = max(scores, key=lambda k: sum(scores[k]['cv']) / len(scores[k]['cv']))
    print(f'\n  最佳 (基于CV): {best_model_name.upper()} (测试={scores[best_model_name]["test"]:.1%})')
    winner = best_model_name

    # ── 保存模型 ──
    torch.save({'mlp_state_dict': mlp_final.state_dict(),
                 'scaler': scaler_final,
                 'input_dim': embed_dim,
                 'num_classes': n_classes,
                 'label_encoder': le},
                OUTPUT_DIR / f'mlp_{TS}.pth')

    with open(OUTPUT_DIR / f'svm_rf_{TS}.pkl', 'wb') as f:
        pickle.dump({
            'svm_model': svm_model,
            'svm_scaler': svm_scaler,
            'rf_model': rf_model,
            'label_encoder': le,
            'pca': pca,
        }, f)

    results = {
        'version': 'v1_wav2vec2',
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'wav2vec2_model': WAV2VEC2_MODEL,
        'n_train': len(train_files), 'n_test': len(test_files),
        'n_train_cats': len(train_cats), 'n_test_cats': len(test_cats),
        'embed_dim': embed_dim,
        'pca_components': pca.n_components_,
        'pca_explained_var': float(pca.explained_variance_ratio_.sum()),
        'svm': {'cv': [float(s) for s in scores['svm']['cv']], 'test': float(svm_test)},
        'rf':  {'cv': [float(s) for s in scores['rf']['cv']],  'test': float(rf_test)},
        'mlp': {'cv': [float(s) for s in mlp_fold_scores], 'test': float(mlp_test)},
        'best_model': winner,
        'classes': le.classes_.tolist(),
        'preprocessing': {
            'sample_rate': SR,
            'target_len_samples': TARGET_LEN,
            'target_duration_sec': DURATION,
            'normalize': 'peak',
            'alignment': 'center_crop_or_reflect_pad',
        },
        'timestamp': datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / f'results_wav2vec2_{TS}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f'\n  保存路径: {OUTPUT_DIR}')
    print('  结果: results_wav2vec2.json')


if __name__ == '__main__':
    main()

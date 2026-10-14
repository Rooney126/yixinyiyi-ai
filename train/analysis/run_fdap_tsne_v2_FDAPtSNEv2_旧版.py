"""
FDAP t-SNE v2: 修复性能 + 缓存 + 归一化问题
# FDAP t-SNE v2（旧版，已废弃）
"""

import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from collections import Counter
from datetime import datetime
from pathlib import Path

import librosa
import matplotlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 配置 ──
SR = 16000
CROP_SAMPLES = int(SR * 5.0)  # 5秒固定
N_MELS = 96
HOP_LENGTH = 512
SEED = 42
PERP = 60
np.random.seed(SEED)

BASE = Path(r'/path/to/dataset')
DATA_DIR = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
MODEL_PATH = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X/model_best.hdf5'
TS = datetime.now().strftime('%Y%m%d_%H%M%S')
CACHE_DIR = BASE / '模型代码/fdap_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_KEY = 'fdap_feats_v2_fixedsr'  # 若输入参数变了就改这个
OUTPUT_PNG = BASE / f'模型代码/t-SNE_FDAP_v2_{TS}.png'
OUTPUT_JSON = BASE / f'模型代码/t-SNE_FDAP_metrics_{TS}.json'

LABELS_10 = ['Angry','Defence','Fighting','Happy','HuntingMind',
             'Mating','MotherCall','Paining','Resting','Warning']
COLORS = {
    'Angry':'#d62728', 'Defence':'#e377c2', 'Fighting':'#ff7f0e', 'Warning':'#bcbd22',
    'Happy':'#2ca02c', 'HuntingMind':'#17becf', 'Mating':'#9467bd',
    'MotherCall':'#1f77b4', 'Paining':'#8c564b', 'Resting':'#7f7f7f',
}

print('═'*60)
print('  FDAP t-SNE v2 — 修复版')
print('═'*60)

# ═══════════════════════════════════
# 1. 加载音频 + Mel 频谱
# ═══════════════════════════════════
print('\n[1/4] 加载音频...')
samples = []
for label in LABELS_10:
    d = DATA_DIR / label
    if not d.exists():
        continue
    for fp in sorted(d.glob('*.mp3'))[:200]:
        try:
            y, _ = librosa.load(str(fp), sr=SR, mono=True)
            # 固定长度
            if len(y) > CROP_SAMPLES:
                start = (len(y) - CROP_SAMPLES) // 2
                y = y[start:start + CROP_SAMPLES]
            else:
                pad = CROP_SAMPLES - len(y)
                y = np.pad(y, (pad // 2, pad - pad // 2))
            # Mel (同一参数, batch后统一 normalize)
            mel = librosa.feature.melspectrogram(
                y=y, sr=SR, n_fft=1024, hop_length=HOP_LENGTH, n_mels=N_MELS)
            samples.append({'waveform': y, 'mel': mel, 'label': label, 'file': str(fp)})
        except Exception as e:
            print(f'  [FAIL] {fp.name}: {e}')

n = len(samples)
labels_all = [s['label'] for s in samples]
print(f'  共 {n} 条')
for l, c in Counter(labels_all).most_common():
    print(f'    {l:<12}: {c:4d}')

# ═══════════════════════════════════
# 2. 缓存检查
# ═══════════════════════════════════
cache_fdap = CACHE_DIR / f'{CACHE_KEY}_fdap.npy'
cache_gap = CACHE_DIR / f'{CACHE_KEY}_gap.npy'
cache_labels = CACHE_DIR / f'{CACHE_KEY}_labels.npy'

weights_loaded = False  # 缓存命中时预设

if cache_fdap.exists() and cache_gap.exists() and cache_labels.exists():
    print('\n[2/4] ✅ 检测到缓存, 跳过特征提取')
    fdap_feats = np.load(cache_fdap)
    gap_feats = np.load(cache_gap)
    labels_loaded = np.load(cache_labels)
    print(f'  FDAP: {fdap_feats.shape} | GAP: {gap_feats.shape}')
else:
    print('\n[2/4] 加载 CNN + 特征提取...')
    import tensorflow as tf
    from tensorflow.keras import layers, models

    def build_cnn():
        inp = layers.Input(shape=(None, N_MELS, 1))
        # 手动归一化 → 不需要 Normalization 层
        blocks = []
        x = None  # 首轮用 inp，后续轮用上一轮输出
        for pool_size in [(2,4),(3,4),(2,5),(2,4),(4,4)]:
            x = layers.Conv2D(32, (3,3), padding='same')(x if blocks else inp)
            x = layers.BatchNormalization()(x)
            x = layers.ELU()(x)
            x = layers.MaxPooling2D(pool_size)(x)
            blocks.append(x)
        return models.Model(inputs=inp, outputs=blocks)

    cnn = build_cnn()
    FDAP_SEGMENTS = [1, 1, 4, 4, 3]
    weights_loaded = False

    try:
        full = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        for i in range(1, 6):
            cnn.get_layer(f'conv2d_{i}').set_weights(
                full.get_layer(f'conv2d_{i}').get_weights())
            cnn.get_layer(f'batch_normalization_{i}').set_weights(
                full.get_layer(f'batch_normalization_{i}').get_weights())
        weights_loaded = True
        del full
        print('  ✅ 加载 model_best.hdf5 权重')
    except Exception as e:
        print(f'  ⚠️ 加载失败 ({e}), 随机权重')

    # ── batch 提 Mel ──
    # 所有音频统一为 (CROP_SAMPLES,) → Mel shape 固定
    # 先全部转 Mel_db, 然后用全局 ref 归一化
    all_mels = []
    for s in samples:
        mel = s['mel']
        all_mels.append(mel)

    # 全局 max 归一化
    all_mels_stack = np.stack(all_mels)  # (N, 96, T)
    global_max = all_mels_stack.max()
    all_mels_db = librosa.power_to_db(all_mels_stack, ref=global_max)  # 统一 ref

    # ── batch CNN predict ──
    # 先转置: (N, 96, T) → (N, T, 96, 1)
    BATCH_CNN = 64
    fdap_feats_list = []
    gap_feats_list = []

    for i in range(0, n, BATCH_CNN):
        batch_mels = all_mels_db[i:i+BATCH_CNN]  # (B, 96, T)
        # 转置并加通道维: (B, T, 96, 1)
        batch_x = np.transpose(batch_mels, (0, 2, 1))[..., np.newaxis].astype(np.float32)
        outputs = cnn.predict(batch_x, verbose=0)  # list of 5 tensors, each (B, H, W, 32)

        B = batch_x.shape[0]
        for b in range(B):
            fdap_vecs = []
            gap_vecs = []
            for idx, out in enumerate(outputs):
                feat = out[b]  # (H_freq, W_time, 32)
                H = feat.shape[0]
                n_seg = FDAP_SEGMENTS[idx]
                seg_len = max(H // n_seg, 1)

                for seg_i in range(n_seg):
                    s_start = seg_i * seg_len
                    s_end = s_start + seg_len if seg_i < n_seg - 1 else H
                    seg = feat[s_start:s_end, :, :]
                    fdap_vecs.append(np.mean(seg, axis=(0, 1)))
                    fdap_vecs.append(np.std(seg, axis=(0, 1)))

                gap_vecs.append(np.mean(feat, axis=(0, 1)))

            fdap_feats_list.append(np.concatenate(fdap_vecs))
            gap_feats_list.append(np.concatenate(gap_vecs))

        if (i // BATCH_CNN) % 5 == 0:
            print(f'  CNN: {min(i+BATCH_CNN, n)}/{n}')

    fdap_feats = np.stack(fdap_feats_list)
    gap_feats = np.stack(gap_feats_list)
    print(f'  FDAP: {fdap_feats.shape} | GAP: {gap_feats.shape}')

    # ── 缓存 ──
    np.save(cache_fdap, fdap_feats)
    np.save(cache_gap, gap_feats)
    np.save(cache_labels, np.array(labels_all))
    print(f'  ✅ 缓存保存: {CACHE_DIR}/')

# ═══════════════════════════════════
# 3. t-SNE 可视化
# ═══════════════════════════════════
print('\n[3/4] t-SNE...')
pca50 = PCA(n_components=50, random_state=SEED)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, feats, title in [
    (axes[0], fdap_feats, f'FDAP ({sum(FDAP_SEGMENTS)}频段×均值+标准差)'),
    (axes[1], gap_feats, 'GAP (全局均值)')
]:
    X = pca50.fit_transform(feats)
    coords = TSNE(n_components=2, perplexity=PERP, random_state=SEED).fit_transform(X)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    for label in LABELS_10:
        mask = np.array([l == label for l in labels_all])
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1], c=COLORS[label],
                   label=label, alpha=0.4, s=5, edgecolors='none')

handles, leg_labels = axes[1].get_legend_handles_labels()
fig.legend(handles, leg_labels, loc='lower center', ncol=10,
           fontsize=7, bbox_to_anchor=(0.5, -0.02))
plt.suptitle(f'Pandeya CNN 特征空间 | {"✅ model_best权重" if weights_loaded else "⚠️ 随机权重"}',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(str(OUTPUT_PNG), dpi=150, bbox_inches='tight')
print(f'\n  ✅ 保存: {OUTPUT_PNG}')

# ═══════════════════════════════════
# 4. 数值指标
# ═══════════════════════════════════
print('\n[4/4] 聚类指标...')
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_int = le.fit_transform(labels_all)

fdap_sil = silhouette_score(fdap_feats, y_int)
gap_sil = silhouette_score(gap_feats, y_int)
print('  轮廓系数 (越高越好):')
print(f'    FDAP: {fdap_sil:.4f}')
print(f'    GAP:  {gap_sil:.4f}')

metrics = {
    'fdap_silhouette': round(float(fdap_sil), 4),
    'gap_silhouette': round(float(gap_sil), 4),
    'fdap_dims': fdap_feats.shape[1],
    'gap_dims': gap_feats.shape[1],
    'n_samples': n,
    'weights_loaded': weights_loaded,
}
with open(OUTPUT_JSON, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f'  ✅ 指标: {OUTPUT_JSON}')

"""
FDAP t-SNE v3: 正确加载 model_best.hdf5，从中层提取 Conv 特征图
输入: 原始波形 (model_best 自带 Kapre Melgram + Normalization2D)
输出: FDAP vs GAP 特征空间的 t-SNE 对比
# FDAP t-SNE v3（旧版，已废弃）
"""

import os, sys, json, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from collections import Counter
from pathlib import Path
from datetime import datetime
import numpy as np
import librosa
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 配置 ──
SR = 16000
CROP_SAMPLES = int(SR * 5.0)  # 5秒, 对应 model_best 的固定输入
SEED = 42
PERP = 60
MAX_PER_CLASS = None  # 不限量, 取全部原始音频
# 只取原始音频 (排除 _aug1 增强副本)
ORIGINAL_ONLY = True
np.random.seed(SEED)

BASE = Path(r'/path/to/dataset')
DATA_DIR = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
MODEL_PATH = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X/model_best.hdf5'
TS = datetime.now().strftime('%Y%m%d_%H%M%S')
CACHE_DIR = BASE / '模型代码/fdap_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LABELS_10 = ['Angry','Defence','Fighting','Happy','HuntingMind',
             'Mating','MotherCall','Paining','Resting','Warning']
COLORS = {
    'Angry':'#d62728', 'Defence':'#e377c2', 'Fighting':'#ff7f0e', 'Warning':'#bcbd22',
    'Happy':'#2ca02c', 'HuntingMind':'#17becf', 'Mating':'#9467bd',
    'MotherCall':'#1f77b4', 'Paining':'#8c564b', 'Resting':'#7f7f7f',
}

print('═'*70)
print('  FDAP t-SNE v3 — 正确加载 model_best')
print('═'*70)

# ═══════════════════════════════════
# 1. 加载音频 (原始波形, 不提前算 Mel)
# ═══════════════════════════════════
print('\n[1/4] 加载音频...')
samples = []
for label in LABELS_10:
    d = DATA_DIR / label
    if not d.exists():
        continue
    files = sorted(d.glob('*.mp3'))
    if ORIGINAL_ONLY:
        files = [f for f in files if '_aug1' not in f.stem]
    if MAX_PER_CLASS:
        files = files[:MAX_PER_CLASS]
    for fp in files:
        try:
            y, _ = librosa.load(str(fp), sr=SR, mono=True)
            # 固定 5 秒
            if len(y) > CROP_SAMPLES:
                start = (len(y) - CROP_SAMPLES) // 2
                y = y[start:start + CROP_SAMPLES]
            else:
                pad = CROP_SAMPLES - len(y)
                y = np.pad(y, (pad // 2, pad - pad // 2))
            samples.append({'raw': y, 'label': label})
        except Exception as e:
            print(f'  [FAIL] {fp.name}: {e}')

n = len(samples)
if n == 0:
    print('❌ 没有加载到任何音频，退出')
    sys.exit(1)

labels_all = [s['label'] for s in samples]
print(f'  共 {n} 条')
for l, c in Counter(labels_all).most_common():
    print(f'    {l:<12}: {c:4d}')

# ═══════════════════════════════════
# 2. 加载 model_best 完整模型
# ═══════════════════════════════════
print('\n[2/4] 加载 model_best.hdf5...')
try:
    import tensorflow as tf
except ImportError:
    print('  ⚠️ TensorFlow 未安装, 自动安装...')
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'tensorflow', 'tf-keras', '-q'], check=True)
    import tensorflow as tf
from tensorflow.keras import layers, models

# MP 层索引 (model_best Sequential 中第 5, 9, 13, 17, 21 层)
MP_LAYER_INDICES = [5, 9, 13, 17, 21]
# FDAP 分段数 (对应论文: 1,1,4,4,3)
# 各层输出 freq 维度: 48, 16, 8, 4, 1
FDAP_SEGMENTS = [1, 1, 4, 4, 3]

weights_loaded = False
try:
    # 尝试加载完整模型 (含 Kapre Melgram + Normalization2D)
    full_model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    
    # 创建子模型: 输入→5个MP层输出
    mp_outputs = [full_model.layers[i].output for i in MP_LAYER_INDICES]
    extractor = models.Model(inputs=full_model.input, outputs=mp_outputs)
    
    weights_loaded = True
    print(f'  ✅ 完整模型加载成功 ({len(extractor.layers)} 层)')
    print(f'  输入 shape: {extractor.input_shape}')
    print(f'  5个MP输出: {[o.shape for o in extractor.outputs]}')
    
    # 释放 full_model (提取器已足够)
    # extractor 保留了所有层的引用
except Exception as e:
    print(f'  ⚠️ 加载完整模型失败: {e}')
    print(f'  → 尝试安装 kapre...')
    
    try:
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'kapre', '-q'], check=True)
        # 重试
        full_model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        mp_outputs = [full_model.layers[i].output for i in MP_LAYER_INDICES]
        extractor = models.Model(inputs=full_model.input, outputs=mp_outputs)
        weights_loaded = True
        print(f'  ✅ 安装 Kapre 后加载成功')
    except Exception as e2:
        print(f'  ❌ 仍失败: {e2}')
        print(f'  → 使用重建架构 + 手动提取权重 (无 Normalization2D)')
        
        # 重建架构 (不含 Kapre 和 Normalization2D)
        # 注意: 这会有输入分布偏移, 特征质量不可保证
        inp = layers.Input(shape=(CROP_SAMPLES,))
        x = layers.Reshape((CROP_SAMPLES, 1))(inp)
        
        # 手动 Mel (静默, normalize by global max)
        # 但我们直接喂 raw → Kapre 是做 Mel 的, 没有 Kapre 我们就要自己做
        
        # 最终: 用 librosa 算 Mel, 用重建的 CNN 算 FDAP
        # 这种模式下 Normalization2D 权重不能用了
        print('  → 回退: librosa Mel → 重建 CNN (仅 Conv2D 权重)')

# ═══════════════════════════════════
# 3. FDAP vs GAP 特征提取
# ═══════════════════════════════════
print('\n[3/4] 特征提取...')

BATCH = 64
CACHE_KEY = 'fdap_v3_full' if weights_loaded else 'fdap_v3_fallback'
cache_fdap = CACHE_DIR / f'{CACHE_KEY}_fdap_{n}x{MAX_PER_CLASS}.npy'
cache_gap = CACHE_DIR / f'{CACHE_KEY}_gap_{n}.npy'
cache_labels = CACHE_DIR / f'{CACHE_KEY}_labels_{n}.npy'

if cache_fdap.exists() and cache_gap.exists():
    print('  ✅ 检测到缓存, 跳过')
    fdap_feats = np.load(cache_fdap)
    gap_feats = np.load(cache_gap)
else:
    if weights_loaded:
        # ── 方法A: 通过完整模型 → 原始波形直接输入 ──
        print('  通过完整模型 (波形→Mel→Conv→MP)')
        fdap_feats_list = []
        gap_feats_list = []
        
        # model_best 的 Kapre Melgram 期望输入 shape (B, 1, T) 
        # 即 (batch, channels, time) — 注意 channels 在第2维!
        for i in range(0, n, BATCH):
            batch_raw = np.array([s['raw'] for s in samples[i:i+BATCH]])  # (B, T)
            batch_x = np.expand_dims(batch_raw, axis=1)  # (B, 1, T)
            
            mp_outputs = extractor.predict(batch_x, verbose=0)
            # mp_outputs: list of 5 tensors, each (B, H_freq, W_time, 32)
            
            B = batch_x.shape[0]
            for b in range(B):
                fdap_vecs = []
                gap_vecs = []
                for idx, feat in enumerate(mp_outputs):
                    f = feat[b]  # (H, W, 32)
                    H = f.shape[0]
                    n_seg = FDAP_SEGMENTS[idx]
                    seg_len = max(H // n_seg, 1)
                    
                    for seg_i in range(n_seg):
                        s_start = seg_i * seg_len
                        s_end = s_start + seg_len if seg_i < n_seg - 1 else H
                        seg = f[s_start:s_end, :, :]
                        fdap_vecs.append(np.mean(seg, axis=(0, 1)))
                        fdap_vecs.append(np.std(seg, axis=(0, 1)))
                    
                    gap_vecs.append(np.mean(f, axis=(0, 1)))
                
                fdap_feats_list.append(np.concatenate(fdap_vecs))
                gap_feats_list.append(np.concatenate(gap_vecs))
            
            if (i // BATCH) % 5 == 0:
                print(f'  进度: {min(i+BATCH, n)}/{n}')
        
        fdap_feats = np.stack(fdap_feats_list)
        gap_feats = np.stack(gap_feats_list)
    
    else:
        # ── 方法B: 回退方案, librosa Mel → 重建 CNN ──
        print('  回退方案: librosa Mel → 重建 CNN')
        
        # 重建 CNN (包含 Normalization2D 但用默认参数)
        inp = layers.Input(shape=(None, 96, 1))
        x = layers.Normalization(axis=-1)(inp)
        blocks = []
        for ps in [(2,4),(3,4),(2,5),(2,4),(4,4)]:
            x = layers.Conv2D(32, (3,3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ELU()(x)
            x = layers.MaxPooling2D(ps)(x)
            blocks.append(x)
        fallback_cnn = models.Model(inputs=inp, outputs=blocks)
        
        # 尝试加载 Conv2D+BN 权重 (不加载 Normalization)
        try:
            full = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
            for i in range(1, 6):
                fallback_cnn.get_layer(f'conv2d_{i}').set_weights(
                    full.get_layer(f'conv2d_{i}').get_weights())
                fallback_cnn.get_layer(f'batch_normalization_{i}').set_weights(
                    full.get_layer(f'batch_normalization_{i}').get_weights())
            del full
            print('  ✅ 已加载 Conv2D+BN 权重')
        except:
            print('  ⚠️ 使用随机权重')
        
        # 手动 Mel
        fdap_feats_list = []
        gap_feats_list = []
        
        for i in range(0, n, BATCH):
            batch_mels = []
            for s in samples[i:i+BATCH]:
                mel = librosa.feature.melspectrogram(y=s['raw'], sr=SR, n_fft=1024, hop_length=512, n_mels=96)
                batch_mels.append(mel)
            # 统一归一化
            all_mels = np.stack(batch_mels)
            global_max = all_mels.max()
            all_mels_db = librosa.power_to_db(all_mels, ref=global_max)  # (B, 96, T)
            batch_x = np.transpose(all_mels_db, (0, 2, 1))[..., np.newaxis].astype(np.float32)
            
            outputs = fallback_cnn.predict(batch_x, verbose=0)
            
            B = batch_x.shape[0]
            for b in range(B):
                fdap_vecs = []
                gap_vecs = []
                for idx, feat in enumerate(outputs):
                    f = feat[b]
                    H = f.shape[0]
                    n_seg = FDAP_SEGMENTS[idx]
                    seg_len = max(H // n_seg, 1)
                    
                    for seg_i in range(n_seg):
                        s_start = seg_i * seg_len
                        s_end = s_start + seg_len if seg_i < n_seg - 1 else H
                        seg = f[s_start:s_end, :, :]
                        fdap_vecs.append(np.mean(seg, axis=(0, 1)))
                        fdap_vecs.append(np.std(seg, axis=(0, 1)))
                    
                    gap_vecs.append(np.mean(f, axis=(0, 1)))
                
                fdap_feats_list.append(np.concatenate(fdap_vecs))
                gap_feats_list.append(np.concatenate(gap_vecs))
            
            if (i // BATCH) % 5 == 0:
                print(f'  进度: {min(i+BATCH, n)}/{n}')
        
        fdap_feats = np.stack(fdap_feats_list)
        gap_feats = np.stack(gap_feats_list)
    
    print(f'  FDAP: {fdap_feats.shape} | GAP: {gap_feats.shape}')
    np.save(cache_fdap, fdap_feats)
    np.save(cache_gap, gap_feats)
    np.save(cache_labels, np.array(labels_all))
    print(f'  ✅ 缓存: {CACHE_DIR}/')

# ═══════════════════════════════════
# 4. t-SNE + 指标
# ═══════════════════════════════════
print('\n[4/4] t-SNE + 聚类指标...')

pca50 = PCA(n_components=50, random_state=SEED)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, feats, title in [
    (axes[0], fdap_feats, 'FDAP (频率分段池化)'),
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

mode_str = '✅ model_best完整加载' if weights_loaded else '⚠️ librosa Mel + 仅Conv2D权重'
plt.suptitle(f'Pandeya CNN 特征空间 | {mode_str}',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout(rect=[0, 0.08, 1, 1])

out_png = BASE / f'模型代码/t-SNE_FDAP_v3_{TS}.png'
plt.savefig(str(out_png), dpi=150, bbox_inches='tight')
print(f'\n  ✅ 图: {out_png}')

# 数值
le = LabelEncoder()
y_int = le.fit_transform(labels_all)
fdap_sil = silhouette_score(fdap_feats, y_int)
gap_sil = silhouette_score(gap_feats, y_int)
print(f'\n  轮廓系数:')
print(f'    FDAP: {fdap_sil:.4f}')
print(f'    GAP:  {gap_sil:.4f}')
print(f'    差距: {fdap_sil - gap_sil:+.4f}')

out_json = BASE / f'模型代码/t-SNE_FDAP_metrics_{TS}.json'
with open(out_json, 'w') as f:
    json.dump({
        'mode': 'full_model' if weights_loaded else 'fallback',
        'fdap_silhouette': round(float(fdap_sil), 4),
        'gap_silhouette': round(float(gap_sil), 4),
        'silhouette_diff': round(float(fdap_sil - gap_sil), 4),
        'n_samples': n,
        'fdap_dims': int(fdap_feats.shape[1]),
        'gap_dims': int(gap_feats.shape[1]),
    }, f, indent=2)
print(f'  ✅ 指标: {out_json}')
print(f'\n  对比已有的 HuBERT t-SNE 图')

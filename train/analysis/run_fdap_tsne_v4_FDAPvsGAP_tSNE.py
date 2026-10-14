"""
FDAP t-SNE v4: h5py + PyTorch，不依赖 TensorFlow
读取 model_best.hdf5 权重 → PyTorch CNN → FDAP vs GAP t-SNE
# FDAP t-SNE v4：Pandeya CNN + FDAP/GAP 聚类可视化
"""

import json
import sys
import warnings

import h5py

warnings.filterwarnings('ignore')
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from collections import Counter
from datetime import datetime
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from torch import nn

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 配置 ──
SR = 12000  # model_best 用的 sr=12000
CROP_SAMPLES = int(SR * 5.0)
SEED = 42
PERP = 60
MAX_PER_CLASS = None
ORIGINAL_ONLY = True
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE = Path(r'/path/to/dataset')
DATA_DIR = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
MODEL_PATH = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X/model_best.hdf5'
TS = datetime.now().strftime('%Y%m%d_%H%M%S')
CACHE_DIR = BASE / '模型代码/fdap_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LABELS_10 = ['Angry','Defence','Fighting','Happy','HuntingMind',
             'Mating','MotherCall','Paining','Resting','Warning']
COLORS = {
    'Angry':'#d62728', 'Defence':'#e377c2', 'Fighting':'#ff7f0e', 'Warning':'#bcbd22',
    'Happy':'#2ca02c', 'HuntingMind':'#17becf', 'Mating':'#9467bd',
    'MotherCall':'#1f77b4', 'Paining':'#8c564b', 'Resting':'#7f7f7f',
}

# Kapre melgram 参数 (从 hdf5 读出)
MEL_CONFIG = dict(sr=12000, n_fft=512, hop_length=256, n_mels=96,
                  fmin=0.0, fmax=6000)

# 模型参数
FDAP_SEGMENTS = [1, 1, 4, 4, 3]  # 各 MP 层 freq 维: 48, 16, 8, 4, 1

print('═'*70)
print('  FDAP t-SNE v4 — PyTorch + h5py（无需 TensorFlow）')
print(f'  设备: {DEVICE}')
print('═'*70)

# ═══════════════════════════════════
# 1. 加载音频
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
    print('❌ 没有音频，退出')
    sys.exit(1)
labels_all = [s['label'] for s in samples]
print(f'  共 {n} 条')
for l, c in Counter(labels_all).most_common():
    print(f'    {l:<12}: {c:4d}')

# ═══════════════════════════════════
# 2. 读 hdf5 权重 + 重建 PyTorch CNN
# ═══════════════════════════════════
print('\n[2/4] 加载权重 + 重建 CNN...')

# ── 读取权重的辅助函数 ──
def read_h5_weight(f, name):
    """从 hdf5 读 Keras 权重, 允许不同路径变体"""
    variants = [
        f'model_weights/{name}/{name}/kernel:0',
        f'model_weights/{name}/{name}/bias:0',
        f'model_weights/{name}/{name}/gamma:0',
        f'model_weights/{name}/{name}/beta:0',
        f'model_weights/{name}/{name}/moving_mean:0',
        f'model_weights/{name}/{name}/moving_variance:0',
    ]
    result = {}
    for v in variants:
        try:
            result[v.split('/')[-1]] = f[v][()]
        except:
            pass
    return result

f_h5 = h5py.File(str(MODEL_PATH), 'r')

# 读 Conv2D: kernel shape (3,3,in_ch,out_ch), bias shape (out_ch,)
conv_weights = []
bn_weights = []
for i in range(1, 6):
    kw = read_h5_weight(f_h5, f'conv2d_{i}')
    bw = read_h5_weight(f_h5, f'batch_normalization_{i}')
    conv_weights.append(kw)
    bn_weights.append(bw)
    print(f'  conv2d_{i}: kernel {kw['kernel:0'].shape}, BN gamma {bw['gamma:0'].shape}')

f_h5.close()
print('  ✅ 权重读取完成')

# ── 重建 CNN ──
class SafeMaxPool2d(nn.Module):
    """MaxPool2d 但小维度过小时输出至少 1 (兼容 Keras 行为)"""
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
    
    def forward(self, x):
        # x: (N, C, H, W)
        _, _, H, W = x.shape
        kH, kW = self.kernel_size
        # 只有当维度 >= kernel 时才正常池化, 否则直接过
        pool_h = kH if H >= kH else 1
        pool_w = kW if W >= kW else 1
        if pool_h == 1 and pool_w == 1:
            return x
        return F.max_pool2d(x, (pool_h, pool_w))


class PandeyaCNN(nn.Module):
    """Pandeya CNN (5× Conv2D 32ch), 输出各 MP 层特征图"""
    def __init__(self, conv_weights, bn_weights):
        super().__init__()
        self.blocks = nn.ModuleList()
        
        for i in range(5):
            kw = conv_weights[i]
            bw = bn_weights[i]
            # kernel: (3,3,in_ch,out_ch) Keras → (out_ch,in_ch,3,3) PyTorch
            w = torch.from_numpy(kw['kernel:0']).permute(3, 2, 0, 1).float()
            b = torch.from_numpy(kw['bias:0']).float()
            in_ch = w.shape[1]
            out_ch = w.shape[0]
            
            conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            conv.weight.data = w
            conv.bias.data = b
            
            bn = nn.BatchNorm2d(out_ch)
            bn.weight.data = torch.from_numpy(bw['gamma:0']).float()
            bn.bias.data = torch.from_numpy(bw['beta:0']).float()
            bn.running_mean.data = torch.from_numpy(bw['moving_mean:0']).float()
            bn.running_var.data = torch.from_numpy(bw['moving_variance:0']).float()
            
            pool_sizes = [(2,4),(3,4),(2,5),(2,4),(4,4)]
            # SafeMaxPool: Keras 兼容, 当输入维 < kernel_size 时输出至少 1
            self.blocks.append(nn.Sequential(conv, bn, nn.ELU(), SafeMaxPool2d(pool_sizes[i])))
    
    def forward(self, x):
        """x: (B, 1, T, F) 即 (batch, channels, time, freq)"""
        outputs = []
        for block in self.blocks:
            x = block(x)  # (B, C, T, F)
            outputs.append(x)  # keep for FDAP
        return outputs  # list of 5 tensors

cnn = PandeyaCNN(conv_weights, bn_weights).to(DEVICE)
cnn.eval()
print(f'  ✅ CNN 重建完成 ({sum(p.numel() for p in cnn.parameters()):,} 参数)')

# ═══════════════════════════════════
# 3. 特征提取 (Mel → CNN → FDAP/GAP)
# ═══════════════════════════════════
print('\n[3/4] 特征提取...')
BATCH = 64

def compute_mel_and_forward(samples_batch, cnn, device):
    """批量: librosa mel → PyTorch CNN → 返回 5 层特征图"""
    batch_raw = np.array([s['raw'] for s in samples_batch])  # (B, T)
    
    # 手动 Mel (参数来自 hdf5 Kapre 配置)
    # 1) Kapre padding='same' ≈ librosa center=True
    mels = []
    for y in batch_raw:
        mel = librosa.feature.melspectrogram(
            y=y, sr=MEL_CONFIG['sr'],
            n_fft=MEL_CONFIG['n_fft'],
            hop_length=MEL_CONFIG['hop_length'],
            n_mels=MEL_CONFIG['n_mels'],
            fmin=MEL_CONFIG['fmin'],
            fmax=MEL_CONFIG['fmax'],
            power=2.0, center=True
        )
        mels.append(mel)  # (96, T)
    
    # decibel (每文件单独 ref, 匹配 Kapre 行为)
    mels_db = np.array([librosa.power_to_db(m, ref=max(m.max(), 1e-10)) for m in mels])
    
    # 转 (B, 1, T, 96) — PyTorch channels_first
    mel_batch = np.transpose(mels_db, (0, 2, 1))  # (B, T, 96)
    mel_batch = np.expand_dims(mel_batch, 1)  # (B, 1, T, 96)
    
    # 2) Normalization2D(int_axis=0, str_axis='data_sample')
    #    = 每个样本, 每个频段, 沿时间维独立归一化
    mel_mean = mel_batch.mean(axis=2, keepdims=True)  # (B, 1, 1, 96)
    mel_std = mel_batch.std(axis=2, keepdims=True) + 1e-5  # (B, 1, 1, 96)
    mel_norm = (mel_batch - mel_mean) / mel_std
    
    # CNN forward
    x = torch.from_numpy(mel_norm).float().to(device)
    with torch.no_grad():
        outputs = cnn(x)  # list of 5 tensors
    
    # 转 numpy
    return [o.cpu().numpy() for o in outputs]

# ── 缓存检查 ──
CACHE_KEY = 'fdap_v4_pytorch'
cache_fdap = CACHE_DIR / f'{CACHE_KEY}_fdap_{n}.npy'
cache_gap = CACHE_DIR / f'{CACHE_KEY}_gap_{n}.npy'
cache_labels = CACHE_DIR / f'{CACHE_KEY}_labels_{n}.npy'

if cache_fdap.exists() and cache_gap.exists():
    print('  ✅ 缓存命中')
    fdap_feats = np.load(cache_fdap)
    gap_feats = np.load(cache_gap)
else:
    fdap_feats_list = []
    gap_feats_list = []
    
    for i in range(0, n, BATCH):
        batch_samples = samples[i:i+BATCH]
        mp_outputs = compute_mel_and_forward(batch_samples, cnn, DEVICE)
        # mp_outputs: list of 5 numpy arrays, each (B, C=32, T, F)
        
        B = len(batch_samples)
        for b in range(B):
            fdap_vecs = []
            gap_vecs = []
            for idx, feat in enumerate(mp_outputs):
                f = feat[b]  # (32, T, F) — PyTorch channel-first
                # Transpose to (F, T, 32) for FDAP along frequency
                f_perm = np.transpose(f, (2, 1, 0))  # (F, T, 32)
                H = f_perm.shape[0]  # frequency dim
                n_seg = FDAP_SEGMENTS[idx]
                seg_len = max(H // n_seg, 1)
                
                for seg_i in range(n_seg):
                    s_start = seg_i * seg_len
                    s_end = s_start + seg_len if seg_i < n_seg - 1 else H
                    seg = f_perm[s_start:s_end, :, :]  # (seg_F, T, 32)
                    fdap_vecs.append(np.mean(seg, axis=(0, 1)))
                    fdap_vecs.append(np.std(seg, axis=(0, 1)))
                
                gap_vecs.append(np.mean(f_perm, axis=(0, 1)))
            
            fdap_feats_list.append(np.concatenate(fdap_vecs))
            gap_feats_list.append(np.concatenate(gap_vecs))
        
        if (i // BATCH) % 3 == 0:
            print(f'  进度: {min(i+BATCH, n)}/{n}')
    
    fdap_feats = np.stack(fdap_feats_list)
    gap_feats = np.stack(gap_feats_list)
    # 清理 NaN/Inf
    fdap_feats = np.nan_to_num(fdap_feats, nan=0.0, posinf=0.0, neginf=0.0)
    gap_feats = np.nan_to_num(gap_feats, nan=0.0, posinf=0.0, neginf=0.0)
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
    ax.set_xticks([]); ax.set_yticks([])
    for label in LABELS_10:
        mask = np.array([l == label for l in labels_all])
        if mask.sum() == 0: continue
        ax.scatter(coords[mask, 0], coords[mask, 1], c=COLORS[label],
                   label=label, alpha=0.4, s=5, edgecolors='none')

handles, leg_labels = axes[1].get_legend_handles_labels()
fig.legend(handles, leg_labels, loc='lower center', ncol=10,
           fontsize=7, bbox_to_anchor=(0.5, -0.02))
plt.suptitle(f'Pandeya CNN (PyTorch, model_best 权重) | SR={SR} | perplexity={PERP}',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout(rect=[0, 0.08, 1, 1])

out_png = BASE / f'模型代码/t-SNE_FDAP_v4_pytorch_{TS}.png'
plt.savefig(str(out_png), dpi=150, bbox_inches='tight')
print(f'\n  ✅ 图: {out_png}')

# 轮廓系数
le = LabelEncoder()
y_int = le.fit_transform(labels_all)
fdap_sil = silhouette_score(fdap_feats, y_int)
gap_sil = silhouette_score(gap_feats, y_int)
print('\n  轮廓系数 (越高越好):')
print(f'    FDAP: {fdap_sil:.4f}')
print(f'    GAP:  {gap_sil:.4f}')
print(f'    差距: {fdap_sil - gap_sil:+.4f}')

out_json = BASE / f'模型代码/t-SNE_FDAP_metrics_v4_{TS}.json'
with open(out_json, 'w') as f:
    json.dump({
        'n_samples': n,
        'fdap_dims': int(fdap_feats.shape[1]),
        'gap_dims': int(gap_feats.shape[1]),
        'fdap_silhouette': round(float(fdap_sil), 4),
        'gap_silhouette': round(float(gap_sil), 4),
        'silhouette_diff': round(float(fdap_sil - gap_sil), 4),
    }, f, indent=2)
print(f'  ✅ 指标: {out_json}')
print('\n  和已有的 HuBERT t-SNE 对比看聚类差异')

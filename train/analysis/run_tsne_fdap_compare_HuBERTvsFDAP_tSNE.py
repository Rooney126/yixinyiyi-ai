"""
t-SNE 对比: mean pooling vs 时域分段池化 (FDAP analog)
看保留时间结构能否改善聚类效果

FDAP analog: 将 HuBERT 的 ~250 时间帧分成 K 段, 每段均值 → K×768 维
vs
Mean pool: 全部帧均值 → 768 维
# t-SNE 对比：HuBERT Mean Pool vs FDAP 时域分段池化
"""

import os
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from collections import Counter
from datetime import datetime
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import HubertModel, Wav2Vec2FeatureExtractor

matplotlib.use('Agg')
import matplotlib.pyplot as plt

SR = 16000
CROP_SAMPLES = int(SR * 2.0)
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
np.random.seed(SEED)

BASE = Path(r'/path/to/dataset')
DATA_DIR = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
TS = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT = BASE / f'模型代码/t-SNE_FDAP对比_{TS}.png'

LABELS_10 = ['Angry','Defence','Fighting','Happy','HuntingMind',
             'Mating','MotherCall','Paining','Resting','Warning']

# ── 高区分度颜色 ──
COLORS = {
    'Angry':'#d62728', 'Defence':'#e377c2', 'Fighting':'#ff7f0e', 'Warning':'#bcbd22',
    'Happy':'#2ca02c', 'HuntingMind':'#17becf', 'Mating':'#9467bd',
    'MotherCall':'#1f77b4', 'Paining':'#8c564b', 'Resting':'#7f7f7f',
}

# 池化策略: 每种返回 (name, description, embedding_dim)
POOL_STRATEGIES = [
    ('mean',    'Mean Pool (全局均值)',  lambda h: h.mean(dim=1)),                 # (B,768)
    ('fdap_2',  'FDAP 2段 (前后半)',     lambda h: h.reshape(h.shape[0], 2, -1, h.shape[2]).mean(dim=2).reshape(h.shape[0], -1)),  # (B,2*768)
    ('fdap_4',  'FDAP 4段',              lambda h: h.reshape(h.shape[0], 4, -1, h.shape[2]).mean(dim=2).reshape(h.shape[0], -1)),  # (B,4*768)
    ('fdap_8',  'FDAP 8段',              lambda h: h.reshape(h.shape[0], 8, -1, h.shape[2]).mean(dim=2).reshape(h.shape[0], -1)),  # (B,8*768)
]

print('═'*50)
print('  t-SNE 对比: Mean Pool vs FDAP 时域分段')
print('  层: [3,4,5]均值 | perplexity: 60')
print('═'*50)

# ── 加载 ──
print('\n[1/3] 加载音频...')
samples = []
for label in LABELS_10:
    d = DATA_DIR / label
    if not d.exists():
        continue
    for fp in sorted(d.glob('*.mp3'))[:200]:  # 每类200条 → 2000条，足够看聚类
        try:
            y,_ = librosa.load(str(fp), sr=SR, mono=True)
            if len(y) > CROP_SAMPLES:
                start = (len(y) - CROP_SAMPLES) // 2
                y = y[start:start + CROP_SAMPLES]
            else:
                pad = CROP_SAMPLES - len(y)
                y = np.pad(y, (pad // 2, pad - pad // 2))
            samples.append({'waveform': y.astype(np.float32), 'label': label})
        except Exception as e:
            pass

n = len(samples)
all_labels = [s['label'] for s in samples]
print(f'  共 {n} 条')
for label, cnt in Counter(all_labels).most_common():
    print(f'    {label:<12}: {cnt:4d}')

# ── HuBERT 嵌入 + 不同池化 ──
print('\n[2/3] HuBERT 前向传播...')
fe = Wav2Vec2FeatureExtractor.from_pretrained('facebook/hubert-base-ls960', local_files_only=True)
hubert = HubertModel.from_pretrained('facebook/hubert-base-ls960', local_files_only=True).to(DEVICE)
hubert.eval()

# 先收集所有 hidden states (3-5层均值), 不池化
all_hidden = []  # list of (time, 768)
with torch.no_grad():
    for i in range(0, n, BATCH_SIZE):
        batch = samples[i:i + BATCH_SIZE]
        waves = [s['waveform'] for s in batch]
        inputs = fe(waves, sampling_rate=SR, return_tensors='pt', padding=True).to(DEVICE)
        out = hubert(inputs.input_values, output_hidden_states=True)
        hs = out.hidden_states
        # L3-5 均值, 保持时间维: (B, T, 768)
        emb_time = torch.stack([hs[l] for l in [3,4,5]], dim=0).mean(dim=0)  # (B, T, 768)
        all_hidden.append(emb_time.cpu())
        if (i // BATCH_SIZE) % 10 == 0:
            print(f'  进度: {i + len(batch)}/{n}')

hidden_stacked = torch.cat(all_hidden, dim=0)  # (N, T, 768)

# 应用不同池化策略
pooled = {}
for name, desc, fn in POOL_STRATEGIES:
    pooled[name] = fn(hidden_stacked).numpy()
    print(f'  {desc}: {pooled[name].shape}')

# ── t-SNE 绘图 ──
print('\n[3/3] t-SNE + 对比绘图...')
n_rows, n_cols = 2, 2  # 4个池化策略排2×2
PERP = 60

fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
axes = axes.flatten()

pca50 = PCA(n_components=50, random_state=SEED)

for idx, (name, desc, fn) in enumerate(POOL_STRATEGIES):
    ax = axes[idx]
    X = pooled[name]
    
    # PCA 预降维
    X_pca = pca50.fit_transform(X)
    print(f'  t-SNE: {name} ({desc})...')
    tsne = TSNE(n_components=2, perplexity=PERP, random_state=SEED)
    coords = tsne.fit_transform(X_pca)
    
    ax.set_title(f'{desc}\n(dim={X.shape[1]})', fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
    for label in LABELS_10:
        mask = np.array([l == label for l in all_labels])
        n_pts = mask.sum()
        if n_pts == 0:
            continue
        pts = coords[mask]
        ax.scatter(pts[:,0], pts[:,1], c=COLORS[label],
                   label=label, alpha=0.4, s=3, edgecolors='none')

# 共享图例
handles, labels_leg = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_leg, loc='lower center', ncol=10,
           fontsize=8, bbox_to_anchor=(0.5, -0.02))

plt.suptitle(f'HuBERT L3-5 | perplexity={PERP} | Mean vs FDAP时间分段池化',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(str(OUTPUT), dpi=150, bbox_inches='tight')
print(f'\n  保存: {OUTPUT}')
print('  解读: 看FDAP分段后, 原本混在一起的类(Happy↔HuntingMind)是否分开')

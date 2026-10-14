"""
t-SNE 参数探索: perplexity × LAYERS
一次性输出 2×3 网格，对比不同配置下嵌入空间的聚类效果
# t-SNE 参数探索：perplexity × 层组合 网格图
"""

import os, sys, warnings
warnings.filterwarnings('ignore')
os.environ['HF_HOME'] = '/mnt/c/Users/陈项目负责人/.cache/huggingface'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from collections import Counter
import numpy as np
import torch
import librosa
from pathlib import Path
from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SR = 16000
CROP_SAMPLES = int(SR * 2.0)
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
np.random.seed(SEED)

BASE = Path('/mnt/c/Users/陈项目负责人/Desktop')
DATA_DIR = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
TS = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT = BASE / f'模型代码/t-SNE_grid_{TS}.png'

LABELS_10 = ['Angry','Defence','Fighting','Happy','HuntingMind',
             'Mating','MotherCall','Paining','Resting','Warning']
# ── 高区分度颜色方案（tab10 衍生） ──
# 4个Aggressive子类用暖色渐变, 2个Happy子类用蓝绿区分
COLORS = {
    'Angry':'#c2185b',      # 深粉红 🌸 用户标注：深粉色
    'Defence':'#f48fb1',    # 浅粉红 🌸 用户标注：浅粉色
    'Fighting':'#ff6f00',   # 橙 🟠 用户标注：橙色
    'Warning':'#fdd835',    # 黄 🟡 用户标注：黄色
    'Happy':'#2e7d32',      # 绿 🟩 用户标注：绿色
    'HuntingMind':'#00bcd4',# 青 🟦 用户标注：青色/浅蓝
    'Mating':'#7b1fa2',     # 紫 🟣 用户标注：紫色
    'MotherCall':'#1565c0', # 深蓝 🟦 用户标注：深蓝色
    'Paining':'#d32f2f',   # 红 🔴 用户标注：红色
    'Resting':'#9e9e9e',    # 灰 ⬜ 用户标注：灰色
}

# 待探索的参数组合
LAYER_CFGS = [
    ('早期层 [3,4,5]', [3,4,5]),
    ('深层 [8,9,10]',  [8,9,10]),
]
PERPLEXITIES = [30, 60, 90]

print('═'*50)
print('  t-SNE 参数探索: 2×3 网格')
print(f'  LAYERS:     {[n for _,n in LAYER_CFGS]}')
print(f'  perplexity: {PERPLEXITIES}')
print('═'*50)

# ── 加载 ──
print('[1/3] 加载音频...')
samples = []
MAX_PER_CLASS = 100  # 每类最多100条，t-SNE够用
for label in LABELS_10:
    d = DATA_DIR / label
    if not d.exists():
        print(f'  [SKIP] {label}: 目录不存在')
        continue
    for i, fp in enumerate(sorted(d.glob('*.mp3'))):
        if i >= MAX_PER_CLASS:
            print(f'  [LIMIT] {label}: 采样{MAX_PER_CLASS}条')
            break
        import subprocess
        # 用ffprobe预检文件是否损坏
        try:
            subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json',
                           '-show_streams', str(fp)],
                          capture_output=True, timeout=5, check=True)
        except Exception:
            print(f'  [CORRUPT] {label}/{fp.name}: 跳过')
            continue
        try:
            y,_ = librosa.load(str(fp), sr=SR, mono=True, duration=6.0, res_type='scipy')
            if len(y) > CROP_SAMPLES:
                start = (len(y) - CROP_SAMPLES) // 2
                y = y[start:start + CROP_SAMPLES]
            else:
                pad = CROP_SAMPLES - len(y)
                y = np.pad(y, (pad // 2, pad - pad // 2))
            samples.append({'waveform': y.astype(np.float32), 'label': label})
        except Exception as e:
            print(f'  [SKIP] {label}/{fp.name}: {e}')
    print(f'  [OK] {label}: {len([s for s in samples if s["label"]==label])}条')

n = len(samples)
print(f'  共 {n} 条')

# ── 嵌入 ──
print('[2/3] 一次 HuBERT 前向 → 提取两个层组合...')
fe = Wav2Vec2FeatureExtractor.from_pretrained('facebook/hubert-base-ls960', local_files_only=True)
hubert = HubertModel.from_pretrained('facebook/hubert-base-ls960', local_files_only=True).to(DEVICE)
hubert.eval()

all_labels = [s['label'] for s in samples]
print('  每类数量:')
for label, cnt in Counter(all_labels).most_common():
    print(f'    {label:<12}: {cnt:4d}')

# 同时提取两组嵌入（从同一批 hidden_states 取不同层均值）
emb_sets = {}
for cfg_name, layers in LAYER_CFGS:
    emb_sets[cfg_name] = []

with torch.no_grad():
    for i in range(0, n, BATCH_SIZE):
        batch = samples[i:i + BATCH_SIZE]
        waves = [s['waveform'] for s in batch]
        inputs = fe(waves, sampling_rate=SR, return_tensors='pt', padding=True).to(DEVICE)
        out = hubert(inputs.input_values, output_hidden_states=True)
        hs = out.hidden_states  # tuple of 13 tensors

        for cfg_name, layers in LAYER_CFGS:
            embs = torch.stack([hs[l] for l in layers], dim=0).mean(dim=0).mean(dim=1)
            emb_sets[cfg_name].extend(embs.cpu().numpy())

        if (i // BATCH_SIZE) % 10 == 0:
            print(f'  进度: {i + len(batch)}/{n}')

for cfg_name, layers in LAYER_CFGS:
    arr = np.stack(emb_sets[cfg_name], axis=0)
    emb_sets[cfg_name] = arr
    print(f'  {cfg_name}: {arr.shape}')

cnt = Counter(all_labels)
for label in LABELS_10:
    print(f'    {label:<12}: {cnt.get(label, 0):4d}')

# ── t-SNE + 绘图 ──
print('[3/3] 6 次 t-SNE + 2×3 网格绘图（可能需要几分钟）...')
n_rows, n_cols = len(LAYER_CFGS), len(PERPLEXITIES)

# 用 PCA 预降维到 50（t-SNE 更快，保留 >95% 方差）
pca50 = PCA(n_components=50, random_state=SEED)
for cfg_name in emb_sets:
    emb_sets[cfg_name] = pca50.fit_transform(emb_sets[cfg_name])

coords_grid = {}  # {(cfg_name, perp): np.array}
for cfg_name in emb_sets:
    X = emb_sets[cfg_name]
    for perp in PERPLEXITIES:
        print(f'  t-SNE: {cfg_name}  perplexity={perp}...')
        tsne = TSNE(n_components=2, perplexity=perp, random_state=SEED, verbose=0)
        coords_grid[(cfg_name, perp)] = tsne.fit_transform(X)

# ── 绘图 ──
fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))

for ri, (cfg_name, _) in enumerate(LAYER_CFGS):
    for ci, perp in enumerate(PERPLEXITIES):
        ax = axes[ri, ci]
        coords = coords_grid[(cfg_name, perp)]

        for label in LABELS_10:
            mask = np.array([l == label for l in all_labels])
            n_pts = mask.sum()
            if n_pts == 0:
                continue
            pts = coords[mask]
            ax.scatter(pts[:,0], pts[:,1], c=COLORS[label],
                       label=label, alpha=0.4, s=5, edgecolors='none')

        # 行/列标签
        if ri == 0:
            ax.set_title(f'perplexity={perp}', fontsize=13, fontweight='bold')
        if ci == 0:
            ax.set_ylabel(cfg_name, fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

# 共享图例（取第一行最后一列的图例）
handles, labels_leg = axes[0, -1].get_legend_handles_labels()
fig.legend(handles, labels_leg, loc='lower center', ncol=10,
           fontsize=9, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(str(OUTPUT), dpi=150, bbox_inches='tight')
print(f'  保存: {OUTPUT}')
print('  解读: 看哪些 perplexity × LAYERS 组合下同类情绪点能聚在一起')

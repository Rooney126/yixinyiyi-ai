"""
嵌入空间可视化 — 看 HuBERT mean-pooled 嵌入是否可分
==================================================
出力: embedding_space.png (UMAP 2D 投影)

用法:
  python test_perframe_vote.py  # (先预热, 共享缓存)
  python vis_embedding_space.py
# 嵌入空间交互式可视化（旧版，可忽略）
"""

import os, sys, time
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import librosa
from collections import Counter
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import matplotlib
matplotlib.use('Agg')  # 无头模式
import matplotlib.pyplot as plt

# ── 配置 ──
SR = 16000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYERS = [3, 4, 5]
BASE = Path(r'/path/to/dataset')
MODEL_DIR = BASE / '模型代码/训练结果/hubert_v2/freeze_model_hubert_20260618_111717'

# 数据集: 原猫语分类 163条  +  CatSound_V2 测试集 5922条?
# 先只跑 163 条, 清晰
DATA_DIR = BASE / '原猫语分类/情绪分类'

# 中文目录 → 英文标签
CN_MAP = {
    '交配（低沉嘶吼）': 'Mating',
    '休息': 'Resting',
    '快乐': 'Happy',
    '愤怒（呜哇哇哇）': 'Angry',
    '打架（尖锐刺耳）': 'Fighting',
    '母体呼叫（警笛）': 'MotherCall',
    '狩猎（短促的嗷呜）': 'HuntingMind',
    '疼痛': 'Paining',
    '警告（呜~）（咕噜）': 'Warning',
    '防御(响尾蛇嘶哈)': 'Defence',
}

# 合并为 6 类的映射 (用于染色)
MERGE_6_COLORS = {
    'Aggressive': '#E03131',  # 红
    'Happy': '#F08C00',       # 橙
    'Mating': '#9C36B5',      # 紫
    'MotherCall': '#1971C2',  # 蓝
    'Paining': '#E64980',     # 粉
    'Resting': '#2F9E44',     # 绿
}
MERGE_MAP = {
    'Angry':'Aggressive', 'Defence':'Aggressive',
    'Fighting':'Aggressive', 'Warning':'Aggressive',
    'Happy':'Happy', 'HuntingMind':'Happy',
    'Mating':'Mating', 'MotherCall':'MotherCall',
    'Paining':'Paining', 'Resting':'Resting',
}


# =========================================================
# 加载模型
# =========================================================
print(f'[加载] HuBERT + Processor...')
t0 = time.time()
model = HubertModel.from_pretrained('facebook/hubert-base-ls960', local_files_only=True)
model.to(DEVICE); model.eval()
for p in model.parameters(): p.requires_grad = False
processor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/hubert-base-ls960', local_files_only=True)
print(f'  完成 ({time.time()-t0:.1f}s)')


def preprocess(y):
    peak = np.max(np.abs(y))
    return y / peak if peak > 1e-10 else np.zeros_like(y)


@torch.no_grad()
def extract_embedding(audio_path):
    """返回 mean-pooled 嵌入 (768,)"""
    y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    y = preprocess(y)
    inputs = processor([y], sampling_rate=SR, return_tensors="pt", padding=True).input_values.to(DEVICE)
    outputs = model(inputs, output_hidden_states=True)
    hs = outputs.hidden_states
    emb = torch.stack([hs[i] for i in LAYERS], dim=0).mean(dim=0).mean(dim=1).cpu().numpy()
    return emb.squeeze(0)  # (768,)


# =========================================================
# 提取嵌入
# =========================================================
print(f'\n提取嵌入: {DATA_DIR}\n')
embeddings = []
true_labels = []  # 6类合并标签
file_names = []

for cn_dir in sorted(DATA_DIR.iterdir()):
    if not cn_dir.is_dir():
        continue
    cn_name = cn_dir.name
    en_label = CN_MAP.get(cn_name)
    if en_label is None:
        continue
    merged = MERGE_MAP[en_label]
    
    files = sorted(cn_dir.glob('*.wav')) + sorted(cn_dir.glob('*.mp3'))
    if not files:
        continue
    
    for fp in files:
        try:
            emb = extract_embedding(fp)
            embeddings.append(emb)
            true_labels.append(merged)
            file_names.append(fp.name)
        except Exception as e:
            print(f'  [!] {fp.name}: {e}')
    
    print(f'  {merged:<12} ({en_label:<12}) → {len(files)} 条')

embeddings = np.array(embeddings)
print(f'\n嵌入矩阵: {embeddings.shape}')

# =========================================================
# 降维 & 可视化
# =========================================================
print(f'\n降维...')

# 先看各类样本数
counter = Counter(true_labels)
print(f'  样本分布:')
for cls, cnt in sorted(counter.items(), key=lambda x: -x[1]):
    print(f'    {cls}: {cnt}')

# UMAP (首选)
use_umap = False
try:
    import umap
    reduc = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    coords = reduc.fit_transform(embeddings)
    use_umap = True
    print(f'  UMAP 完成: {coords.shape}')
except ImportError:
    print(f'  [*] umap-learn 未安装, 回退 PCA + t-SNE...')
    # PCA 降噪 → t-SNE
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    pca = PCA(n_components=50, random_state=42)
    emb_pca = pca.fit_transform(embeddings)
    print(f'  PCA: {embeddings.shape} → {emb_pca.shape} (保留 {pca.explained_variance_ratio_.sum():.1%} 方差)')
    
    tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings)//2),
                random_state=42, max_iter=1000, learning_rate='auto')
    coords = tsne.fit_transform(emb_pca)
    print(f'  t-SNE 完成')

# ── 绘图 ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 左图: 每种 6 类合并情绪一种颜色
all_merged = sorted(set(true_labels))
colors = [MERGE_6_COLORS.get(c, '#666666') for c in all_merged]
for cls, c in zip(all_merged, colors):
    mask = np.array([l == cls for l in true_labels])
    n = mask.sum()
    ax1.scatter(coords[mask, 0], coords[mask, 1], c=c, label=f'{cls} ({n})',
                alpha=0.7, s=30, edgecolors='w', linewidth=0.5)
ax1.set_title('HuBERT Mean-Pooled Embedding Space (6类合并)', fontsize=13)
ax1.legend(fontsize=9, loc='upper right', markerscale=1.2)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlabel('UMAP/t-SNE 1')
ax1.set_ylabel('UMAP/t-SNE 2')
ax1.grid(alpha=0.15)

# 右图: 10 类原始标签
# 把 true_labels 换回 10 类
true_labels_10 = []
for cn_dir in sorted(DATA_DIR.iterdir()):
    if not cn_dir.is_dir():
        continue
    en_label = CN_MAP.get(cn_dir.name)
    if en_label is None:
        continue
    files = sorted(cn_dir.glob('*.wav')) + sorted(cn_dir.glob('*.mp3'))
    for _ in files:
        true_labels_10.append(en_label)

all_10 = sorted(set(true_labels_10))
# 10 类配色
COLORS_10 = plt.cm.tab10(np.linspace(0, 1, 10))
for i, cls in enumerate(all_10):
    mask = np.array([l == cls for l in true_labels_10])
    n = mask.sum()
    ax2.scatter(coords[mask, 0], coords[mask, 1], c=COLORS_10[i],
                label=f'{cls} ({n})', alpha=0.7, s=30, edgecolors='w', linewidth=0.5)
ax2.set_title('HuBERT Mean-Pooled Embedding Space (10类原始)', fontsize=13)
ax2.legend(fontsize=8, loc='upper right', markerscale=1.2, ncol=2)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlabel('UMAP/t-SNE 1')
ax2.set_ylabel('UMAP/t-SNE 2')
ax2.grid(alpha=0.15)

plt.tight_layout()
TS = datetime.now().strftime('%Y%m%d_%H%M%S')
save_path = BASE / f'模型代码/embedding_space_{TS}.png'
plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
print(f'\n✅ 已保存: {save_path}')

# ── 额外: 打印类间距离矩阵 (L2距离) ──
print(f'\n── 类间嵌入平均距离 (L2) ──')
class_centers = {}
for cls in all_merged:
    mask = np.array([l == cls for l in true_labels])
    class_centers[cls] = embeddings[mask].mean(axis=0)
cls_names = sorted(class_centers.keys())
print(f'{"":>14}', end='')
for c in cls_names:
    print(f'  {c:>12}', end='')
print()
for c1 in cls_names:
    print(f'{c1:>14}', end='')
    for c2 in cls_names:
        d = np.linalg.norm(class_centers[c1] - class_centers[c2])
        if c1 == c2:
            print(f'  {"—":>12}', end='')
        else:
            print(f'  {d:>8.2f}', end='')
    print()

# ── 额外: 类内/类间距离比 (判断可分性) ──
print(f'\n── 类内方差 / 最近类距离比 (越小越好) ──')
intra_var = {}
for cls in all_merged:
    mask = np.array([l == cls for l in true_labels])
    center = class_centers[cls]
    var = np.mean(np.linalg.norm(embeddings[mask] - center, axis=1)**2)**0.5  # RMS距离
    intra_var[cls] = var

for cls in sorted(all_merged):
    center = class_centers[cls]
    nearest = min(
        np.linalg.norm(center - class_centers[other])
        for other in all_merged if other != cls
    )
    ratio = intra_var[cls] / nearest if nearest > 0 else float('inf')
    print(f'  {cls:<14} 类内RMS={intra_var[cls]:.2f}  最近类距离={nearest:.2f}  比例={ratio:.3f}')

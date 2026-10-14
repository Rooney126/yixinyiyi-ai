"""
原始 Mel (VAD 切分后) t-SNE: 看数据本身是否可分
不经过 CNN，直接从原始音频 → VAD → Mel → PCA → t-SNE
"""
import os, sys, warnings, random
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import librosa
import hashlib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SR = 16000
N_MELS = 128
N_FFT = 512
HOP_LENGTH = 256
VAD_FRAMES = 100
VAD_TOP_DB = 25
N_SAMPLES = 500  # 子集大小, 全量太慢

BASE = Path('/mnt/c/Users/陈项目负责人/Desktop')
DATA_DIR = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
OUTPUT = BASE / '模型代码/tsne_vad_raw.png'
LABELS = ["Angry","Defence","Fighting","Happy","HuntingMind","Mating","MotherCall","Paining","Resting","Warning"]
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))

def extract_longest_vocal(y):
    segs = librosa.effects.split(y, top_db=VAD_TOP_DB, frame_length=512, hop_length=HOP_LENGTH)
    if len(segs) == 0: return y
    longest = max(segs, key=lambda s: s[1] - s[0])
    return y[longest[0]:longest[1]]

print(f'加载 CatSound_V2, 随机抽样 {N_SAMPLES} 条...')
import csv
files, all_labels = [], []
with open(DATA_DIR / 'Catmood_NAYA_1xAug.csv') as f:
    reader = csv.reader(f); next(reader)
    for row in reader:
        if len(row) < 3: continue
        li = int(row[2])
        fn = os.path.basename(row[1].strip())
        lp = DATA_DIR / LABELS[li] / fn
        if lp.exists():
            files.append((str(lp), li))

random.seed(42)
files = random.sample(files, min(N_SAMPLES, len(files)))

print(f'提取 VAD + Mel ({len(files)} 条)...')
mels = []
for i, (fp, li) in enumerate(files):
    y, _ = librosa.load(fp, sr=SR, mono=True)
    y = extract_longest_vocal(y)
    peak = np.max(np.abs(y))
    y = y / peak if peak > 1e-10 else np.zeros_like(y)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    # 中心裁/pad 到 VAD_FRAMES
    if mel_norm.shape[1] < VAD_FRAMES:
        mel_norm = np.pad(mel_norm, ((0,0), (0, VAD_FRAMES - mel_norm.shape[1])))
    else:
        s = (mel_norm.shape[1] - VAD_FRAMES) // 2
        mel_norm = mel_norm[:, s:s+VAD_FRAMES]
    mels.append(mel_norm.flatten())
    if (i+1) % 100 == 0:
        print(f'  {i+1}/{len(files)}')

X = np.array(mels)
y = np.array([li for _, li in files])
print(f'\n矩阵: {X.shape} ({X.shape[1]}维)')

print(f'PCA 降维 (50维)...')
pca = PCA(n_components=min(50, X.shape[0], X.shape[1]), random_state=42)
X_pca = pca.fit_transform(X)
print(f'  PCA 方差保留: {pca.explained_variance_ratio_.sum():.1%}')

print(f't-SNE 降维...')
tsne = TSNE(n_components=2, perplexity=40, random_state=42, verbose=1)
z = tsne.fit_transform(X_pca)

print(f'绘图...')
fig, ax = plt.subplots(figsize=(12, 10))
for i in range(10):
    mask = y == i
    ax.scatter(z[mask,0], z[mask,1], c=[COLORS[i]], label=LABELS[i], s=10, alpha=0.6)
ax.legend(fontsize=9, markerscale=3)
ax.set_title(f'原始 VAD-Mel t-SNE (子集{N_SAMPLES}条, PCA→t-SNE)', fontsize=13)
ax.axis('off')
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150)
print(f'已保存: {OUTPUT}')

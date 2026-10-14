"""
CNN FDAP 嵌入 t-SNE 可视化
用 fold3_best 提 CatSound_V2 所有样本的嵌入 → t-SNE
"""
import os
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stdout.reconfigure(encoding='utf-8')
import csv
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch import nn

matplotlib.use('Agg')
import matplotlib.pyplot as plt

SR = 16000
N_MELS = 128
N_FFT = 512
HOP_LENGTH = 256
MAX_T = 300
BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE = Path('/Desktop')
DATA_DIR = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
MODEL_PATH = BASE / '模型代码/训练结果/fdap_cnn_v1/mel_cache/fold3_best.pth'
OUTPUT_DIR = BASE / '模型代码'
OUTPUT = OUTPUT_DIR / 'tsne_cnn_fold3.png'

LABELS = [
    "Angry","Defence","Fighting","Happy","HuntingMind",
    "Mating","MotherCall","Paining","Resting","Warning"
]

class FDAPMobileCatCNN(nn.Module):
    def __init__(self, num_classes=10, fdap_bands=4):
        super().__init__()
        self.fdap_bands = fdap_bands
        self.conv1 = nn.Sequential(nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((1,2)))
        self.conv3 = nn.Sequential(nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((1,2)))
        self.conv4 = nn.Sequential(nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Identity())
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(512,128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128,10))
    def forward(self, x, return_embed=False):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x); x = self.conv4(x)
        b,c,f,t = x.shape
        bs = f // self.fdap_bands
        fdap = []
        for i in range(self.fdap_bands):
            end = (i+1)*bs if i<3 else f
            fdap.append(x[:,:,i*bs:end,:].mean(dim=[2,3]))
        x = torch.cat(fdap, dim=1)
        if return_embed:
            return x
        return self.classifier(x)

def load_catsound_v2(data_dir):
    csv_path = data_dir / 'Catmood_NAYA_1xAug.csv'
    files, labels = [], []
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 3: continue
            label_idx = int(row[2])
            filename = os.path.basename(row[1].strip())
            local_path = data_dir / LABELS[label_idx] / filename
            if local_path.exists():
                files.append(str(local_path))
                labels.append(label_idx)
    return files, np.array(labels)

def load_mel_from_cache(file_path, cache_dir):
    """从缓存加载 Mel 频谱"""
    import hashlib
    cache_name = hashlib.md5(file_path.encode()).hexdigest() + '.npy'
    cache_path = os.path.join(cache_dir, cache_name)
    mel = np.load(cache_path)
    if mel.shape[1] < MAX_T:
        mel = np.pad(mel, ((0,0), (0, MAX_T - mel.shape[1])))
    else:
        mel = mel[:, :MAX_T]
    return mel

def extract_embeddings(model, files, labels_all, cache_dir, batch_size=64):
    model.eval()
    all_emb, all_labels = [], []
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        batch_mels = []
        for fp in batch_files:
            mel = load_mel_from_cache(fp, cache_dir)
            batch_mels.append(mel)
        mels = torch.FloatTensor(np.array(batch_mels)).unsqueeze(1).to(DEVICE)
        with torch.no_grad():
            emb = model(mels, return_embed=True).cpu().numpy()
        all_emb.append(emb)
        all_labels.extend(labels_all[i:i+batch_size])
        if (i//batch_size) % 10 == 0:
            print(f'  提取中: {i+len(batch_files)}/{len(files)}')
    return np.vstack(all_emb), np.array(all_labels)

def main():
    print(f'设备: {DEVICE}')
    print('加载 fold3 模型...')
    model = FDAPMobileCatCNN(num_classes=10).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    
    print('加载 CatSound_V2...')
    files, labels = load_catsound_v2(DATA_DIR)
    print(f'  总样本: {len(files)}')
    
    print('提取 512 维 FDAP 嵌入 (从 Mel 缓存)...')
    CACHE_DIR = BASE / '模型代码/训练结果/fdap_cnn_v1/mel_cache'
    embs, lbls = extract_embeddings(model, files, labels, CACHE_DIR, BATCH_SIZE)
    print(f'  嵌入: {embs.shape}')
    
    print('t-SNE 降维 (n_jobs=-1)...')
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, n_jobs=-1, verbose=1)
    z = tsne.fit_transform(embs)
    
    print('绘图...')
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(figsize=(12, 10))
    for i in range(10):
        mask = lbls == i
        ax.scatter(z[mask,0], z[mask,1], c=[colors[i]], label=LABELS[i], s=8, alpha=0.6)
    ax.legend(fontsize=9, loc='best', markerscale=3)
    ax.set_title(f'CNN FDAP 嵌入 t-SNE (fold3, {len(files)}条, {embs.shape[1]}维)', fontsize=13)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=150)
    print(f'已保存: {OUTPUT}')

if __name__ == '__main__':
    main()

"""t-SNE 对比: 原始ch64 vs 增强ch64"""
import csv
import hashlib
import os
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.manifold import TSNE
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SR=16000; N_MELS=128; N_FFT=512; HOP_LENGTH=256; VAD_FRAMES=100
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
BASE=Path('/Desktop')
DATA_DIR=BASE/'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
CACHE_DIR=BASE/'模型代码/训练结果/fdap_cnn_v2/mel_cache_vad'
LABELS=["Angry","Defence","Fighting","Happy","HuntingMind","Mating","MotherCall","Paining","Resting","Warning"]
COLORS = plt.cm.tab10(np.linspace(0,1,10))

class FDAPMobileCatCNN_ch64(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1=nn.Sequential(nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2=nn.Sequential(nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((1,2)))
        self.conv3=nn.Sequential(nn.Conv2d(32,48,3,padding=1), nn.BatchNorm2d(48), nn.ReLU(), nn.MaxPool2d((1,2)))
        self.conv4=nn.Sequential(nn.Conv2d(48,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Identity())
        self.classifier=nn.Sequential(nn.Dropout(0.3), nn.Linear(256,96), nn.ReLU(), nn.Dropout(0.3), nn.Linear(96,num_classes))
    def forward(self, x, embed=False):
        x=self.conv1(x); x=self.conv2(x); x=self.conv3(x); x=self.conv4(x)
        b,c,f,t=x.shape; bs=f//4
        fdap=[x[:,:,i*bs:(i+1)*bs if i<3 else f,:].mean(dim=[2,3]) for i in range(4)]
        embed_vec=torch.cat(fdap,dim=1)
        if embed: return embed_vec
        return self.classifier(embed_vec)

@torch.no_grad()
def extract_embeddings(model, files, cache_dir):
    model.eval()
    embs, labs = [], []
    for i, fp in enumerate(files):
        cn = hashlib.md5(fp.encode()).hexdigest() + '.npy'
        mel = np.load(os.path.join(cache_dir, cn))
        if mel.shape[1] < VAD_FRAMES:
            mel = np.pad(mel, ((0,0), (0, VAD_FRAMES-mel.shape[1])))
        else:
            s = (mel.shape[1]-VAD_FRAMES)//2
            mel = mel[:, s:s+VAD_FRAMES]
        t_mel = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0).to(DEVICE)
        emb = model(t_mel, embed=True).cpu().numpy()[0]
        embs.append(emb)
        labs.append(labels[i])
        if (i+1)%1000==0: print(f'  {i+1}/{len(files)}')
    return np.array(embs), np.array(labs)

# 加载数据
print('加载数据...')
with open(DATA_DIR/'Catmood_NAYA_1xAug.csv', encoding='utf-8') as f:
    reader = csv.reader(f); next(reader)
    files, labels = [], []
    for row in reader:
        li=int(row[2]); fn=os.path.basename(row[1].strip())
        lp=DATA_DIR/LABELS[li]/fn
        if lp.exists(): files.append(str(lp)); labels.append(LABELS[li])

print(f'  {len(files)} 条')

# 两个模型
models = [
    ('原始 ch64 (83.3%)', BASE/'模型代码/训练结果/fdap_cnn_ch64/fold4_best.pth'),
    ('增强 ch64 (85.0%)', BASE/'模型代码/训练结果/fdap_cnn_ch64_aug/fold4_best.pth'),
]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, (title, pth) in zip(axes, models):
    print(f'\n{title}')
    model = FDAPMobileCatCNN_ch64().to(DEVICE)
    state = torch.load(str(pth), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=False)
    
    embs, labs = extract_embeddings(model, files, CACHE_DIR)
    print(f'  嵌入: {embs.shape}, 运行t-SNE...')
    tsne = TSNE(n_components=2, perplexity=50, random_state=42, max_iter=1000)
    xy = tsne.fit_transform(embs)
    
    for li, label in enumerate(LABELS):
        mask = labs == label
        ax.scatter(xy[mask,0], xy[mask,1], c=[COLORS[li]], label=label, 
                   s=5, alpha=0.7)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

handles = [plt.Rectangle((0,0),1,1, color=COLORS[i]) for i in range(10)]
fig.legend(handles, LABELS, loc='lower center', ncol=5, fontsize=9)
plt.tight_layout(rect=[0,0.08,1,1])
out = str(BASE/'模型代码/tsne_comparison.png')
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'\n✅ 保存: {out}')

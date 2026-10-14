"""
ch64 CNN FDAP 嵌入 t-SNE 可视化
加载 best fold 权重 → 提取 256 维 FDAP 嵌入 → t-SNE → 散点图
"""
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
MODEL_PATH=BASE/'模型代码/训练结果/fdap_cnn_ch64/fold4_best.pth'
OUTPUT=BASE/'模型代码/tsne_ch64_fold4.png'
LABELS=["Angry","Defence","Fighting","Happy","HuntingMind","Mating","MotherCall","Paining","Resting","Warning"]
COLORS = plt.cm.tab10(np.linspace(0,1,10))

class FDAPMobileCatCNN_ch64(nn.Module):
    """ch64: 70K参数, conv4=64ch, FDAP=256维"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1=nn.Sequential(nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2=nn.Sequential(nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((1,2)))
        self.conv3=nn.Sequential(nn.Conv2d(32,48,3,padding=1), nn.BatchNorm2d(48), nn.ReLU(), nn.MaxPool2d((1,2)))
        self.conv4=nn.Sequential(nn.Conv2d(48,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Identity())
        self.classifier=nn.Sequential(nn.Dropout(0.3), nn.Linear(256,96), nn.ReLU(), nn.Dropout(0.3), nn.Linear(96,num_classes))
    def forward(self, x, embed=False):
        x=self.conv1(x); x=self.conv2(x); x=self.conv3(x); x=self.conv4(x)
        b,c,f,t=x.shape
        fdap=[x[:,:,i*(f//4):(i+1)*(f//4) if i<3 else f,:].mean(dim=[2,3]) for i in range(4)]
        x=torch.cat(fdap,dim=1)
        return x if embed else self.classifier(x)

print(f'设备: {DEVICE}')
print(f'模型: {MODEL_PATH}')
model=FDAPMobileCatCNN_ch64().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()
print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')

# 加载数据
print('加载 CatSound_V2...')
files, labels = [], []
with open(DATA_DIR/'Catmood_NAYA_1xAug.csv') as f:
    reader=csv.reader(f); next(reader)
    for row in reader:
        if len(row)<3: continue
        li=int(row[2]); fn=os.path.basename(row[1].strip())
        lp=DATA_DIR/LABELS[li]/fn
        if lp.exists():
            files.append(str(lp)); labels.append(li)
print(f'  {len(files)}条')

# 加载 Mel 缓存
print('加载 Mel 缓存...')
mels=[]
for fp in files:
    cn=hashlib.md5(fp.encode()).hexdigest()+'.npy'
    mel=np.load(os.path.join(CACHE_DIR, cn))
    if mel.shape[1]<VAD_FRAMES:
        mel=np.pad(mel, ((0,0),(0,VAD_FRAMES-mel.shape[1])))
    else:
        start=(mel.shape[1]-VAD_FRAMES)//2
        mel=mel[:,start:start+VAD_FRAMES]
    mels.append(mel)
mels=np.stack(mels)
print(f'  {list(mels.shape)}')

# 提取 FDAP 嵌入
print('提取 FDAP 嵌入 (256维)...')
embeds=[]
with torch.no_grad():
    for i in range(0,len(mels),128):
        batch=torch.FloatTensor(mels[i:i+128]).unsqueeze(1).to(DEVICE)
        embeds.append(model(batch, embed=True).cpu().numpy())
embeds=np.concatenate(embeds)
print(f'  嵌入: {embeds.shape}')

# t-SNE
print('t-SNE (n_components=2, perplexity=40, random_state=42)...')
tsne=TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000)
xy=tsne.fit_transform(embeds)
print(f'  t-SNE 完成: {xy.shape}')

# 绘图
fig,ax=plt.subplots(figsize=(14,10))
for i in range(10):
    mask=np.array(labels)==i
    ax.scatter(xy[mask,0], xy[mask,1], c=[COLORS[i]],
               label=LABELS[i], alpha=0.6, s=8, edgecolors='none')
ax.legend(fontsize=11, markerscale=3)
ax.set_title(f'ch64 FDAP Embeddings t-SNE (256-dim→2D)\n'
             f'{len(files)} samples · perplexity=40',
             fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.savefig(OUTPUT, dpi=200)
print(f'保存: {OUTPUT}')

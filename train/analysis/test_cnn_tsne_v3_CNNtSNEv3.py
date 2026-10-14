"""
CNN FDAP 嵌入 t-SNE (fold4, F1=84.1%)
从缓存加载 Mel → CNN 提 384 维嵌入 → t-SNE
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
BATCH_SIZE=128
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

BASE=Path('/Desktop')
DATA_DIR=BASE/'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
CACHE_DIR=BASE/'模型代码/训练结果/fdap_cnn_v2/mel_cache_vad'
MODEL_PATH=BASE/'模型代码/训练结果/fdap_cnn_v3/fold4_best.pth'
OUTPUT=BASE/'模型代码/tsne_cnn_fold4.png'
LABELS=["Angry","Defence","Fighting","Happy","HuntingMind","Mating","MotherCall","Paining","Resting","Warning"]

class FDAPMobileCatCNN(nn.Module):
    def __init__(self, num_classes=10, fdap_bands=4):
        super().__init__()
        self.fdap_bands=fdap_bands
        self.conv1=nn.Sequential(nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2=nn.Sequential(nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((1,2)))
        self.conv3=nn.Sequential(nn.Conv2d(32,48,3,padding=1), nn.BatchNorm2d(48), nn.ReLU(), nn.MaxPool2d((1,2)))
        self.conv4=nn.Sequential(nn.Conv2d(48,96,3,padding=1), nn.BatchNorm2d(96), nn.ReLU(), nn.Identity())
        self.classifier=nn.Sequential(nn.Dropout(0.3), nn.Linear(384,96), nn.ReLU(), nn.Dropout(0.3), nn.Linear(96,num_classes))
    def forward(self, x, embed=False):
        x=self.conv1(x); x=self.conv2(x); x=self.conv3(x); x=self.conv4(x)
        b,c,f,t=x.shape; bs=f//4
        fdap=[x[:,:,i*bs:(i+1)*bs if i<3 else f,:].mean(dim=[2,3]) for i in range(4)]
        x=torch.cat(fdap,dim=1)
        return x if embed else self.classifier(x)

print(f'设备: {DEVICE}')
print(f'加载模型: {MODEL_PATH}')
model=FDAPMobileCatCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()
print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')

# 加载文件列表和标签
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
print(f'{len(files)} 条')

# 从缓存加载 Mel 并提嵌入
print('提嵌入...')
embs, lbls = [], []
for i in range(0, len(files), BATCH_SIZE):
    batch_files=files[i:i+BATCH_SIZE]
    batch_mels=[]
    for fp in batch_files:
        cn=hashlib.md5(fp.encode()).hexdigest()+'.npy'
        mel=np.load(CACHE_DIR/cn)
        if mel.shape[1]<VAD_FRAMES:
            mel=np.pad(mel,((0,0),(0,VAD_FRAMES-mel.shape[1])))
        else:
            s=(mel.shape[1]-VAD_FRAMES)//2; mel=mel[:,s:s+VAD_FRAMES]
        batch_mels.append(mel)
    mels=torch.FloatTensor(np.array(batch_mels)).unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        emb=model(mels, embed=True).cpu().numpy()
    embs.append(emb); lbls.extend(labels[i:i+BATCH_SIZE])
    if (i//BATCH_SIZE)%5==0: print(f'  {i+len(batch_files)}/{len(files)}')

X=np.vstack(embs); y=np.array(lbls)
print(f'嵌入: {X.shape}')

print('t-SNE...')
z=TSNE(n_components=2, perplexity=40, random_state=42, verbose=1).fit_transform(X)

print('绘图...')
colors=plt.cm.tab10(np.linspace(0,1,10))
fig,ax=plt.subplots(figsize=(12,10))
for i in range(10):
    m=y==i; ax.scatter(z[m,0],z[m,1],c=[colors[i]],label=LABELS[i],s=8,alpha=0.6)
ax.legend(fontsize=9,markerscale=3)
ax.set_title(f'CNN FDAP t-SNE (fold4, F1=84.1%, {len(files)}条, 384维)',fontsize=13)
ax.axis('off'); plt.tight_layout(); plt.savefig(OUTPUT,dpi=150)
print(f'已保存: {OUTPUT}')

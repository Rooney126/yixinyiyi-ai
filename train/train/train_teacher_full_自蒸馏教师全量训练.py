"""
自蒸馏 Step 1: 全量训 ch64 Teacher on CatSound_V2 (全部 5922 条, 不 split)
→ dump soft logits → 供 student 蒸馏
"""
import os, sys, json, csv, warnings, time, hashlib
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import librosa

SR=16000; N_MELS=128; N_FFT=512; HOP_LENGTH=256; VAD_FRAMES=100; VAD_TOP_DB=25
BATCH_SIZE=64; EPOCHS=100; LR=1e-3; WEIGHT_DECAY=1e-4; NUM_WORKERS=2

BASE=Path('/mnt/c/Users/陈项目负责人/Desktop')
DATA_DIR=BASE/'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
OUTPUT_DIR=BASE/'模型代码/训练结果/ch64_teacher_full'
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
# 复用已有的 mel_cache
V2_CACHE=BASE/'模型代码/训练结果/fdap_cnn_v2/mel_cache_vad'
V3_CACHE=BASE/'模型代码/训练结果/fdap_cnn_v3/mel_cache'
CH64_CACHE=BASE/'模型代码/训练结果/fdap_cnn_ch64/mel_cache'

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
print(f'设备: {DEVICE}')
if DEVICE=='cuda': print(f'  GPU: {torch.cuda.get_device_name(0)}')

LABELS=["Angry","Defence","Fighting","Happy","HuntingMind","Mating","MotherCall","Paining","Resting","Warning"]

def load_catsound_v2(data_dir):
    csv_path=data_dir/'Catmood_NAYA_1xAug.csv'
    files,labels,cat_ids,is_aug=[],[],[],[]
    with open(csv_path,encoding='utf-8') as f:
        reader=csv.reader(f)
        try: next(reader)
        except: return np.array(files),np.array(labels),np.array(cat_ids)
        for row in reader:
            if len(row)<3: continue
            li=int(row[2]); fn=os.path.basename(row[1].strip())
            lp=data_dir/LABELS[li]/fn
            if lp.exists():
                files.append(str(lp)); labels.append(LABELS[li])
                raw_stem=lp.stem.replace('_aug1(1)','').lower()
                cat_ids.append(raw_stem); is_aug.append('_aug1(1)' in lp.stem)
    return np.array(files),np.array(labels),np.array(cat_ids),np.array(is_aug)

def extract_longest_vocal(y):
    segs=librosa.effects.split(y,top_db=VAD_TOP_DB,frame_length=512,hop_length=HOP_LENGTH)
    if len(segs)==0: return y
    longest=max(segs,key=lambda s:s[1]-s[0])
    return y[longest[0]:longest[1]]

def load_mel(fp):
    """从 mel_cache 加载, fallback 实时计算"""
    h=hashlib.md5(fp.encode()).hexdigest()
    for cd in [V2_CACHE,V3_CACHE,CH64_CACHE]:
        cp=os.path.join(cd,h+'.npy')
        if os.path.exists(cp):
            mel=np.load(cp)
            if mel.shape[1]<VAD_FRAMES:
                mel=np.pad(mel,((0,0),(0,VAD_FRAMES-mel.shape[1])))
            else:
                s=(mel.shape[1]-VAD_FRAMES)//2
                mel=mel[:,s:s+VAD_FRAMES]
            return mel
    y,_=librosa.load(fp,sr=SR,mono=True)
    y=extract_longest_vocal(y)
    peak=np.max(np.abs(y))
    y=y/peak if peak>1e-10 else np.zeros_like(y)
    mel=librosa.feature.melspectrogram(y=y,sr=SR,n_mels=N_MELS,n_fft=N_FFT,hop_length=HOP_LENGTH)
    mel_db=librosa.power_to_db(mel,ref=np.max)
    mel_norm=(mel_db-mel_db.min())/(mel_db.max()-mel_db.min()+1e-8)
    if mel_norm.shape[1]<VAD_FRAMES:
        mel_norm=np.pad(mel_norm,((0,0),(0,VAD_FRAMES-mel_norm.shape[1])))
    else:
        s=(mel_norm.shape[1]-VAD_FRAMES)//2
        mel_norm=mel_norm[:,s:s+VAD_FRAMES]
    return mel_norm

# ch64 模型
class FDAPMobileCatCNN_ch64(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.conv1=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2))
        self.conv2=nn.Sequential(nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d((1,2)))
        self.conv3=nn.Sequential(nn.Conv2d(32,48,3,padding=1),nn.BatchNorm2d(48),nn.ReLU(),nn.MaxPool2d((1,2)))
        self.conv4=nn.Sequential(nn.Conv2d(48,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.Identity())
        self.classifier=nn.Sequential(nn.Dropout(0.3),nn.Linear(256,96),nn.ReLU(),nn.Dropout(0.3),nn.Linear(96,num_classes))
    def forward(self,x):
        x=self.conv1(x);x=self.conv2(x);x=self.conv3(x);x=self.conv4(x)
        b,c,f,t=x.shape;bs=f//4
        fdap=[x[:,:,i*bs:(i+1)*bs if i<3 else f,:].mean(dim=[2,3]) for i in range(4)]
        x=torch.cat(fdap,dim=1)
        return self.classifier(x)

def preload_mels(file_list):
    """预加载全部 Mel 到内存"""
    print('  预加载 Mel...', flush=True, end=' ')
    t0=time.time()
    mels=[load_mel(f) for f in file_list]
    print(f'{len(mels)} 条, {time.time()-t0:.0f}s', flush=True)
    return torch.FloatTensor(np.stack(mels)).unsqueeze(1)

def train_teacher(model,train_mels,train_labels,file_stem='teacher'):
    """全量训练, 不 split"""
    print(f'\n{"─"*50}\n全量训练: {len(train_labels)} 条')
    ds=torch.utils.data.TensorDataset(train_mels,torch.LongTensor(train_labels))
    loader=DataLoader(ds,BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=True)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    best_loss=float('inf');no_improve=0;patience=15

    for ep in range(1,EPOCHS+1):
        t0=time.time()
        model.train();train_loss=0
        for m,l in loader:
            m,l=m.to(DEVICE),l.to(DEVICE)
            opt.zero_grad()
            loss=F.cross_entropy(model(m),l)
            loss.backward();opt.step()
            train_loss+=loss.item()
        sched.step()
        avg_loss=train_loss/len(loader)
        if avg_loss<best_loss:
            best_loss=avg_loss;no_improve=0
            torch.save(model.state_dict(),OUTPUT_DIR/f'{file_stem}_best.pth')
        else: no_improve+=1
        if ep%5==1 or ep==EPOCHS or no_improve==1:
            print(f'  E{ep:3d}/{EPOCHS} L={avg_loss:.4f} Best={best_loss:.4f} {time.time()-t0:.0f}s', flush=True)
        if no_improve>=patience:
            print(f'  早停 @{ep}', flush=True);break
    print(f'  完成: Best Loss={best_loss:.4f}', flush=True)

@torch.no_grad()
def dump_logits(model,mels,labels,out_path):
    """全量推理, 存 logits"""
    print(f'\n  Dump logits ({len(labels)} 条)...', flush=True)
    model.eval()
    loader=DataLoader(TensorDataset(mels,labels),BATCH_SIZE,shuffle=False,pin_memory=True)
    all_logits=[]
    for m,_ in loader:
        logits=model(m.to(DEVICE)).cpu().numpy()
        all_logits.append(logits)
    all_logits=np.concatenate(all_logits)
    np.save(out_path,all_logits)
    print(f'  保存: {out_path} ({all_logits.shape})', flush=True)

def main():
    print(f'\n{"█"*60}')
    print(f'  自蒸馏 Teacher: 全量 ch64 (70K), 不 split')
    print(f'{"█"*60}')
    print('\n[1] 加载数据...', flush=True)
    files,labels,cat_ids,is_aug=load_catsound_v2(DATA_DIR)
    le=LabelEncoder();y=le.fit_transform(labels)
    print(f'  {len(files)}条, {len(le.classes_)}类', flush=True)
    all_mels=preload_mels(files.tolist())
    all_labels=torch.LongTensor(y)
    ts=time.strftime('%Y%m%d_%H%M%S')
    print('\n[2] 全量训练 Teacher...', flush=True)
    model=FDAPMobileCatCNN_ch64().to(DEVICE)
    p=sum(p.numel() for p in model.parameters())
    print(f'  参数量: {p:,}', flush=True)
    train_teacher(model,all_mels,all_labels,'teacher_full')

    print('\n[3] Dump soft logits...', flush=True)
    model.load_state_dict(torch.load(OUTPUT_DIR/'teacher_full_best.pth',map_location=DEVICE))
    dump_logits(model,all_mels,all_labels,OUTPUT_DIR/'teacher_logits.npy')

    print(f'\n完成!', flush=True)

if __name__=='__main__':
    main()

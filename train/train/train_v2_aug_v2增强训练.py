"""
v2 + 音频域增强 (Speed ±10%, Pitch ±2)
对比 v2 基线 (83.2%) — 单一变量: 增强方式
不能预缓存 Mel, 每次实时加载 wav
"""
import os, sys, time, json, csv, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import librosa

SR=16000; N_MELS=128; N_FFT=512; HOP_LENGTH=256; VAD_FRAMES=100; VAD_TOP_DB=25
BATCH_SIZE=64; EPOCHS=100; LR=1e-3; WEIGHT_DECAY=1e-4; NUM_WORKERS=2

BASE=Path('/mnt/c/Users/陈项目负责人/Desktop')
DATA_DIR=BASE/'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
OUTPUT_DIR=BASE/'模型代码/训练结果/fdap_cnn_v2_aug'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LABELS=["Angry","Defence","Fighting","Happy","HuntingMind","Mating","MotherCall","Paining","Resting","Warning"]
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
print(f'设备: {DEVICE}')
print(f'  GPU: {torch.cuda.get_device_name(0)}' if DEVICE=='cuda' else '')

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
    return y[max(segs,key=lambda s:s[1]-s[0])[0]:max(segs,key=lambda s:s[1]-s[0])[1]]

def wav_to_mel(fp,augment=False):
    y,_=librosa.load(fp,sr=SR,mono=True)
    if augment:
        speed=np.random.uniform(0.9,1.1)
        y=librosa.effects.time_stretch(y=y,rate=speed)
        pitch=np.random.randint(-2,3)
        if pitch!=0: y=librosa.effects.pitch_shift(y=y,sr=SR,n_steps=pitch)
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

class CatWavDataset(Dataset):
    def __init__(self,file_list,labels,augment=False):
        self.files=file_list;self.labels=labels;self.augment=augment
    def __len__(self): return len(self.labels)
    def __getitem__(self,idx):
        mel=wav_to_mel(self.files[idx],augment=self.augment)
        if self.augment:
            noise=np.random.randn(*mel.shape)*0.01
            mel=mel+noise
            shift=np.random.randint(-int(0.1*mel.shape[1]),int(0.1*mel.shape[1]))
            mel=np.roll(mel,shift,axis=1)
        return torch.FloatTensor(mel).unsqueeze(0),torch.LongTensor([self.labels[idx]])[0]

class FDAPMobileCatCNN(nn.Module):
    """v2 模型: 99K, 4层Conv2D + FDAP"""
    def __init__(self,num_classes=10):
        super().__init__()
        self.conv1=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2))
        self.conv2=nn.Sequential(nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d((1,2)))
        self.conv3=nn.Sequential(nn.Conv2d(32,48,3,padding=1),nn.BatchNorm2d(48),nn.ReLU(),nn.MaxPool2d((1,2)))
        self.conv4=nn.Sequential(nn.Conv2d(48,96,3,padding=1),nn.BatchNorm2d(96),nn.ReLU(),nn.Identity())
        self.classifier=nn.Sequential(nn.Dropout(0.3),nn.Linear(384,96),nn.ReLU(),nn.Dropout(0.3),nn.Linear(96,num_classes))
    def forward(self,x):
        x=self.conv1(x);x=self.conv2(x);x=self.conv3(x);x=self.conv4(x)
        b,c,f,t=x.shape;bs=f//4
        fdap=[x[:,:,i*bs:(i+1)*bs if i<3 else f,:].mean(dim=[2,3]) for i in range(4)]
        x=torch.cat(fdap,dim=1)
        return self.classifier(x)

def train_one_fold(fold,train_files,train_labels,test_files,test_labels):
    print(f'\n{"─"*50}\n折 {fold}: 训练 {len(train_files)} → 测试 {len(test_files)}')
    print('  预计算测试集 Mel...')
    test_mels=[wav_to_mel(f,augment=False) for f in test_files]
    test_mels=torch.FloatTensor(np.stack(test_mels)).unsqueeze(1)
    train_ds=CatWavDataset(train_files,train_labels,augment=True)
    test_ds=torch.utils.data.TensorDataset(test_mels,torch.LongTensor(test_labels))
    train_loader=DataLoader(train_ds,BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=True)
    test_loader=DataLoader(test_ds,BATCH_SIZE,shuffle=False,pin_memory=True)
    model=FDAPMobileCatCNN().to(DEVICE)
    p=sum(p.numel() for p in model.parameters())
    print(f'  参数量: {p:,}')
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    best_acc,best_f1,best_ep=0,0,0
    no_improve,patience=0,15
    for ep in range(1,EPOCHS+1):
        t0=time.time()
        model.train();train_loss=0
        for m,l in train_loader:
            m,l=m.to(DEVICE),l.to(DEVICE)
            opt.zero_grad();loss=F.cross_entropy(model(m),l)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step()
            train_loss+=loss.item()
        model.eval();all_p,all_l=[],[]
        with torch.no_grad():
            for m,l in test_loader:
                o=model(m.to(DEVICE));all_p.append(o.argmax(1).cpu());all_l.append(l)
        p=torch.cat(all_p).numpy();la=torch.cat(all_l).numpy()
        acc=(p==la).mean();f1=f1_score(la,p,average='macro',zero_division=0)
        sched.step();et=time.time()-t0
        if f1>best_f1:
            best_acc,best_f1,best_ep=acc,f1,ep;no_improve=0
            torch.save(model.state_dict(),OUTPUT_DIR/f'fold{fold}_best.pth')
        else: no_improve+=1
        if ep%5==1 or ep==EPOCHS or no_improve==1:
            print(f'  E{ep:3d}/{EPOCHS} L={train_loss/len(train_loader):.4f} Acc={acc:.1%} F1={f1:.1%} Best={best_f1:.1%}@{best_ep} {et:.0f}s')
        if no_improve>=patience:
            print(f'  早停 @{ep}, Best={best_f1:.1%}');break
    print(f'  折{fold}: Acc={best_acc:.1%} F1={best_f1:.1%} (ep{best_ep})')
    return best_acc,best_f1,best_ep

def main():
    print(f'\n{"█"*60}')
    print(f'  v2 + 音频域增强 (Speed±10%, Pitch±2)')
    print(f'  对比 v2 基线 (83.2%) — 单一变量: 增强方式')
    print(f'{"█"*60}')
    print('\n[1] 加载数据...')
    files,labels,cat_ids,is_aug=load_catsound_v2(DATA_DIR)
    le=LabelEncoder();y=le.fit_transform(labels)
    print(f'  {len(files)}条')
    print('\n[2] GroupKFold 快速测试 (1折)...')
    gkf=GroupKFold(n_splits=5)
    clean_mask=~is_aug;clean_idx=np.where(clean_mask)[0]
    for fold,(tr_c,te_c) in enumerate(gkf.split(clean_idx,y[clean_idx],cat_ids[clean_idx]),1):
        if fold>1: break
        tr_cats=set(cat_ids[clean_idx[tr_c]]);te_cats=set(cat_ids[clean_idx[te_c]])
        tr_m=np.array([cid in tr_cats for cid in cat_ids]);te_m=np.array([cid in te_cats for cid in cat_ids])
        a,f,_=train_one_fold(fold,files[tr_m].tolist(),y[tr_m].tolist(),files[te_m].tolist(),y[te_m].tolist())
        print(f'\n折{fold}: Acc={a:.1%} F1={f:.1%}')
    result={'model':'v2_aug','fold1':{'acc':float(a),'f1':float(f)}}
    ts=time.strftime('%Y%m%d_%H%M%S')
    with open(OUTPUT_DIR/f'result_{ts}.json','w') as fp: json.dump(result,fp,indent=2)

if __name__=='__main__':
    main()

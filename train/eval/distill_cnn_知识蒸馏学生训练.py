"""
知识蒸馏: 用 Teacher (HuBERT) 的 soft labels 训练 Student (ch64 CNN)

损失函数: L = α * CE(y_hard, y_pred) + (1-α) * T² * KL(soft_teacher/T, y_pred/T)
"""
import csv
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader

# ─── 超参 ───
SR=16000; N_MELS=128; N_FFT=512; HOP_LENGTH=256; VAD_FRAMES=100; VAD_TOP_DB=25
BATCH_SIZE=64; EPOCHS=100; LR=1e-3; WEIGHT_DECAY=1e-4; NUM_WORKERS=2

# 蒸馏参数 (可调!)
T=4           # 温度
ALPHA=0.5     # α: 1=纯CE, 0=纯模仿teacher (teacher 82%，hard label 不能丢太多)

BASE=Path('/Desktop')
DATA_DIR=BASE/'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
# Teacher 输出的 soft logits (由 train_teacher_full.py 生成)
TEACHER_LOGITS=BASE/'模型代码/训练结果/ch64_teacher_full/teacher_logits.npy'
OUTPUT_DIR=BASE/'模型代码/训练结果/ch64_distill'
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

# 复用 v2 mel_cache 加速
V2_CACHE=BASE/'模型代码/训练结果/fdap_cnn_v2/mel_cache_vad'
V3_CACHE=BASE/'模型代码/训练结果/fdap_cnn_v3/mel_cache'

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
print(f'设备: {DEVICE}')
if DEVICE=='cuda': print(f'  GPU: {torch.cuda.get_device_name(0)}')

LABELS=["Angry","Defence","Fighting","Happy","HuntingMind","Mating","MotherCall","Paining","Resting","Warning"]

# ─── 数据加载 ───
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

def load_mel_v2(fp):
    """从 v2/v3 cache 加载 Mel, fallback 到实时计算"""
    import hashlib
    h=hashlib.md5(fp.encode()).hexdigest()
    for cd in [V2_CACHE,V3_CACHE]:
        cp=os.path.join(cd,h+'.npy')
        if os.path.exists(cp):
            mel=np.load(cp)
            if mel.shape[1]<VAD_FRAMES:
                mel=np.pad(mel,((0,0),(0,VAD_FRAMES-mel.shape[1])))
            else:
                s=(mel.shape[1]-VAD_FRAMES)//2
                mel=mel[:,s:s+VAD_FRAMES]
            return mel
    # fallback
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

# ─── 模型: ch64 CNN (和 train_fdap_cnn_v2_ch64.py 一致) ───
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

# ─── 蒸馏损失 ───
def distillation_loss(student_logits,teacher_logits,labels,T=4.0,alpha=0.3):
    """CE(y_hard, y_pred) + α * T² * KL(soft_teacher/T, student/T)"""
    # 硬损失: cross-entropy
    loss_hard=F.cross_entropy(student_logits,labels)
    # 软损失: KL divergence
    soft_teacher=F.softmax(teacher_logits/T,dim=1)
    soft_student=F.log_softmax(student_logits/T,dim=1)
    loss_soft=F.kl_div(soft_student,soft_teacher,reduction='batchmean')*(T*T)
    return alpha*loss_hard+(1-alpha)*loss_soft

# ─── 训练函数 ───
def train_student(fold,train_files,train_labels,test_files,test_labels,soft_logits_train):
    print(f'\n{"─"*50}\n折 {fold}: 训练 {len(train_files)} → 测试 {len(test_labels)}', flush=True)

    # 测试集 Mel
    print('  加载测试集 Mel...', flush=True, end=' ')
    t0=time.time()
    test_mels=[load_mel_v2(f) for f in test_files]
    test_mels=torch.FloatTensor(np.stack(test_mels)).unsqueeze(1)
    test_ds=torch.utils.data.TensorDataset(test_mels,torch.LongTensor(test_labels))
    test_loader=DataLoader(test_ds,BATCH_SIZE,shuffle=False,pin_memory=True)
    print(f'{time.time()-t0:.0f}s', flush=True)

    # 训练集 Mel
    print('  加载训练集 Mel...', flush=True, end=' ')
    t0=time.time()
    train_mels=[load_mel_v2(f) for f in train_files]
    train_mels=torch.FloatTensor(np.stack(train_mels)).unsqueeze(1)
    soft_train=torch.FloatTensor(soft_logits_train)
    train_ds=torch.utils.data.TensorDataset(train_mels,torch.LongTensor(train_labels),soft_train)
    train_loader=DataLoader(train_ds,BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=True)
    print(f'{time.time()-t0:.0f}s', flush=True)

    model=FDAPMobileCatCNN_ch64().to(DEVICE)
    p=sum(p.numel() for p in model.parameters())
    print(f'  参数量: {p:,}', flush=True)

    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    best_acc,best_f1,best_ep=0,0,0
    no_improve,patience=0,15

    for ep in range(1,EPOCHS+1):
        t0=time.time()
        model.train();train_loss=0
        for m,l,soft in train_loader:
            m,l,soft=m.to(DEVICE),l.to(DEVICE),soft.to(DEVICE)
            opt.zero_grad()
            logits=model(m)
            loss=distillation_loss(logits,soft,l,T=T,alpha=ALPHA)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.0)
            opt.step()
            train_loss+=loss.item()
        model.eval();all_p,all_l=[],[]
        with torch.no_grad():
            for m,l in test_loader:
                o=model(m.to(DEVICE));all_p.append(o.argmax(1).cpu());all_l.append(l)
        p=torch.cat(all_p).numpy();la=torch.cat(all_l).numpy()
        acc=(p==la).mean();f1=f1_score(la,p,average='macro',zero_division=0)
        sched.step()
        if f1>best_f1:
            best_acc,best_f1,best_ep=acc,f1,ep;no_improve=0
            torch.save(model.state_dict(),OUTPUT_DIR/f'fold{fold}_best.pth')
        else: no_improve+=1
        if ep%5==1 or ep==EPOCHS or no_improve==1:
            print(f'  E{ep:3d}/{EPOCHS} L={train_loss/len(train_loader):.4f} Acc={acc:.1%} F1={f1:.1%} Best={best_f1:.1%}@{best_ep} {time.time()-t0:.0f}s', flush=True)
        if no_improve>=patience:
            print(f'  早停 @{ep}, Best={best_f1:.1%}', flush=True);break
    print(f'  折{fold}完成: Acc={best_acc:.1%} F1={best_f1:.1%} (ep{best_ep})', flush=True)
    return best_acc,best_f1

def main():
    print(f'\n{"█"*60}')
    print('  KD: Teacher(HuBERT)→Student(ch64, 70K)')
    print(f'  T={T}, α={ALPHA}')
    print(f'{"█"*60}')

    print('\n[1] 加载数据...', flush=True)
    files,labels,cat_ids,is_aug=load_catsound_v2(DATA_DIR)
    le=LabelEncoder();y=le.fit_transform(labels)
    print(f'  {len(files)}条, {len(le.classes_)}类', flush=True)

    print('\n[2] 加载 Teacher soft logits...', flush=True)
    if not os.path.exists(str(TEACHER_LOGITS)):
        print(f'  错误: 找不到 {TEACHER_LOGITS}', flush=True)
        print('  请先运行 train_teacher_full.py', flush=True)
        return
    soft_logits=np.load(str(TEACHER_LOGITS))
    assert len(soft_logits)==len(files), f'soft logits shape错: {soft_logits.shape} vs {len(files)}'
    print(f'  teacher_logits.npy: {soft_logits.shape}', flush=True)

    print('\n[3] GroupKFold + 蒸馏训练...', flush=True)
    gkf=GroupKFold(n_splits=5)
    clean_mask=~is_aug;clean_idx=np.where(clean_mask)[0]
    for fold,(tr_c,te_c) in enumerate(gkf.split(clean_idx,y[clean_idx],cat_ids[clean_idx]),1):
        if fold>1: break
        tr_cats=set(cat_ids[clean_idx[tr_c]]);te_cats=set(cat_ids[clean_idx[te_c]])
        tr_m=np.array([cid in tr_cats for cid in cat_ids]);te_m=np.array([cid in te_cats for cid in cat_ids])
        print(f'  fold1: 训练 {tr_m.sum()} 条, 测试 {te_m.sum()} 条', flush=True)

        a,f=train_student(fold,files[tr_m].tolist(),y[tr_m].tolist(),
                          files[te_m].tolist(),y[te_m].tolist(),
                          soft_logits[tr_m])

    result={
        'model':f'ch64_distill_T{T}_a{ALPHA}',
        'config':{'T':T,'alpha':ALPHA,'ep':EPOCHS,'bs':BATCH_SIZE,'lr':LR},
        'fold1':{'acc':float(a),'f1':float(f)}}
    ts=time.strftime('%Y%m%d_%H%M%S')
    with open(OUTPUT_DIR/f'result_{ts}.json','w') as fp: json.dump(result,fp,indent=2)
    print(f'\n结果: {OUTPUT_DIR}/result_{ts}.json', flush=True)
    print(f'蒸馏 F1={f:.1%} (vs ch64基线 83.3%)', flush=True)

if __name__=='__main__':
    main()

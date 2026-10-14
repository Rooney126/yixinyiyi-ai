"""
知识蒸馏第一步: Teacher (HuBERT 95M 冻结 + Linear head)
训练 → dump soft labels → 供 student 蒸馏
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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, HubertModel

# ─── 超参 ───
SR=16000
BATCH_SIZE=16     # HuBERT 95M 大, batch 小一点
EPOCHS=30
LR=1e-3
WEIGHT_DECAY=1e-4
VAD_TOP_DB=25
HOP_LENGTH=256
MAX_AUDIO_LEN=80000    # 5s 音频上限 (VAD 后猫声一般 1-3s, 给余量到 5s)

BASE=Path('/Desktop')
DATA_DIR=BASE/'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
OUTPUT_DIR=BASE/'模型代码/训练结果/hubert_teacher'
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
SOFT_LABELS_DIR=OUTPUT_DIR/'soft_labels'
SOFT_LABELS_DIR.mkdir(parents=True,exist_ok=True)

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

# ─── 模型: HubertTeacher ───
class HubertTeacher(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        print('  加载 HuBERT...', flush=True, end=' ')
        self.hubert=HubertModel.from_pretrained("facebook/hubert-base-ls960")
        for p in self.hubert.parameters(): p.requires_grad=False
        self.classifier=nn.Linear(768,num_classes)
        print(f'{sum(p.numel() for p in self.hubert.parameters())/1e6:.0f}M (冻结)+{sum(p.numel() for p in self.classifier.parameters()):,} (可训)')

    def forward(self,input_values,attention_mask=None):
        outputs=self.hubert(input_values,attention_mask=attention_mask)
        hidden=outputs.last_hidden_state  # (B,T_subsampled,768)
        if attention_mask is not None:
            # 下采样 attention_mask 匹配 HuBERT CNN 输出帧数
            # HuBERT conv stride=320: 80000 → ~250 frames
            input_lens=attention_mask.sum(dim=1)  # (B,) 有效音频长度
            feat_lens=self.hubert._get_feat_extract_output_lengths(input_lens)  # (B,) 下采样后帧数
            B,T,_=hidden.shape
            # 逐序列: 前 feat_lens[i] 帧有效, 其余 padding
            idx=torch.arange(T,device=hidden.device).unsqueeze(0).expand(B,-1)  # (B,T)
            mask=idx<feat_lens.unsqueeze(1).float()  # (B,T) bool
            mask=mask.unsqueeze(-1).float()  # (B,T,1)
            hidden=(hidden*mask).sum(dim=1)/(mask.sum(dim=1)+1e-10)
        else:
            hidden=hidden.mean(dim=1)
        return self.classifier(hidden)

# ─── 数据处理器 ───
processor=AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

def preprocess_wav(file_path):
    """加载 wav → VAD → 归一化 → HuBERT 特征提取"""
    y,_=librosa.load(file_path,sr=SR,mono=True)
    y=extract_longest_vocal(y)
    peak=np.max(np.abs(y))
    y=y/peak if peak>1e-10 else np.zeros_like(y)
    # HuBERT feature extractor: 自动重采样+归一化
    inputs=processor(y,sampling_rate=SR,return_tensors="pt",return_attention_mask=True,padding="max_length",truncation=True,
                     max_length=MAX_AUDIO_LEN)
    return inputs['input_values'].squeeze(0), inputs['attention_mask'].squeeze(0)

# ─── Dataset ───
class CatWavRawDataset(Dataset):
    """惰性加载 wav → raw waveform, 首次访问时缓存"""
    def __init__(self,file_list,labels):
        self.files=list(file_list);self.labels=list(labels)
        self._cache={}  # idx → tensor
        print(f'  惰性加载模式: {len(self.files)} 条, 首次 epoch 逐条加载', flush=True)
    def __len__(self): return len(self.labels)
    def __getitem__(self,idx):
        if idx not in self._cache:
            self._cache[idx]=preprocess_wav(self.files[idx])
        wav,mask=self._cache[idx]
        return wav,mask,self.labels[idx]

# ─── 训练函数 ───
def train_teacher(fold,train_files,train_labels,test_files,test_labels):
    print(f'\n{"─"*50}\n折 {fold}: 训练 {len(train_files)} → 测试 {len(test_files)}')
    train_ds=CatWavRawDataset(train_files,train_labels)
    test_ds=CatWavRawDataset(test_files,test_labels)
    train_loader=DataLoader(train_ds,BATCH_SIZE,shuffle=True,num_workers=0,pin_memory=True)
    test_loader=DataLoader(test_ds,BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=True)

    model=HubertTeacher().to(DEVICE)
    opt=torch.optim.AdamW(model.classifier.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    best_acc,best_f1,best_ep=0,0,0
    no_improve,patience=0,10

    for ep in range(1,EPOCHS+1):
        t0=time.time()
        model.train();train_loss=0
        for w,m,l in train_loader:
            w,m,l=w.to(DEVICE),m.to(DEVICE),l.to(DEVICE)
            opt.zero_grad()
            loss=F.cross_entropy(model(w,m),l)
            loss.backward();opt.step()
            train_loss+=loss.item()
        model.eval();all_p,all_l=[],[]
        with torch.no_grad():
            for w,m,l in test_loader:
                o=model(w.to(DEVICE),m.to(DEVICE));all_p.append(o.argmax(1).cpu());all_l.append(l)
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
    return best_acc,best_f1,best_ep

# ─── dump soft labels ───
@torch.no_grad()
def dump_soft_labels(fold,model,files):
    """对整个数据集跑 teacher, 存 soft labels (logits), 用 DataLoader batch 处理"""
    print(f'\n  Dump 折{fold} soft labels ({len(files)} 条)...', flush=True)
    model.eval()
    # 重用 CatWavRawDataset (惰性加载, 逐条 cache)
    ds=CatWavRawDataset(files,[0]*len(files))  # dummy labels
    loader=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=True)
    all_logits=[]
    t0=time.time()
    for i,(wavs,masks,_) in enumerate(loader):
        logits=model(wavs.to(DEVICE),masks.to(DEVICE)).cpu().numpy()
        all_logits.append(logits)
        if (i+1)%50==0:
            done=min((i+1)*BATCH_SIZE,len(files))
            print(f'    {done}/{len(files)} ({time.time()-t0:.0f}s)', flush=True)
    all_logits=np.concatenate(all_logits)
    np.save(SOFT_LABELS_DIR/f'fold{fold}_logits.npy',all_logits)
    print(f'  保存: fold{fold}_logits.npy {all_logits.shape} ({time.time()-t0:.0f}s)', flush=True)

# ─── main ───
def main():
    print(f'\n{"█"*60}')
    print('  Teacher: HuBERT freeze + Linear(768→10)')
    print('  快速测试: 1 折 → dump soft labels')
    print(f'{"█"*60}')

    print('\n[1] 加载数据...', flush=True)
    files,labels,cat_ids,is_aug=load_catsound_v2(DATA_DIR)
    le=LabelEncoder();y=le.fit_transform(labels)
    print(f'  {len(files)}条, {len(le.classes_)}类', flush=True)

    print('\n[2] GroupKFold...', flush=True)
    gkf=GroupKFold(n_splits=5)
    clean_mask=~is_aug;clean_idx=np.where(clean_mask)[0]
    for fold,(tr_c,te_c) in enumerate(gkf.split(clean_idx,y[clean_idx],cat_ids[clean_idx]),1):
        if fold>1: break
        tr_cats=set(cat_ids[clean_idx[tr_c]]);te_cats=set(cat_ids[clean_idx[te_c]])
        tr_m=np.array([cid in tr_cats for cid in cat_ids]);te_m=np.array([cid in te_cats for cid in cat_ids])
        print(f'  fold1: 训练 {tr_m.sum()} 条, 测试 {te_m.sum()} 条')

        # 训练 teacher
        a,f,_=train_teacher(fold,files[tr_m].tolist(),y[tr_m].tolist(),
                            files[te_m].tolist(),y[te_m].tolist())

        # 保存 teacher 结果
        result={'model':'HuBERT_Teacher','fold1':{'acc':float(a),'f1':float(f)}}
        with open(OUTPUT_DIR/f'result_{time.strftime("%Y%m%d_%H%M%S")}.json','w') as fp:
            json.dump(result,fp,indent=2)

        # dump 全量 soft labels
        model=HubertTeacher().to(DEVICE)
        model.load_state_dict(torch.load(OUTPUT_DIR/f'fold{fold}_best.pth',map_location=DEVICE))
        dump_soft_labels(fold,model,files.tolist())

    print(f'\n完成! Teacher F1={f:.1%}', flush=True)

if __name__=='__main__':
    main()

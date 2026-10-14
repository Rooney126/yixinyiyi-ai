"""
评估 v2_aug fold1_best.pth — 复用 v2 mel_cache, 10分钟→几秒
"""
import os, sys, json, csv, warnings, hashlib
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import librosa

SR=16000; N_MELS=128; N_FFT=512; HOP_LENGTH=256; VAD_FRAMES=100; VAD_TOP_DB=25
BATCH_SIZE=64; NUM_WORKERS=2

BASE=Path('/mnt/c/Users/陈项目负责人/Desktop')
DATA_DIR=BASE/'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
OUTPUT_DIR=BASE/'模型代码/训练结果/fdap_cnn_v2_aug'
V2_CACHE=BASE/'模型代码/训练结果/fdap_cnn_v2/mel_cache_vad'
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
print(f'设备: {DEVICE}')

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

def load_mel_fast(fp):
    """尝试从 v2 cache 加载, 否则实时计算"""
    h=hashlib.md5(fp.encode()).hexdigest()
    cp=os.path.join(V2_CACHE,h+'.npy')
    if os.path.exists(cp):
        mel=np.load(cp)
        # 裁剪/填充到 VAD_FRAMES
        if mel.shape[1]<VAD_FRAMES:
            mel=np.pad(mel,((0,0),(0,VAD_FRAMES-mel.shape[1])))
        else:
            s=(mel.shape[1]-VAD_FRAMES)//2
            mel=mel[:,s:s+VAD_FRAMES]
        return mel
    # fallback: 实时计算
    y,_=librosa.load(fp,sr=SR,mono=True)
    segs=librosa.effects.split(y,top_db=VAD_TOP_DB,frame_length=512,hop_length=HOP_LENGTH)
    if len(segs): y=y[max(segs,key=lambda s:s[1]-s[0])[0]:max(segs,key=lambda s:s[1]-s[0])[1]]
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

class FDAPMobileCatCNN(nn.Module):
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

def main():
    print('\n[1] 加载数据...', flush=True)
    files,labels,cat_ids,is_aug=load_catsound_v2(DATA_DIR)
    le=LabelEncoder();y=le.fit_transform(labels)
    print(f'  {len(files)}条, {len(le.classes_)}类', flush=True)

    print('\n[2] 复现 fold=1 的 GroupKFold 划分...', flush=True)
    gkf=GroupKFold(n_splits=5)
    clean_mask=~is_aug;clean_idx=np.where(clean_mask)[0]
    for fold,(tr_c,te_c) in enumerate(gkf.split(clean_idx,y[clean_idx],cat_ids[clean_idx]),1):
        if fold>1: break
        tr_cats=set(cat_ids[clean_idx[tr_c]]);te_cats=set(cat_ids[clean_idx[te_c]])
        tr_m=np.array([cid in tr_cats for cid in cat_ids]);te_m=np.array([cid in te_cats for cid in cat_ids])
        print(f'  fold1: 训练 {tr_m.sum()} 条, 测试 {te_m.sum()} 条', flush=True)
        test_files=files[te_m].tolist();test_labels=y[te_m].tolist()

    # 检查缓存命中率
    n_cached=sum(1 for fp in test_files if os.path.exists(os.path.join(V2_CACHE,hashlib.md5(fp.encode()).hexdigest()+'.npy')))
    print(f'  缓存命中: {n_cached}/{len(test_files)}', flush=True)

    print('\n[3] 加载模型...', flush=True)
    model=FDAPMobileCatCNN().to(DEVICE)
    state=torch.load(OUTPUT_DIR/'fold1_best.pth',map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    p=sum(p.numel() for p in model.parameters())
    print(f'  参数量: {p:,}', flush=True)

    print('\n[4] 加载测试集 Mel (cache 加速)...', flush=True)
    t0=__import__('time').time()
    test_mels=[load_mel_fast(fp) for fp in test_files]
    test_mels=torch.FloatTensor(np.stack(test_mels)).unsqueeze(1)
    test_ds=torch.utils.data.TensorDataset(test_mels,torch.LongTensor(test_labels))
    test_loader=DataLoader(test_ds,BATCH_SIZE,shuffle=False,pin_memory=True)
    print(f'  完成 ({__import__("time").time()-t0:.1f}s)', flush=True)

    print('\n[5] 推理...', flush=True)
    all_p,all_l=[],[]
    with torch.no_grad():
        for m,l in test_loader:
            o=model(m.to(DEVICE))
            all_p.append(o.argmax(1).cpu());all_l.append(l)
    p=torch.cat(all_p).numpy();la=torch.cat(all_l).numpy()
    acc=(p==la).mean();f1=f1_score(la,p,average='macro',zero_division=0)
    print(f'\n  Acc={acc:.4f} ({acc:.1%})', flush=True)
    print(f'  F1 ={f1:.4f} ({f1:.1%})', flush=True)

    print(f'\n  各类 F1:')
    cf1=f1_score(la,p,average=None,zero_division=0)
    for i,lb in enumerate(le.classes_):
        n=(la==i).sum()
        print(f'    {lb:15s} ({n:3d}条): F1={cf1[i]:.1%}')

    result={'model':'v2_aug_eval','fold1':{'acc':float(acc),'f1':float(f1)}}
    ts=__import__('time').strftime('%Y%m%d_%H%M%S')
    with open(OUTPUT_DIR/f'eval_result_{ts}.json','w') as fp: json.dump(result,fp,indent=2)
    print(f'\n结果已保存: eval_result_{ts}.json', flush=True)

if __name__=='__main__':
    main()

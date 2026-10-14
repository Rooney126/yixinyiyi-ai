"""
CNN 跨数据集测试: 38 新猫语 + 163 原猫语
用 fold4_best.pth (F1=84.1%), 同 VAD+1.6s 预处理
"""
import os
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stdout.reconfigure(encoding='utf-8')
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
from torch import nn

SR=16000; N_MELS=128; N_FFT=512; HOP_LENGTH=256; VAD_FRAMES=100; VAD_TOP_DB=25
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

BASE=Path('/Desktop')
MODEL_PATH=BASE/'模型代码/训练结果/fdap_cnn_v3/fold4_best.pth'
NEW_CAT_DIR=BASE/'新猫语分类/新情绪分类'
ORIG163_DIR=BASE/'原猫语分类/情绪分类'

LABELS=["Angry","Defence","Fighting","Happy","HuntingMind","Mating","MotherCall","Paining","Resting","Warning"]

# 中文→10类映射
CN_MAP={'友好':'Happy','吸引注意':'Mating','愤怒示威':'Angry','母体呼唤':'MotherCall','渴求':'Mating','疼痛惊吓':'Paining','警告驱逐':'Warning'}

# 6类合并
MERGE_6={'Aggressive':{'Angry','Defence','Fighting','Warning'},'Happy':{'Happy','HuntingMind'},
         'Mating':{'Mating'},'MotherCall':{'MotherCall'},'Paining':{'Paining'},'Resting':{'Resting'}}
LABEL_6={}
for six,tens in MERGE_6.items():
    for t in tens: LABEL_6[t]=six
CN_6={'友好':'Happy','吸引注意':'Mating','愤怒示威':'Aggressive','母体呼唤':'MotherCall','渴求':'Mating','疼痛惊吓':'Paining','警告驱逐':'Aggressive'}

class FDAPMobileCatCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2))
        self.conv2=nn.Sequential(nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d((1,2)))
        self.conv3=nn.Sequential(nn.Conv2d(32,48,3,padding=1),nn.BatchNorm2d(48),nn.ReLU(),nn.MaxPool2d((1,2)))
        self.conv4=nn.Sequential(nn.Conv2d(48,96,3,padding=1),nn.BatchNorm2d(96),nn.ReLU(),nn.Identity())
        self.classifier=nn.Sequential(nn.Dropout(0.3),nn.Linear(384,96),nn.ReLU(),nn.Dropout(0.3),nn.Linear(96,10))
    def forward(self,x):
        x=self.conv1(x);x=self.conv2(x);x=self.conv3(x);x=self.conv4(x)
        b,c,f,t=x.shape;bs=f//4
        fdap=[x[:,:,i*bs:(i+1)*bs if i<3 else f,:].mean(dim=[2,3]) for i in range(4)]
        return self.classifier(torch.cat(fdap,dim=1))

def vad_mel(path):
    y,_=librosa.load(path,sr=SR,mono=True)
    segs=librosa.effects.split(y,top_db=VAD_TOP_DB,frame_length=512,hop_length=HOP_LENGTH)
    if len(segs)==0: seg=y
    else: seg=y[max(segs,key=lambda s:s[1]-s[0])[0]:max(segs,key=lambda s:s[1]-s[0])[1]]
    peak=np.max(np.abs(seg)); seg=seg/peak if peak>1e-10 else np.zeros_like(seg)
    mel=librosa.feature.melspectrogram(y=seg,sr=SR,n_mels=N_MELS,n_fft=N_FFT,hop_length=HOP_LENGTH)
    mel_db=librosa.power_to_db(mel,ref=np.max)
    mel_norm=(mel_db-mel_db.min())/(mel_db.max()-mel_db.min()+1e-8)
    if mel_norm.shape[1]<VAD_FRAMES:
        mel_norm=np.pad(mel_norm,((0,0),(0,VAD_FRAMES-mel_norm.shape[1])))
    else:
        s=(mel_norm.shape[1]-VAD_FRAMES)//2;mel_norm=mel_norm[:,s:s+VAD_FRAMES]
    return torch.FloatTensor(mel_norm).unsqueeze(0).unsqueeze(0)

def eval_dataset(model, data_dir, label_map, name):
    print(f'\n{"="*50}\n {name}\n{"="*50}')
    audio_files=[]
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir() or d.name not in label_map: continue
        gt=label_map[d.name]
        for f in d.iterdir():
            if f.suffix.lower() in('.wav','.mp3','.m4a','.ogg'):
                audio_files.append((str(f),gt,d.name))
    print(f'{len(audio_files)}条, {len(set(l for _,l,_ in audio_files))}类')
    
    res_10=defaultdict(lambda:{'c':0,'t':0,'ps':[]})
    res_6=defaultdict(lambda:{'c':0,'t':0,'ps':[]})
    for path,gt10,cn in audio_files:
        mel=vad_mel(path).to(DEVICE)
        with torch.no_grad():
            pi=model(mel).argmax(1).item()
            p10=LABELS[pi]
        gt6=LABEL_6[gt10]; p6=LABEL_6[p10]
        res_10[gt10]['t']+=1;res_10[gt10]['c']+=p10==gt10;res_10[gt10]['ps'].append(p10)
        res_6[gt6]['t']+=1;res_6[gt6]['c']+=p6==gt6;res_6[gt6]['ps'].append(p6)
    
    print('\n--- 10类 ---')
    tc,ta=0,0
    for l in sorted(res_10):
        r=res_10[l];a=r['c']/r['t'] if r['t']>0 else 0
        tc+=r['c'];ta+=r['t']
        print(f'  {l:12s}  {r["c"]:2d}/{r["t"]:2d}={a:.1%}')
    print(f'  总体: {tc}/{ta}={tc/ta:.1%}')
    
    print('\n--- 6类 ---')
    tc,ta=0,0
    for l in ['Aggressive','Happy','Mating','MotherCall','Paining','Resting']:
        if l not in res_6: continue
        r=res_6[l];a=r['c']/r['t'] if r['t']>0 else 0
        tc+=r['c'];ta+=r['t']
        print(f'  {l:12s}  {r["c"]:2d}/{r["t"]:2d}={a:.1%}')
    print(f'  总体: {tc}/{ta}={tc/ta:.1%}')
    return tc/ta

# 原猫语163（中文目录名→10类）
ORIG_CN_MAP={
    '愤怒（呜哇哇哇）':'Angry','防御(响尾蛇嘶哈)':'Defence','打架（尖锐刺耳）':'Fighting',
    '快乐':'Happy','狩猎（短促的嗷呜）':'HuntingMind','交配（低沉嘶吼）':'Mating',
    '母体呼叫（警笛）':'MotherCall','疼痛':'Paining','休息':'Resting','警告（呜~）（咕噜）':'Warning'}

print(f'设备: {DEVICE}')
model=FDAPMobileCatCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH,map_location=DEVICE,weights_only=True))
model.eval()
print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')

eval_dataset(model, NEW_CAT_DIR, CN_MAP, '新猫语 38条')
eval_dataset(model, ORIG163_DIR, ORIG_CN_MAP, '原猫语 163条')

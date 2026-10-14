"""
增强 ch64 fold2 → 163条自标注数据验证
===============================
对比原始 fold1 的推理结果
"""
import sys
import time
import warnings

warnings.filterwarnings('ignore')
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import torch
from torch import nn

SR = 16000; N_MELS = 128; N_FFT = 512; HOP_LENGTH = 256
VAD_FRAMES = 100; VAD_TOP_DB = 25
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE = Path('/Desktop')
DATA_DIR = BASE / '原猫语分类/情绪分类'

LABELS = ["Angry","Defence","Fighting","Happy","HuntingMind",
          "Mating","MotherCall","Paining","Resting","Warning"]

# 中文目录 → Pandeya 标签
CN_MAP = {
    '愤怒（呜哇哇哇）': 'Angry', '防御(响尾蛇嘶哈)': 'Defence',
    '打架（尖锐刺耳）': 'Fighting', '快乐': 'Happy',
    '狩猎（短促的嗷呜）': 'HuntingMind', '交配（低沉嘶吼）': 'Mating',
    '母体呼叫（警笛）': 'MotherCall', '疼痛': 'Paining',
    '休息': 'Resting', '警告（呜~）（咕噜）': 'Warning',
}

# 6类合并 (跟之前一致)
MERGE_6 = {
    'Angry':'Aggressive', 'Defence':'Aggressive', 'Fighting':'Aggressive', 'Warning':'Aggressive',
    'Happy':'Happy', 'HuntingMind':'Happy',
    'Mating':'Mating', 'MotherCall':'MotherCall',
    'Paining':'Paining', 'Resting':'Resting',
}

# ─── 模型 ───
class FDAPMobileCatCNN_ch64(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((1,2)))
        self.conv3 = nn.Sequential(nn.Conv2d(32,48,3,padding=1), nn.BatchNorm2d(48), nn.ReLU(), nn.MaxPool2d((1,2)))
        self.conv4 = nn.Sequential(nn.Conv2d(48,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Identity())
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256,96), nn.ReLU(), nn.Dropout(0.3), nn.Linear(96,num_classes))
    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x); x = self.conv4(x)
        b,c,f,t = x.shape; bs = f // 4
        fdap = [x[:,:,i*bs:(i+1)*bs if i<3 else f,:].mean(dim=[2,3]) for i in range(4)]
        x = torch.cat(fdap, dim=1)
        return self.classifier(x)

def extract_longest_vocal(y, sr=SR, top_db=VAD_TOP_DB):
    segs = librosa.effects.split(y, top_db=top_db, frame_length=512, hop_length=HOP_LENGTH)
    if len(segs) == 0: return y
    longest = max(segs, key=lambda s: s[1]-s[0])
    return y[longest[0]:longest[1]]

def preprocess(audio_path):
    y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    y = extract_longest_vocal(y)
    peak = np.max(np.abs(y))
    y = y / peak if peak > 1e-10 else np.zeros_like(y)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    if mel_norm.shape[1] < VAD_FRAMES:
        mel_norm = np.pad(mel_norm, ((0,0), (0, VAD_FRAMES - mel_norm.shape[1])))
    else:
        s = (mel_norm.shape[1] - VAD_FRAMES) // 2
        mel_norm = mel_norm[:, s:s+VAD_FRAMES]
    return torch.FloatTensor(mel_norm).unsqueeze(0).unsqueeze(0)

# ─── 加载模型 ───
def load_model(path):
    model = FDAPMobileCatCNN_ch64().to(DEVICE)
    state = torch.load(str(path), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

# ─── 评估 ───
def evaluate(model_path, label):
    print(f'\n{"█"*50}')
    print(f'  {label}')
    print(f'  权重: {model_path.name}')
    print(f'{"█"*50}')
    
    model = load_model(model_path)
    
    total_correct_10 = 0
    total_correct_6 = 0
    total = 0
    
    for cn_dir, en_label in CN_MAP.items():
        merged_6 = MERGE_6[en_label]
        files = sorted((DATA_DIR / cn_dir).glob('*.wav'))
        if not files:
            print(f'  {cn_dir:<16} → 无文件')
            continue
        
        correct_10 = 0
        correct_6 = 0
        preds_10 = Counter()
        preds_6 = Counter()
        
        for fp in files:
            mel = preprocess(fp).to(DEVICE)
            with torch.no_grad():
                logits = model(mel)
                prob = torch.softmax(logits, dim=1)
                pred_10 = LABELS[prob.argmax().item()]
                pred_6 = MERGE_6[pred_10]
            
            preds_10[pred_10] += 1
            preds_6[pred_6] += 1
            
            if pred_10 == en_label:
                correct_10 += 1
            if pred_6 == merged_6:
                correct_6 += 1
        
        total_correct_10 += correct_10
        total_correct_6 += correct_6
        total += len(files)
        
        acc_10 = correct_10 / len(files)
        acc_6 = correct_6 / len(files)
        detail = ', '.join([f'{k}={v}' for k,v in preds_10.most_common(3)])
        emoji = '✅' if acc_6 >= 0.5 else '❌'
        print(f'  {emoji} {merged_6:<12} ({en_label:<12}) 10类:{correct_10}/{len(files)}={acc_10:.0%}  6类:{correct_6}/{len(files)}={acc_6:.0%}  → {detail}')
    
    print('\n  ─────────────────────────')
    print(f'  10类: {total_correct_10}/{total} = {total_correct_10/total:.1%}')
    print(f'  6类:  {total_correct_6}/{total} = {total_correct_6/total:.1%}')
    
    return total_correct_6 / total

# ─── 主程序 ───
if __name__ == '__main__':
    print(f'设备: {DEVICE}')
    t0 = time.time()
    
    # 原始 fold1
    acc_orig = evaluate(
        BASE / '模型代码/训练结果/fdap_cnn_ch64/fold1_best.pth',
        '原始 ch64 fold1'
    )
    
    # 增强 fold2
    acc_aug = evaluate(
        BASE / '模型代码/训练结果/fdap_cnn_ch64_aug/fold2_best.pth',
        '增强 ch64 fold2'
    )
    
    print(f'\n{"█"*50}')
    print('  对比')
    print(f'  原始 fold1 6类: {acc_orig:.1%}')
    print(f'  增强 fold2 6类: {acc_aug:.1%}')
    delta = acc_aug - acc_orig
    arrow = '↑' if delta > 0 else '↓'
    print(f'  变化: {delta:+.1%} {arrow}')
    print(f'  耗时: {time.time()-t0:.0f}s')
    print(f'{"█"*50}')

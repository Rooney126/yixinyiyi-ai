"""
CNN + FDAP 推理 40 条真实猫声
对比 HuBERT + GAP (之前37.5%) 的结果
# FDAP 特征 → 新猫语 40条推理（Pandeya CNN 路线）
"""

import os, sys, json, warnings, h5py
warnings.filterwarnings('ignore')
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline

# ── 路径 ──
BASE = Path(r'/path/to/dataset')
MODEL_PATH = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X/model_best.hdf5'
CACHE_DIR = BASE / '模型代码/fdap_cache'
DATA40_DIR = BASE / '新猫语分类/新情绪分类'
OUTPUT = BASE / '模型代码/结果_FDAP_40条推理.json'

# ── 配置 ──
SR = 12000
CROP_SAMPLES = int(SR * 5.0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MEL_CONFIG = dict(sr=12000, n_fft=512, hop_length=256, n_mels=96, fmin=0.0, fmax=6000)
FDAP_SEGMENTS = [1, 1, 4, 4, 3]

# 标签映射
CN_TO_PANDEYA = {
    '友好': 'Happy', '吸引注意': 'Mating', '愤怒示威': 'Angry',
    '母体呼唤': 'MotherCall', '渴求': 'Mating',
    '疼痛惊吓': 'Paining', '警告驱逐': 'Warning',
}
MERGE_6 = {'Angry':'Aggressive','Defence':'Aggressive','Fighting':'Aggressive','Warning':'Aggressive',
           'Happy':'Happy','HuntingMind':'Happy','Mating':'Mating',
           'MotherCall':'MotherCall','Paining':'Paining','Resting':'Resting'}
PANDEYA_LABELS = ['Angry','Defence','Fighting','Happy','HuntingMind',
                  'Mating','MotherCall','Paining','Resting','Warning']

print('═'*60)
print('CNN + FDAP → 40条真实猫声推理')
print(f'设备: {DEVICE}')
print('═'*60)

# ═══════════════════════════════════
# 1. 重建 CNN + 加载权重 (同 FDAP t-SNE)
# ═══════════════════════════════════
print('\n[1/4] 重建 CNN + 加载权重...')

class SafeMaxPool2d(nn.Module):
    def __init__(self, kernel_size): super().__init__(); self.kernel_size = kernel_size
    def forward(self, x):
        _,_,H,W = x.shape; kH,kW = self.kernel_size
        pool_h = kH if H >= kH else 1; pool_w = kW if W >= kW else 1
        if pool_h == 1 and pool_w == 1: return x
        return F.max_pool2d(x, (pool_h, pool_w))

class PandeyaCNN(nn.Module):
    def __init__(self, conv_weights, bn_weights):
        super().__init__(); self.blocks = nn.ModuleList()
        for i in range(5):
            kw, bw = conv_weights[i], bn_weights[i]
            w = torch.from_numpy(kw['kernel:0']).permute(3,2,0,1).float()
            b = torch.from_numpy(kw['bias:0']).float()
            conv = nn.Conv2d(w.shape[1], w.shape[0], 3, padding=1)
            conv.weight.data = w; conv.bias.data = b
            bn = nn.BatchNorm2d(w.shape[0])
            bn.weight.data = torch.from_numpy(bw['gamma:0']).float()
            bn.bias.data = torch.from_numpy(bw['beta:0']).float()
            bn.running_mean.data = torch.from_numpy(bw['moving_mean:0']).float()
            bn.running_var.data = torch.from_numpy(bw['moving_variance:0']).float()
            pool = SafeMaxPool2d([(2,4),(3,4),(2,5),(2,4),(4,4)][i])
            self.blocks.append(nn.Sequential(conv, bn, nn.ELU(), pool))
    def forward(self, x):
        outputs = []
        for block in self.blocks:
            x = block(x); outputs.append(x)
        return outputs

def read_h5_weight(f, name):
    variants = [f'model_weights/{name}/{name}/{k}:0' for k in ['kernel','bias','gamma','beta','moving_mean','moving_variance']]
    return {v.split('/')[-1]: f[v][()] for v in variants if v in f}

f_h5 = h5py.File(str(MODEL_PATH), 'r')
conv_w = [read_h5_weight(f_h5, f'conv2d_{i}') for i in range(1,6)]
bn_w = [read_h5_weight(f_h5, f'batch_normalization_{i}') for i in range(1,6)]
f_h5.close()

cnn = PandeyaCNN(conv_w, bn_w).to(DEVICE)
cnn.eval()
print(f'  ✅ CNN {sum(p.numel() for p in cnn.parameters()):,} 参数')

# ═══════════════════════════════════
# 2. 加载 CatSound_V2 FDAP 特征 (缓存) + 训练 SVM
# ═══════════════════════════════════
print('\n[2/4] 加载 CatSound_V2 FDAP + 训练 SVM...')
fdap_cache = sorted(CACHE_DIR.glob('*_fdap_*.npy'))[-1]
label_cache = sorted(CACHE_DIR.glob('*_labels_*.npy'))[-1]
X_train = np.load(fdap_cache)
y_train = np.load(label_cache).tolist()
print(f'  训练集: {X_train.shape}, {len(y_train)} 条')

# 6类标签
y_train_6 = [MERGE_6.get(l,l) for l in y_train]

clf = make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=10000, dual='auto', random_state=42))
clf.fit(X_train, y_train_6)
print(f'  ✅ SVM 训练完成')

# ═══════════════════════════════════
# 3. FDAP 提取 40条音频
# ═══════════════════════════════════
print('\n[3/4] FDAP 提取 40条音频...')

def extract_fdap(audio_path, cnn, device):
    y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    if len(y) > CROP_SAMPLES:
        start = (len(y) - CROP_SAMPLES) // 2
        y = y[start:start + CROP_SAMPLES]
    else:
        pad = CROP_SAMPLES - len(y)
        y = np.pad(y, (pad // 2, pad - pad // 2))
    mel = librosa.feature.melspectrogram(y=y, **MEL_CONFIG, power=2.0, center=True)
    mel_db = librosa.power_to_db(mel, ref=max(mel.max(), 1e-10))
    x = np.expand_dims(np.expand_dims(mel_db.T, 0), 0)  # (1,1,T,96)
    # 每频段独立归一化 (匹配 Kapre)
    mel_mean = x.mean(axis=2, keepdims=True)
    mel_std = x.std(axis=2, keepdims=True) + 1e-5
    x_norm = (x - mel_mean) / mel_std
    x_t = torch.from_numpy(x_norm).float().to(device)
    with torch.no_grad():
        outputs = cnn(x_t)
    fdap_vecs = []
    for idx, feat in enumerate(outputs):
        f = feat[0].cpu().numpy()
        f_p = np.transpose(f, (2, 1, 0))  # (F,T,C)
        H = f_p.shape[0]; n_seg = FDAP_SEGMENTS[idx]; seg_len = max(H // n_seg, 1)
        for si in range(n_seg):
            s = si * seg_len
            e = s + seg_len if si < n_seg - 1 else H
            seg = f_p[s:e, :, :]
            fdap_vecs.append(np.mean(seg, axis=(0,1)))
            fdap_vecs.append(np.std(seg, axis=(0,1)))
    vec = np.concatenate(fdap_vecs).reshape(1, -1)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

all_files = []
for subdir in sorted(DATA40_DIR.iterdir()):
    if not subdir.is_dir(): continue
    for f in sorted(subdir.glob('*')):
        all_files.append((f, subdir.name))

results = []
correct = 0
total = 0
correct_by_cn = defaultdict(lambda: {'c':0,'t':0})

for idx, (path, cn_label) in enumerate(all_files):
    expected = MERGE_6.get(CN_TO_PANDEYA.get(cn_label,'?'), '?')
    try:
        fdap = extract_fdap(path, cnn, DEVICE)
        pred = clf.predict(fdap)[0]
        prob = clf.decision_function(fdap)[0]
        # top5
        top_indices = np.argsort(prob)[::-1][:5]
        classes = clf.classes_
        # softmax 转概率
        prob_s = prob - prob.max()
        prob_probs = np.exp(prob_s) / np.exp(prob_s).sum()
        top5 = [{'label': classes[i], 'prob': round(float(prob_probs[i]), 4)} for i in top_indices[:5]]
    except Exception as e:
        print(f'  [FAIL] {path.name}: {e}')
        pred = 'ERR'
        top5 = []
    
    total += 1
    if pred == expected:
        correct += 1
        correct_by_cn[cn_label]['c'] += 1
    correct_by_cn[cn_label]['t'] += 1
    mark = '✅' if pred == expected else '❌'
    print(f'({idx+1:2d}) {mark} {path.name[:30]:30s} {cn_label:8s} → {pred:<12s} (期望:{expected})')
    
    results.append({
        'file': path.name, 'cn_label': cn_label, 'expected': expected,
        'pred': pred, 'top5': top5,
    })

# ── 汇总 ──
print(f'\n{"═"*60}')
print(f'CNN+FDAP 40条: {correct}/{total} = {correct/total*100:.1f}%')
print(f'HuBERT+GAP 40条(之前): 15/40 = 37.5%')
print(f'{"═"*60}\n')

print('各类正确率:')
for cn in sorted(correct_by_cn.keys()):
    info = correct_by_cn[cn]
    r = info['c']/info['t']*100 if info['t'] > 0 else 0
    print(f'  {cn:8s}: {info["c"]}/{info["t"]} ({r:.0f}%)')

with open(OUTPUT, 'w', encoding='utf-8') as f:
    json.dump({
        'accuracy': round(correct/total, 4),
        'correct': correct, 'total': total,
        'per_category': {k: {'correct': v['c'], 'total': v['t'],
                             'accuracy': round(v['c']/v['t'], 4) if v['t']>0 else 0}
                         for k,v in correct_by_cn.items()},
        'results': results,
    }, f, ensure_ascii=False, indent=2)
print(f'\n结果 → {OUTPUT}')

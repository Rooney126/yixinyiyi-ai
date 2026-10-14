"""
HuBERT freeze SVM 纯原始版 (不含 _aug1)
和 FDAP (2961条) 公平对比
# HuBERT + 未增强原始音频（2961条）GroupKFold 对比实验
"""

import os, sys, json, warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

import numpy as np
import torch
import librosa
import joblib
from pathlib import Path
from collections import Counter
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from transformers import HubertModel, Wav2Vec2FeatureExtractor

# ── 配置 ──
SR = 16000
CROP_SAMPLES = int(SR * 5.0)
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

BASE = Path(r'/path/to/dataset')
DATA_DIR = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
CACHE_DIR = BASE / '模型代码/hubert_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = BASE / '模型代码/结果_HuBERT_原始版SVN对比.json'

LABELS_10 = ['Angry','Defence','Fighting','Happy','HuntingMind',
             'Mating','MotherCall','Paining','Resting','Warning']

MERGE_6 = {
    'Angry':'Aggressive','Defence':'Aggressive','Fighting':'Aggressive','Warning':'Aggressive',
    'Happy':'Happy','HuntingMind':'Happy','Mating':'Mating',
    'MotherCall':'MotherCall','Paining':'Paining','Resting':'Resting',
}

print('═'*50)
print('HuBERT freeze SVM — 纯原始版 (不含 _aug1)')
print(f'设备: {DEVICE}')
print('═'*50)

# ── 1. 加载音频 ──
print('\n[1/3] 加载音频 (仅原始)...')
samples = []
for label in LABELS_10:
    d = DATA_DIR / label
    if not d.exists():
        continue
    for fp in sorted(d.glob('*.mp3')):
        if '_aug1' in fp.stem:
            continue
        try:
            y,_ = librosa.load(str(fp), sr=SR, mono=True)
            if len(y) > CROP_SAMPLES:
                start = (len(y) - CROP_SAMPLES) // 2
                y = y[start:start + CROP_SAMPLES]
            else:
                pad = CROP_SAMPLES - len(y)
                y = np.pad(y, (pad // 2, pad - pad // 2))
            samples.append({'waveform': y.astype(np.float32), 'label': label})
        except:
            pass

n = len(samples)
labels_all = [s['label'] for s in samples]
print(f'  共 {n} 条')
for l, c in Counter(labels_all).most_common():
    print(f'    {l:<12}: {c:4d}')

# ── 2. HuBERT 嵌入 ──
print('\n[2/3] HuBERT 嵌入...')
cache_file = CACHE_DIR / f'hubert_originals_{n}.npy'
cache_labels = CACHE_DIR / f'hubert_originals_labels_{n}.npy'

if cache_file.exists():
    embs = np.load(cache_file)
    labels_loaded = np.load(cache_labels).tolist()
    print(f'  缓存命中: {embs.shape}')
else:
    fe = Wav2Vec2FeatureExtractor.from_pretrained('facebook/hubert-base-ls960', local_files_only=True)
    model = HubertModel.from_pretrained('facebook/hubert-base-ls960', local_files_only=True).to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    all_embs = []
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            batch = samples[i:i+BATCH_SIZE]
            waves = [s['waveform'] for s in batch]
            inputs = fe(waves, sampling_rate=SR, return_tensors='pt', padding=True).to(DEVICE)
            outputs = model(inputs.input_values, output_hidden_states=True)
            hs = outputs.hidden_states
            emb = torch.stack([hs[l] for l in [3,4,5]], dim=0).mean(dim=0).mean(dim=1).cpu().numpy()
            all_embs.append(emb)
    
    embs = np.concatenate(all_embs, axis=0)
    np.save(cache_file, embs)
    np.save(cache_labels, np.array(labels_all))
    print(f'  嵌入: {embs.shape}')

# ── 3. SVM 交叉验证 ──
print('\n[3/3] SVM 交叉验证...')
le = LabelEncoder()
y = le.fit_transform(labels_all)
y6 = le.fit_transform([MERGE_6.get(l,l) for l in labels_all])

results = {}
for (mode_name, y_, key) in [('10类', y, 'hubert_10c'), ('6类', y6, 'hubert_6c')]:
    print(f'\n  {mode_name} LinearSVM (5折):')
    pipe = make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=10000, dual='auto', random_state=42))
    scores = cross_val_score(pipe, embs, y_, cv=5, scoring='accuracy')
    print(f'    {scores}')
    print(f'    均值: {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%')
    results[key] = {'mean': round(float(scores.mean()),4), 'std': round(float(scores.std()),4)}

print(f'\n{"═"*50}')
print('对比 (均只含原始音频):')
print(f'  HuBERT 原始({n}条) 10类: {results["hubert_10c"]["mean"]*100:.1f}%')
print(f'  HuBERT 原始({n}条) 6类:  {results["hubert_6c"]["mean"]*100:.1f}%')
print(f'  CNN+FDAP 原始({n}条) 10类: 65.1%')
print(f'  CNN+GAP  原始({n}条) 10类: 65.1%')
print(f'{"═"*50}')

with open(OUTPUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\n结果已保存: {OUTPUT}')

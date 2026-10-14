"""
事件分割实验: CatSound_V2 每文件切成单段猫叫 → HuBERT freeze + SVM
对比: 原始全文件训练 CV vs 单段训练 CV
# 事件分割实验：叫声/非叫声二分类切割
"""

import os, sys, time, json, warnings
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupKFold
from transformers import HubertModel, Wav2Vec2FeatureExtractor

SR = 16000
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYERS = [3, 4, 5]  # HuBERT 最优层

BASE = Path(r'/path/to/dataset')
DATA_DIR = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
LABELS = [
    'Angry', 'Defence', 'Fighting', 'Happy', 'HuntingMind',
    'Mating', 'MotherCall', 'Paining', 'Resting', 'Warning'
]

print('═' * 65)
print('  事件分割 + HuBERT freeze + SVM 实验')
print('═' * 65)

# ── 1. 事件分割 ──
print('\n[1/4] 事件分割...')
segments = []  # [{path, label, cat_id}], cat_id = 原文件 stem
files_processed = 0
seg_count = 0
zero_files = 0

for label in LABELS:
    dir_path = DATA_DIR / label
    if not dir_path.exists(): continue
    for fp in sorted(dir_path.glob('*.mp3')):
        files_processed += 1
        try:
            y, _sr = librosa.load(str(fp), sr=SR, mono=True)
            intervals = librosa.effects.split(y, top_db=30,
                                               frame_length=2048, hop_length=512)
            if len(intervals) == 0:
                zero_files += 1
                continue
            
            stem = fp.stem
            cat_id = stem.replace('_aug1(1)', '').lower()
            
            for si, (start, end) in enumerate(intervals):
                seg = y[start:end]
                dur = len(seg) / SR
                if dur > 0.15:  # 过滤太短
                    segments.append({
                        'waveform': seg,
                        'label': label,
                        'cat_id': cat_id,
                        'orig_file': str(fp.name),
                        'seg_idx': si,
                    })
                    seg_count += 1
        except Exception as e:
            zero_files += 1

print(f'  处理 {files_processed} 个文件 → {seg_count} 个单段')
print(f'  平均 {seg_count/files_processed:.2f} 段/文件')
print(f'  无声/失败文件: {zero_files}')

# 段长分布分析
seg_durations = np.array([len(s['waveform']) / SR for s in segments])
print(f'\n  段长分析:')
print(f'    平均: {seg_durations.mean():.2f}s, 中位数: {np.median(seg_durations):.2f}s')
print(f'    最短: {seg_durations.min():.2f}s, 最长: {seg_durations.max():.2f}s')
print(f'    <0.3s: {(seg_durations < 0.3).sum()}, <0.5s: {(seg_durations < 0.5).sum()}')
print(f'    >2.0s: {(seg_durations > 2.0).sum()}, >3.0s: {(seg_durations > 3.0).sum()}')

# 各类段数统计
for label in LABELS:
    n = sum(1 for s in segments if s['label'] == label)
    n_files = sum(1 for s in segments if s['label'] == label)
    print(f'    {label:>12s}: {n} 段')

# ── 2. 加载 HuBERT ──
print('\n[2/4] 加载 HuBERT...')
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    'facebook/hubert-base-ls960', local_files_only=True)
hubert = HubertModel.from_pretrained(
    'facebook/hubert-base-ls960', local_files_only=True).to(DEVICE)
hubert.eval()
print(f'  设备: {DEVICE}')

# ── 3. 提取嵌入 ──
print('\n[3/4] 提取嵌入...')
all_embs = []
all_labels = [s['label'] for s in segments]
all_cats = [s['cat_id'] for s in segments]

t0 = time.time()
with torch.no_grad():
    for i in range(0, len(segments), BATCH_SIZE):
        batch = segments[i:i+BATCH_SIZE]
        waveforms = [s['waveform'] for s in batch]
        
        inputs = feature_extractor(
            waveforms, sampling_rate=SR,
            return_tensors='pt', padding=True,
            truncation=True, max_length=SR * 3  # 单段 <3s
        ).to(DEVICE)
        
        outputs = hubert(inputs.input_values, output_hidden_states=True)
        hs = outputs.hidden_states
        emb = torch.stack([hs[l] for l in LAYERS], dim=0).mean(dim=0).mean(dim=1)
        all_embs.append(emb.cpu().numpy())

X = np.concatenate(all_embs, axis=0)
le = LabelEncoder()
y = le.fit_transform(all_labels)
print(f'  完成 ({time.time()-t0:.1f}s), X={X.shape}, y={y.shape}')

# ── 4. GroupKFold SVM ──
print('\n[4/4] GroupKFold SVM...')
gkf = GroupKFold(n_splits=5)
fold_accs = []

for fold, (tr, te) in enumerate(gkf.split(segments, y, all_cats), 1):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr])
    X_te = scaler.transform(X[te])
    svm = LinearSVC(C=1.0, max_iter=10000, dual='auto', class_weight='balanced')
    svm.fit(X_tr, y[tr])
    acc = svm.score(X_te, y[te])
    fold_accs.append(acc)
    print(f'  Fold {fold}: {acc:.1%}')

mean_acc = np.mean(fold_accs)
std_acc = np.std(fold_accs)
print(f'\n  事件分割 SVM 5折 CV: {mean_acc:.1%} ± {std_acc:.1%}')
print(f'  原文 (全文件) SVM 5折 CV: 82.2% ± 1.3%')
print(f'  差异: {mean_acc - 0.822:+.1%}')

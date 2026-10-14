"""
事件分割实验 v2: CatSound_V2 事件分割 CV + 新猫语推理
对比:
  1. CatSound_V2 全文件 CV (baseline 82.2%)
  2. CatSound_V2 事件分割 CV (看分割是否破坏信息)
  3. 新猫语 全文件 推理 → 2s center crop (baseline 44%)
  4. 新猫语 事件分割 → 逐段投票 (看数据形态是否主因)
# 事件分割实验 v2：检测猫叫声起始/结束位置
"""

import os, sys, time, json, warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from collections import Counter, defaultdict
import numpy as np
import torch
import librosa
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupKFold
from transformers import HubertModel, Wav2Vec2FeatureExtractor

# ── 配置 ──
SR = 16000
CROP = int(SR * 2.0)  # 2s center crop
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYERS = [3, 4, 5]

BASE = Path(r'/path/to/dataset')
DATA_DIR = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
NEW_CAT_DIR = BASE / '新猫语分类/新情绪分类'
MODEL_DIR = BASE / '模型代码/训练结果/hubert_v2/freeze_model_hubert_20260618_111717'

LABELS_10 = [
    'Angry', 'Defence', 'Fighting', 'Happy', 'HuntingMind',
    'Mating', 'MotherCall', 'Paining', 'Resting', 'Warning'
]

# 6 类合并（来自 MEMORY.md 6→7 穹顶方案）
MERGE_10 = {
    'Angry':'Aggressive', 'Defence':'Aggressive',
    'Fighting':'Aggressive', 'Warning':'Aggressive',
    'Happy':'Happy', 'HuntingMind':'Happy',
    'Mating':'Mating', 'MotherCall':'MotherCall',
    'Paining':'Paining', 'Resting':'Resting',
}

# 新猫语 5 个用户标签（不与 Pandeya 6 类一一对应）
NEW_LABELS = ['友好', '愤怒示威', '吸引注意', '渴求', '警告驱逐']

print('═' * 65)
print('  事件分割实验 v2 — CatSound_V2 CV + 新猫语推理')
print('═' * 65)


def center_crop(y):
    if len(y) > CROP:
        start = (len(y) - CROP) // 2
        y = y[start:start + CROP]
    else:
        pad = CROP - len(y)
        y = np.pad(y, (pad // 2, pad - pad // 2))
    return y


# =========================================================
# 1. CatSound_V2 事件分割
# =========================================================
print('\n[1/5] CatSound_V2 事件分割...')
segments = []
files_processed = 0
seg_count = 0
zero_files = 0

for label in LABELS_10:
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
                if dur > 0.15:
                    segments.append({
                        'waveform': seg, 'label': label,
                        'cat_id': cat_id, 'orig_file': str(fp.name), 'seg_idx': si,
                    })
                    seg_count += 1
        except:
            zero_files += 1

print(f'  处理 {files_processed} 个文件 → {seg_count} 个单段 ({seg_count/files_processed:.2f} 段/文件)')
print(f'  无声/失败: {zero_files}')

seg_durations = np.array([len(s['waveform']) / SR for s in segments])
print(f'  段长: 平均 {seg_durations.mean():.2f}s, 中位 {np.median(seg_durations):.2f}s')


# =========================================================
# 2. 加载 HuBERT
# =========================================================
print('\n[2/5] 加载 HuBERT...')
t0 = time.time()
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    'facebook/hubert-base-ls960', local_files_only=True)
hubert = HubertModel.from_pretrained(
    'facebook/hubert-base-ls960', local_files_only=True).to(DEVICE)
hubert.eval()
print(f'  完成 ({time.time()-t0:.1f}s), 设备: {DEVICE}')


def extract_embeddings_batch(waveforms, max_len=SR*3):
    """批量提取嵌入"""
    inputs = feature_extractor(
        waveforms, sampling_rate=SR, return_tensors='pt', padding=True,
        truncation=True, max_length=max_len
    ).to(DEVICE)
    with torch.no_grad():
        outputs = hubert(inputs.input_values, output_hidden_states=True)
        hs = outputs.hidden_states
        emb = torch.stack([hs[l] for l in LAYERS], dim=0).mean(dim=0).mean(dim=1)
    return emb.cpu().numpy()


# =========================================================
# 3. CatSound_V2 事件分割版 CV
# =========================================================
print('\n[3/5] CatSound_V2 事件分割 GroupKFold SVM CV...')
all_embs = []
all_labels = [s['label'] for s in segments]
all_cats = [s['cat_id'] for s in segments]

for i in range(0, len(segments), BATCH_SIZE):
    batch = segments[i:i+BATCH_SIZE]
    embs = extract_embeddings_batch([s['waveform'] for s in batch])
    all_embs.append(embs)
    if (i // BATCH_SIZE) % 20 == 0:
        print(f'  嵌入: {i+len(batch)}/{len(segments)}')

X_seg = np.concatenate(all_embs, axis=0)
le_seg = LabelEncoder()
y_seg = le_seg.fit_transform(all_labels)
print(f'  嵌入完成, X={X_seg.shape}')

gkf = GroupKFold(n_splits=5)
fold_accs = []
for fold, (tr, te) in enumerate(gkf.split(segments, y_seg, all_cats), 1):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_seg[tr])
    X_te = scaler.transform(X_seg[te])
    svm = LinearSVC(C=1.0, max_iter=10000, dual='auto', class_weight='balanced')
    svm.fit(X_tr, y_seg[tr])
    acc = svm.score(X_te, y_seg[te])
    fold_accs.append(acc)
    print(f'  Fold {fold}: {acc:.1%}')

mean_acc_seg = np.mean(fold_accs)
std_acc_seg = np.std(fold_accs)
print(f'\n  事件分割 CV: {mean_acc_seg:.1%} ± {std_acc_seg:.1%}')
print(f'  全文件 CV (参考): 82.2% ± 1.3%')
print(f'  差异: {mean_acc_seg - 0.822:+.1%}')


# =========================================================
# 4. 加载已有模型
# =========================================================
print(f'\n[4/5] 加载已有 freeze_model: {MODEL_DIR.name}')
svm_data = joblib.load(str(MODEL_DIR / 'svm.pkl'))
svm_full, scaler_full, le_full = svm_data['svm'], svm_data['scaler'], svm_data['label_encoder']
print(f'  模型: SVM 10类, 标签={list(le_full.classes_)}')

# =========================================================
# 5. 新猫语推理
# =========================================================
print('\n[5/5] 新猫语推理')

# ── 5a. 加载 ──
new_cat_files = []
for cn_dir in sorted(NEW_CAT_DIR.iterdir()):
    if not cn_dir.is_dir(): continue
    for fp in sorted(cn_dir.glob('*.wav')):
        try:
            y, _ = librosa.load(str(fp), sr=SR, mono=True)
            new_cat_files.append({'waveform': y.astype(np.float32), 'label_cn': cn_dir.name, 'path': str(fp)})
        except:
            pass

print(f'\n  新猫语: {len(new_cat_files)} 个文件')
for l, n in Counter(f['label_cn'] for f in new_cat_files).most_common():
    print(f'    {l}: {n}')

# ── 5b. 全文件推理 (2s center crop) ──
print('\n  [5b] 全文件推理 (2s center crop, baseline)...')
correct_file, total_file = 0, 0
per_class_file = defaultdict(lambda: {'total': 0, 'correct': 0})
pred_details_file = []

for f in new_cat_files:
    y = center_crop(f['waveform'])
    peak = np.max(np.abs(y))
    y_norm = y / peak if peak > 1e-10 else y
    emb = extract_embeddings_batch([y_norm])  # (1, 768)
    emb_s = scaler_full.transform(emb)
    pred = svm_full.predict(emb_s)[0]
    pred_label = le_full.inverse_transform([pred])[0]
    pred_6 = MERGE_10.get(pred_label, pred_label)
    
    pred_details_file.append({
        'file': Path(f['path']).name, 'true': f['label_cn'],
        'pred_10': pred_label, 'pred_6': pred_6,
    })

print(f'\n  ── 全文件推理结果（10类 / 6类）──')
for d in pred_details_file:
    print(f'    {d["file"][:40]:<42} 真={d["true"]:<8}  10类={d["pred_10"]:<12}  6类={d["pred_6"]}')

# 按真实标签汇总预测分布
print(f'\n  ── 按新猫语标签汇总预测分布 ──')
for label in NEW_LABELS:
    items = [d for d in pred_details_file if d['true'] == label]
    if not items: continue
    from collections import Counter
    c10 = Counter(d['pred_10'] for d in items)
    c6 = Counter(d['pred_6'] for d in items)
    print(f'  {label}: {len(items)} 个')
    print(f'    10类分布: {dict(c10.most_common())}')
    print(f'    6类分布:  {dict(c6.most_common())}')

# ── 5c. 事件分割→逐段推理→投票 ──
print('\n  [5c] 事件分割 → 逐段推理 → 投票...')
correct_seg, total_seg = 0, 0
per_class_seg = defaultdict(lambda: {'total': 0, 'correct': 0})
pred_details_seg = []

for f in new_cat_files:
    y = f['waveform']
    intervals = librosa.effects.split(y, top_db=30, frame_length=2048, hop_length=512)
    segments_new = []
    for start, end in intervals:
        seg = y[start:end]
        if len(seg) / SR > 0.15:
            segments_new.append(seg)
    if len(segments_new) == 0:
        segments_new = [y]  # fallback
    
    # 批量提嵌入
    seg_embs = []
    for seg in segments_new:
        peak = np.max(np.abs(seg))
        seg_norm = seg / peak if peak > 1e-10 else seg
        seg_embs.append(seg_norm)
    
    all_seg_embs = extract_embeddings_batch(seg_embs)
    all_seg_s = scaler_full.transform(all_seg_embs)
    seg_preds = svm_full.predict(all_seg_s)
    
    # 收集投票
    votes_raw = [le_full.inverse_transform([p])[0] for p in seg_preds]
    votes_6 = [MERGE_10.get(v, v) for v in votes_raw]
    vote_counter_10 = Counter(votes_raw)
    vote_counter_6 = Counter(votes_6)
    final_pred_10 = vote_counter_10.most_common(1)[0][0]
    final_pred_6 = vote_counter_6.most_common(1)[0][0]
    
    vote_detail = ', '.join(f'{k}({v})' for k, v in vote_counter_10.most_common(3))
    pred_details_seg.append({
        'file': Path(f['path']).name, 'true': f['label_cn'],
        'segments': len(segments_new),
        'pred_10_vote': final_pred_10, 'pred_6_vote': final_pred_6,
        'votes_detail': vote_detail,
    })

print(f'\n  ── 事件分割→投票结果（10类 / 6类）──')
for d in pred_details_seg:
    print(f'    {d["file"][:35]:<37} 真={d["true"]:<8} 段={d["segments"]} 投10类={d["pred_10_vote"]:<12} 投6类={d["pred_6_vote"]}')

print(f'\n  ── 按新猫语标签汇总投票分布 ──')
for label in NEW_LABELS:
    items = [d for d in pred_details_seg if d['true'] == label]
    if not items: continue
    c10 = Counter(d['pred_10_vote'] for d in items)
    c6 = Counter(d['pred_6_vote'] for d in items)
    print(f'  {label}: {len(items)} 个')
    print(f'    10类分布: {dict(c10.most_common())}')
    print(f'    6类分布:  {dict(c6.most_common())}')


# =========================================================
# 汇总
# =========================================================
print(f'\n{"="*60}')
print(f'  CatSound_V2 内部验证')
print(f'{"="*60}')
print(f'  全文件 CV (参考):        82.2% ± 1.3%')
print(f'  事件分割 CV:              {mean_acc_seg:.1%} ± {std_acc_seg:.1%}')
print(f'  → 差异: {mean_acc_seg-0.822:+.1%}')
print(f'')
print(f'  新猫语 40 条预测分布已输出，标签映射由项目负责人判断')

result = {
    'catv2_seg_cv_mean': mean_acc_seg, 'catv2_seg_cv_std': std_acc_seg,
    'catv2_full_cv_ref': 0.822,
    'pred_details_file': pred_details_file,
    'pred_details_seg': pred_details_seg,
}
ts = time.strftime('%Y%m%d_%H%M%S')
result_path = BASE / f'模型代码/event_seg_result_{ts}.json'
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False, default=str)
print(f'\n✅ 结果保存: {result_path}')

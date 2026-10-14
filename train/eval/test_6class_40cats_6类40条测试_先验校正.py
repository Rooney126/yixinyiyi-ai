"""
6类情绪穹顶模型 + 先验校正 → 新猫语 40条音频推理
准确率 + 混淆矩阵
# 6类模型 + 先验校正 → 新猫语 40条测试
"""

import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ['HF_HOME'] = '/.cache/huggingface'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from collections import Counter, defaultdict
from pathlib import Path

import joblib
import librosa
import numpy as np
import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor

# ── 路径 ──
BASE = Path('/Desktop')
DATA_DIR = BASE / '新猫语分类/新情绪分类'
MODEL_DIR = BASE / '模型代码/训练结果/hubert_v2/freeze_model_hubert_20260618_152123'

SR = 16000
CROP_SAMPLES = int(SR * 5.0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── 6类 → 中文标签映射 ──
EMOTION_6 = ['Aggressive', 'Happy', 'Mating', 'MotherCall', 'Paining', 'Resting']
EMOTION_CN = ['😾 愤怒/威胁', '😊 友好/兴奋', '💕 渴求/吸引', '📞 母体呼唤', '😖 疼痛/惊吓', '😴 满足/休息']

# 目录名 → 6类映射（用户的标注方式）
CN_TO_6 = {
    '友好': 'Happy',
    '吸引注意': 'Mating',
    '愤怒示威': 'Aggressive',
    '母体呼唤': 'MotherCall',
    '渴求': 'Mating',
    '疼痛惊吓': 'Paining',
    '警告驱逐': 'Aggressive',
}

# 先验校正系数（CatSound_V2 5922条）
CLASS_PRIORS = np.array([0.4022, 0.1979, 0.1017, 0.1000, 0.0983, 0.1000])
UNIFORM_PRIOR = 1.0 / 6

# ── 加载模型 ──
print(f'设备: {DEVICE}')
print('加载模型...')
data = joblib.load(str(MODEL_DIR / 'svm.pkl'))
svm = data['svm']
scaler = data['scaler']
le = data['label_encoder']
print(f'  SVM 类别: {le.classes_.tolist()}')
print(f'  class_weight: {svm.class_weight}')

model = HubertModel.from_pretrained('facebook/hubert-base-ls960', local_files_only=True)
model.to(DEVICE)
model.eval()
for p in model.parameters(): p.requires_grad = False
processor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/hubert-base-ls960', local_files_only=True)
print('模型加载完成')
print()

# ── 获取所有音频 ──
audio_files = []
for cn_dir in CN_TO_6:
    dir_path = DATA_DIR / cn_dir
    if not dir_path.exists():
        continue
    for f in sorted(dir_path.iterdir()):
        if f.suffix.lower() in ('.wav', '.mp3', '.m4a', '.webm'):
            audio_files.append((str(f), cn_dir, CN_TO_6[cn_dir]))

print(f'找到 {len(audio_files)} 条音频')
print()

# ── 逐条推理 ──
results = []
correct_6 = 0
total = 0
confusion = Counter()  # (true, pred)
per_class = defaultdict(lambda: {'correct': 0, 'total': 0})

for filepath, cn_label, gt_6 in audio_files:
    try:
        y, _ = librosa.load(filepath, sr=SR, mono=True)
    except Exception as e:
        print(f'  ❌ 加载失败: {filepath} → {e}')
        continue

    total += 1
    per_class[gt_6]['total'] += 1

    # 归一化
    peak = np.max(np.abs(y))
    if peak > 1e-10:
        y = y / peak
    # 裁剪/填充到 5s
    if len(y) > CROP_SAMPLES:
        start = (len(y) - CROP_SAMPLES) // 2
        y = y[start:start + CROP_SAMPLES]
    else:
        pad = CROP_SAMPLES - len(y)
        y = np.pad(y, (pad // 2, pad - pad // 2))

    with torch.no_grad():
        inputs = processor([y], sampling_rate=SR, return_tensors='pt', padding=True).to(DEVICE)
        outputs = model(inputs.input_values, output_hidden_states=True)
        hs = outputs.hidden_states
        emb = torch.stack([hs[i] for i in [3, 4, 5]], dim=0).mean(dim=0).mean(dim=1).cpu().numpy()

    emb_scaled = scaler.transform(emb)
    raw_scores = svm.decision_function(emb_scaled)[0]

    # 先验校正
    adjusted = raw_scores - np.log(CLASS_PRIORS) + np.log(UNIFORM_PRIOR)
    adjusted = adjusted - adjusted.max()
    probs = np.exp(adjusted) / np.exp(adjusted).sum()
    pred_6 = le.classes_[np.argmax(probs)]

    # 无校正版本（对比）
    raw_adj = raw_scores - raw_scores.max()
    raw_probs = np.exp(raw_adj) / np.exp(raw_adj).sum()
    pred_raw = le.classes_[np.argmax(raw_probs)]

    is_correct = pred_6 == gt_6
    if is_correct:
        correct_6 += 1
        per_class[gt_6]['correct'] += 1

    confusion[(gt_6, pred_6)] += 1

    top5_idx = np.argsort(probs)[::-1]
    top5 = [{'label': le.classes_[i], 'prob': round(float(probs[i]), 4)} for i in top5_idx[:5]]

    results.append({
        'file': Path(filepath).name,
        'cn_label': cn_label,
        'expected': gt_6,
        'pred': pred_6,
        'correct': is_correct,
        'pred_raw': pred_raw,
        'prob_raw': round(float(raw_probs[np.argmax(raw_probs)]), 4),
        'prob_corrected': round(float(probs[np.argmax(probs)]), 4),
        'top5': top5,
    })

    mark = '✅' if is_correct else '❌'
    print(f'  {mark} {Path(filepath).name[:30]:30s}  期望={gt_6:12s}  预测={pred_6:12s}  (校正前={pred_raw:12s})')

# ── 输出结果 ──
print()
print('═' * 55)
print('结果汇总')
print('═' * 55)
print(f'  总条数:   {total}')
print(f'  正确:     {correct_6} ({correct_6/total*100:.1f}%)')
print()

# 混淆矩阵
print('═' * 55)
print('混淆矩阵 (6类)')
print('═' * 55)
header = '真实\\预测  ' + ''.join(f'{c:12s}' for c in EMOTION_6)
print(header)
for true in EMOTION_6:
    row = f'{true:10s} '
    for pred in EMOTION_6:
        row += f'{confusion[(true, pred)]:5d}       '
    print(row)
print()

# 每类准确率
print('═' * 55)
print('每类准确率')
print('═' * 55)
for cls in EMOTION_6:
    c = per_class[cls]['correct']
    t = per_class[cls]['total']
    if t > 0:
        print(f'  {cls:15s} {c}/{t} = {c/t*100:.1f}%')
    else:
        print(f'  {cls:15s} (无数据)')

# 校正前后对比
print()
print('═' * 55)
print('先验校正效果（整体准确率对比）')
print('═' * 55)
correct_raw = sum(1 for r in results if r['pred_raw'] == r['expected'])
print(f'  校正前: {correct_raw}/{total} = {correct_raw/total*100:.1f}%')
print(f'  校正后: {correct_6}/{total} = {correct_6/total*100:.1f}%')

# 保存结果
output_path = BASE / '模型代码/结果_6类推理_40猫.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump({
        'accuracy_6class': round(correct_6 / total, 4) if total > 0 else 0,
        'total': total,
        'correct_6': correct_6,
        'per_category': {
            cls: {
                'correct': per_class[cls]['correct'],
                'total': per_class[cls]['total'],
                'accuracy': round(per_class[cls]['correct'] / per_class[cls]['total'], 4) if per_class[cls]['total'] > 0 else 0
            } for cls in EMOTION_6
        },
        'confusion_6': {
            true: {pred: confusion[(true, pred)] for pred in EMOTION_6}
            for true in EMOTION_6
        },
        'results': results,
    }, f, ensure_ascii=False, indent=2)
print(f'\n结果已保存: {output_path}')

"""
10类 freeze 模型 → 新猫语分类 40条音频推理
每文件逐条推理（不按ID去重），显示10类概率分布
# 10类冻洁模型 → 新猫语 40条音频推理（按类别统计）
"""

import os, sys, json, re, warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import joblib
import torch
import librosa
from transformers import HubertModel, Wav2Vec2FeatureExtractor

# ── 路径 ──
BASE = Path(r'/path/to/dataset')
DATA_DIR = BASE / '新猫语分类/新情绪分类'
MODEL_DIR = BASE / '模型代码/训练结果/hubert_v2/freeze_model_hubert_20260618_111717'
OUTPUT = BASE / '模型代码/结果_10类推理_新猫语.json'

SR = 16000
CROP_SAMPLES = int(SR * 5.0)  # 5秒
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 10类 Pandeya 全标签（含 HuntingMind 全称）
PANDEYA_LABELS = [
    'Angry', 'Defence', 'Fighting', 'Happy', 'HuntingMind',
    'Mating', 'MotherCall', 'Paining', 'Resting', 'Warning'
]

# 用户中文标签 → 最接近的 Pandeya 类
CN_TO_PANDEYA = {
    '友好': 'Happy',
    '吸引注意': 'Mating',
    '愤怒示威': 'Angry',
    '母体呼唤': 'MotherCall',
    '渴求': 'Mating',
    '疼痛惊吓': 'Paining',
    '警告驱逐': 'Warning',
}

# 产品6类合并映射
MERGE_6 = {
    'Angry':'Aggressive', 'Defence':'Aggressive',
    'Fighting':'Aggressive', 'Warning':'Aggressive',
    'Happy':'Happy', 'HuntingMind':'Happy',
    'Mating':'Mating', 'MotherCall':'MotherCall',
    'Paining':'Paining', 'Resting':'Resting',
}

print(f'设备: {DEVICE}')

# ── 加载模型 ──
t0 = __import__('time').time()
data = joblib.load(str(MODEL_DIR / 'svm.pkl'))
svm = data['svm']
scaler = data['scaler']
le = data['label_encoder']
label_names = list(le.classes_)
print(f'  SVM {len(label_names)}类: {label_names}')

model = HubertModel.from_pretrained('facebook/hubert-base-ls960', local_files_only=True)
model.to(DEVICE)
model.eval()
for p in model.parameters():
    p.requires_grad = False
processor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/hubert-base-ls960', local_files_only=True)
print(f'  加载耗时: {__import__("time").time()-t0:.1f}s')

# ── 收集所有文件 ──
all_files = []
for subdir in sorted(DATA_DIR.iterdir()):
    if not subdir.is_dir():
        continue
    cn_label = subdir.name
    for f in sorted(subdir.glob('*')):
        all_files.append((f, cn_label))

print(f'\n共 {len(all_files)} 条音频')

# ── 每文件推理 ──
results = []
correct_6 = 0
total = 0
pred_counts_6 = Counter()
correct_by_cn = defaultdict(lambda: {'correct': 0, 'total': 0})

for idx, (path, cn_label) in enumerate(all_files):
    expected_pandeya = CN_TO_PANDEYA.get(cn_label, '?')
    expected_6 = MERGE_6.get(expected_pandeya, expected_pandeya)

    # 加载音频
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    peak = np.max(np.abs(y))
    if peak > 1e-10:
        y = y / peak
    # 居中裁剪/填充到 5秒
    if len(y) > CROP_SAMPLES:
        start = (len(y) - CROP_SAMPLES) // 2
        y = y[start:start + CROP_SAMPLES]
    else:
        pad = CROP_SAMPLES - len(y)
        y = np.pad(y, (pad // 2, pad - pad // 2))

    # HuBERT 嵌入
    with torch.no_grad():
        inputs = processor([y], sampling_rate=SR, return_tensors='pt', padding=True).to(DEVICE)
        outputs = model(inputs.input_values, output_hidden_states=True)
        hs = outputs.hidden_states
        emb = torch.stack([hs[i] for i in [3, 4, 5]], dim=0).mean(dim=0).mean(dim=1).cpu().numpy()

    # SVM 预测
    emb_scaled = scaler.transform(emb)
    scores = svm.decision_function(emb_scaled)[0]
    scores = scores - scores.max()
    probs = np.exp(scores) / np.exp(scores).sum()
    top_idx = np.argsort(probs)[::-1]

    top_label = label_names[top_idx[0]]
    top_6 = MERGE_6.get(top_label, top_label)

    # 统计6类正确性
    total += 1
    if top_6 == expected_6:
        correct_6 += 1
    correct_by_cn[cn_label]['total'] += 1
    if top_6 == expected_6:
        correct_by_cn[cn_label]['correct'] += 1
    pred_counts_6[top_6] += 1

    # 显示
    mark = '✅' if top_6 == expected_6 else '❌'
    print(f'\n({idx+1:2d}/{len(all_files)}) {mark} {path.name[:40]}...')
    print(f'    标签: [{cn_label}] → 期望: {expected_6}')
    print(f'    模型TOP5:')
    for i in top_idx[:5]:
        lbl = label_names[i]
        p6 = MERGE_6.get(lbl, lbl)
        bar = '█' * int(probs[i] * 25)
        print(f'      {lbl:<12} → {p6:<12} {probs[i]*100:5.1f}% {bar}')

    results.append({
        'file': path.name,
        'cn_label': cn_label,
        'expected': expected_6,
        'top5': [{'label': label_names[i], 'merged': MERGE_6.get(label_names[i], label_names[i]),
                  'prob': round(float(probs[i]), 4)} for i in top_idx[:5]],
    })

# ── 汇总 ──
print(f'\n{"═"*60}')
print(f' 总: {total} 条')
print(f' 6类正确: {correct_6}/{total} ({correct_6/total*100:.1f}%)')
print(f'{"═"*60}\n')

print('各类正确率 (6类):')
for cn_label in sorted(correct_by_cn.keys()):
    info = correct_by_cn[cn_label]
    rate = info['correct'] / info['total'] * 100
    print(f'  {cn_label:8s}: {info["correct"]}/{info["total"]} ({rate:.0f}%)')

print(f'\n预测分布 (6类):')
for label, cnt in pred_counts_6.most_common():
    print(f'  {label:<12}: {cnt} ({cnt/total*100:.0f}%)')

# 混淆矩阵 (6类)
classes_6 = sorted(set(MERGE_6.values()))
conf_6 = {e: {p: 0 for p in classes_6} for e in classes_6}
for r in results:
    e = r['expected']
    p = r['top5'][0]['merged']
    conf_6[e][p] += 1

print(f'\n混淆矩阵 (6类, 行=期望, 列=预测):')
header = '        ' + ' '.join(f'{c:>10}' for c in classes_6)
print(header)
for e in classes_6:
    row = f'{e:>8}:'
    for p in classes_6:
        row += f'{conf_6[e][p]:>10}'
    print(row)

with open(OUTPUT, 'w', encoding='utf-8') as f:
    json.dump({
        'accuracy_6class': round(correct_6 / total, 4),
        'total': total,
        'correct_6': correct_6,
        'per_category': {k: {'correct': v['correct'], 'total': v['total'],
                             'accuracy': round(v['correct']/v['total'], 4) if v['total'] > 0 else 0}
                         for k, v in correct_by_cn.items()},
        'confusion_6': conf_6,
        'results': results,
    }, f, ensure_ascii=False, indent=2)

print(f'\n完整结果 → {OUTPUT}')

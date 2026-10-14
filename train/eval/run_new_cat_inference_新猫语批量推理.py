"""
新猫语分类 40 条 — 推理分析
用 HuBERT freeze + SVM (10类) 看看模型在真实录音上到底怎么分
# 新猫语测试集批量推理（含6类合并评估）
"""

import os
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from pathlib import Path

import joblib
import librosa
import numpy as np
import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor

SR = 16000
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE = Path(r'/path/to/dataset')
MODEL_DIR = BASE / '模型代码/训练结果/hubert_v2/freeze_model_hubert_20260618_111717'
DATA_DIR = BASE / '新猫语分类/新情绪分类'
LAYERS = [3, 4, 5]

CN_LABELS = [
    'Angry', 'Defence', 'Fighting', 'Happy', 'HuntingMind',
    'Mating', 'MotherCall', 'Paining', 'Resting', 'Warning'
]

CN_MAP = {
    '友好': 'Happy', '吸引注意': 'Mating',
    '愤怒示威': 'Angry', '渴求': 'Hungry',
    '警告驱逐': 'Warning', '母体呼唤': 'MotherCall',
    '疼痛惊吓': 'Paining',
}

MERGE_6 = {
    'Angry':'Aggressive', 'Defence':'Aggressive', 'Fighting':'Aggressive',
    'Warning':'Aggressive', 'Happy':'Happy', 'HuntingMind':'Happy',
    'Mating':'Mating', 'MotherCall':'MotherCall',
    'Paining':'Paining', 'Resting':'Resting',
}

print('═' * 60)
print('  新猫语分类 推理分析')
print('═' * 60)

# 加载模型
print('\n[1/3] 加载 HuBERT...')
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    'facebook/hubert-base-ls960', local_files_only=True)
hubert = HubertModel.from_pretrained(
    'facebook/hubert-base-ls960', local_files_only=True).to(DEVICE)
hubert.eval()
print(f'  设备: {DEVICE}, 参数量: {sum(p.numel() for p in hubert.parameters()):,}')

print('\n[2/3] 加载 SVM...')
saved = joblib.load(str(MODEL_DIR / 'svm.pkl'))
svm = saved['svm']
scaler = saved['scaler']
le = saved['label_encoder']
print(f'  SVM 类型: {type(svm).__name__}')
print(f'  标签: {le.classes_}')

# 收集文件
print('\n[3/3] 加载音频...')
all_files = []  # (path, true_label)
for cn_dir in sorted(DATA_DIR.iterdir()):
    if not cn_dir.is_dir(): continue
    en_map = CN_MAP.get(cn_dir.name)
    for fp in sorted(cn_dir.glob('*.wav')) + sorted(cn_dir.glob('*.mp3')):
        all_files.append((str(fp), cn_dir.name, en_map or 'Unknown'))

print(f'  共 {len(all_files)} 条音频')

# 推理
print('\n' + '=' * 60)
print('  推理结果')
print('=' * 60)

results_10 = []
results_6 = []
total = 0
correct_10 = 0
correct_6 = 0
per_class = {}

with torch.no_grad():
    for i in range(0, len(all_files), BATCH_SIZE):
        batch = all_files[i:i+BATCH_SIZE]
        waveforms = []
        for fp, _, _ in batch:
            y, sr = librosa.load(fp, sr=SR, mono=True)
            waveforms.append(y)
        
        inputs = feature_extractor(
            waveforms, sampling_rate=SR,
            return_tensors='pt', padding=True,
            truncation=True, max_length=SR * 5  # 5秒上限
        ).to(DEVICE)
        
        outputs = hubert(inputs.input_values, output_hidden_states=True)
        hs = outputs.hidden_states
        # layers [3,4,5] mean → time mean → 768-dim
        emb = torch.stack([hs[l] for l in LAYERS], dim=0).mean(dim=0).mean(dim=1).cpu().numpy()
        
        emb_scaled = scaler.transform(emb)
        pred_indices = svm.predict(emb_scaled)
        pred_labels = le.inverse_transform(pred_indices)
        scores = svm.decision_function(emb_scaled) if hasattr(svm, 'decision_function') else None
        
        for j, pred_label in enumerate(pred_labels):
            fp, true_cn, true_en = batch[j]
            pred_6 = MERGE_6[pred_label]
            true_6 = MERGE_6.get(true_en, true_en)
            
            total += 1
            if pred_label == true_en:
                correct_10 += 1
            if pred_6 == true_6:
                correct_6 += 1
            
            # 详细输出
            fname = os.path.basename(fp)[:40]
            check_10 = '✓' if pred_label == true_en else '✗'
            check_6 = '✓' if pred_6 == true_6 else ' '
            
            # Top-3 by SVM decision values (one-vs-rest scores)
            if scores is not None:
                top3 = np.argsort(-scores[j])[:3]
                top3_str = ', '.join(f'{le.inverse_transform([k])[0]}({scores[j][k]:.2f})' for k in top3)
            else:
                top3_str = '-'
            
            rev = f'{check_10} [{true_cn:<6s}→{pred_label:<12s}] {check_6} {fname:<42s} | top3: {top3_str}'
            print(rev)
            
            if true_en not in per_class:
                per_class[true_en] = {'total': 0, 'corr': 0, 'preds': {}}
            per_class[true_en]['total'] += 1
            if pred_label == true_en:
                per_class[true_en]['corr'] += 1
            if pred_label not in per_class[true_en]['preds']:
                per_class[true_en]['preds'][pred_label] = 0
            per_class[true_en]['preds'][pred_label] += 1

# 汇总
print(f'\n{"="*60}')
print('  结果汇总')
print(f'{"="*60}')
print(f'  10类准确率: {correct_10}/{total} = {correct_10/total:.1%}')
print(f'  6类合并准确率: {correct_6}/{total} = {correct_6/total:.1%}')

print('\n  ─── 各类详情 ───')
print(f'  {"真实标签":<16s} {"总数":>4s} {"正确":>4s} {"准确率":>7s} {"预测分布":<40s}')
print(f'  {"─"*75}')
for label, info in sorted(per_class.items()):
    acc = info['corr'] / info['total']
    pred_dist = ', '.join(
        f'{p}({c})' for p, c in sorted(info['preds'].items(), key=lambda x: -x[1])[:5]
    )
    print(f'  {label:<16s} {info["total"]:>4d} {info["corr"]:>4d} {acc:>6.1%}  {pred_dist:<40s}')

print(f'\n{"="*60}')
print('  结束')
print(f'{"="*60}')

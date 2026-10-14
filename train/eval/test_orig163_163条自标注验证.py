# 163条自标注数据 -> 模型准确性验证
"""6类模型在原猫语分类163条上的验证"""
import os
import sys

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

import time
from collections import Counter
from pathlib import Path

import joblib
import librosa
import numpy as np
import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor

SR = 16000; DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYERS = [3, 4, 5]

BASE = Path('C:/Users/chen/Desktop')
MODEL_DIR = BASE / '模型代码/训练结果/hubert_v2/freeze_model_hubert_20260618_152123'

# 加载模型
print('[加载] 6类模型...')
t0 = time.time()
data = joblib.load(str(MODEL_DIR / 'svm.pkl'))
svm, scaler, le = data['svm'], data['scaler'], data['label_encoder']
print(f'  类别: {list(le.classes_)}')

model = HubertModel.from_pretrained('facebook/hubert-base-ls960', local_files_only=True)
model.to(DEVICE); model.eval()
for p in model.parameters(): p.requires_grad = False
processor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/hubert-base-ls960', local_files_only=True)
print(f'  加载完成 ({time.time()-t0:.0f}s)')

def predict(audio_path):
    y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    peak = np.max(np.abs(y))
    y = y / peak if peak > 1e-10 else np.zeros_like(y)
    with torch.no_grad():
        inputs = processor([y], sampling_rate=SR, return_tensors="pt", padding=True).input_values.to(DEVICE)
        outputs = model(inputs, output_hidden_states=True)
        hs = outputs.hidden_states
        emb = torch.stack([hs[i] for i in LAYERS], dim=0).mean(dim=0).mean(dim=1).cpu().numpy()
    emb_s = scaler.transform(emb)
    scores = svm.decision_function(emb_s)[0]
    scores -= scores.max()
    probs = np.exp(scores) / np.exp(scores).sum()
    top_idx = np.argmax(probs)
    return le.classes_[top_idx], probs[top_idx]

# 中文→Pandeya→6类映射
CN_MAP = {
    '交配（低沉嘶吼）': 'Mating',
    '休息': 'Resting',
    '快乐': 'Happy',
    '愤怒（呜哇哇哇）': 'Angry',
    '打架（尖锐刺耳）': 'Fighting',
    '母体呼叫（警笛）': 'MotherCall',
    '狩猎（短促的嗷呜）': 'HuntingMind',
    '疼痛': 'Paining',
    '警告（呜~）（咕噜）': 'Warning',
    '防御(响尾蛇嘶哈)': 'Defence',
}
MERGE_6 = {
    'Angry':'Aggressive', 'Defence':'Aggressive', 'Fighting':'Aggressive', 'Warning':'Aggressive',
    'Happy':'Happy', 'HuntingMind':'Happy',
    'Mating':'Mating', 'MotherCall':'MotherCall',
    'Paining':'Paining', 'Resting':'Resting',
}

DATA_DIR = BASE / '原猫语分类/情绪分类'
print('\n═══ 原猫语分类 163条验证 ═══\n')

total_correct = 0
total_samples = 0

for cn_dir, en_label in CN_MAP.items():
    merged_expected = MERGE_6[en_label]
    files = sorted((DATA_DIR / cn_dir).glob('*.wav'))
    if not files:
        print(f'  {cn_dir:<16} → 无文件')
        continue
    
    # 跑本目录所有音频
    correct = 0
    preds = Counter()
    for fp in files:
        pred, prob = predict(fp)
        preds[pred] += 1
        if pred == merged_expected:
            correct += 1
    
    total_correct += correct
    total_samples += len(files)
    acc = correct / len(files)
    
    # 显示：真实→预测分布
    pred_detail = ', '.join([f'{k}={v}' for k, v in preds.most_common()])
    emoji = '✅' if acc >= 0.5 else '❌'
    print(f'  {emoji} {merged_expected:<12} ({en_label:<12}) {correct}/{len(files)}={acc:.0%}  {pred_detail}')

print('\n  ─────────────────────────')
print(f'  总正确率: {total_correct}/{total_samples} = {total_correct/total_samples:.1%}')

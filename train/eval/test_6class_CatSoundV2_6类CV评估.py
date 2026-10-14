# 6类情绪穹顶 → CatSound_V2 交叉验证评估
"""新6类模型验证 - 跑新猫语分类40条 + CatSound_V2各类验证"""
import os, sys
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from pathlib import Path
import numpy as np
import torch
import librosa
import joblib
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import time

SR = 16000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYERS = [3, 4, 5]

BASE = Path('C:/Users/陈项目负责人/Desktop')
MODEL_DIR = BASE / '模型代码/训练结果/hubert_v2/freeze_model_hubert_20260618_152123'

# 加载
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

# 1. CatSound_V2 各类验证
print(f'\n═══ CatSound_V2 各类验证 ═══')
# 合并前的原始类 → 取第一条文件
ORIG_TO_TEST = {
    'Aggressive': ['Angry', 'Defence', 'Fighting', 'Warning'],
    'Happy': ['Happy', 'HuntingMind'],
    'Mating': ['Mating'],
    'MotherCall': ['MotherCall'],
    'Paining': ['Paining'],
    'Resting': ['Resting'],
}

DATA_DIR = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
correct = 0
for merged_cls, orig_list in ORIG_TO_TEST.items():
    test = []
    for orig in orig_list:
        files = sorted((DATA_DIR / orig).glob('*.mp3'))
        test += [f for f in files if '_aug1' not in f.name]
    if not test:
        print(f'  {merged_cls}: 无文件')
        continue
    pred, prob = predict(test[0])
    ok = '✅' if pred == merged_cls else '❌'
    if ok == '✅': correct += 1
    print(f'  {merged_cls:<12} → {pred:<12} {prob*100:5.1f}%  {ok}')
print(f'  各类正确率: {correct}/{len(le.classes_)}')

# 2. 新猫语分类40条
print(f'\n═══ 新猫语分类40条 ═══')
NEW_DIR = BASE / '新猫语分类/新情绪分类'
from collections import Counter
all_preds = Counter()
cat_files = sorted(NEW_DIR.rglob('*.wav'))
print(f'  扫描到 {len(cat_files)} 条')
for i, fp in enumerate(cat_files):
    try:
        pred, prob = predict(fp)
        all_preds[pred] += 1
    except Exception as e:
        print(f'  [!] {fp.name}: {e}')
    if (i+1) % 20 == 0:
        print(f'  推理: {i+1}/{len(cat_files)}')

print(f'\n── 分布 ──')
for label, count in all_preds.most_common():
    pct = count / len(cat_files) * 100
    print(f'  {label:<12} {count} 条 ({pct:.1f}%)')

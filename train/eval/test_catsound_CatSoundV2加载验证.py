"""
CatSound_V2 各类别快速验证 — 每个类取一条非增强音频测推理
# CatSound_V2 基础加载和标签验证脚本
"""
import os
import sys

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

import time
from pathlib import Path

import joblib
import librosa
import numpy as np
import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor

SR = 16000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYERS = [3, 4, 5]

BASE = Path('C:/Users/chen/Desktop')
MODEL_DIR = BASE / '模型代码/训练结果/hubert_v2/freeze_model_hubert_20260618_111717'
DATA_DIR = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X'

# 加载模型
print('[加载] HuBERT + SVM...')
t0 = time.time()
data = joblib.load(str(MODEL_DIR / 'svm.pkl'))
svm, scaler, le = data['svm'], data['scaler'], data['label_encoder']
print(f'  SVM: {len(le.classes_)} 类')

model = HubertModel.from_pretrained('facebook/hubert-base-ls960', local_files_only=True)
model.to(DEVICE); model.eval()
for p in model.parameters(): p.requires_grad = False
processor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/hubert-base-ls960', local_files_only=True)
print(f'  HuBERT 加载完成 ({time.time()-t0:.0f}s)')

def preprocess(y):
    peak = np.max(np.abs(y))
    return y / peak if peak > 1e-10 else np.zeros_like(y)

def predict(audio_path):
    y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    y = preprocess(y)
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

# 遍历10类，取第一条非增强音频
print(f'\n{"真实标签":<12} → {"模型预测":<12}  置信度   {"结果":<6}')
print('─' * 50)
correct = 0
for cls in ['Angry','Defence','Fighting','Happy','HuntingMind','Mating','MotherCall','Paining','Resting','Warning']:
    files = sorted((DATA_DIR / cls).glob('*.mp3'))
    test = [f for f in files if '_aug1' not in f.name]
    if not test:
        print(f'{cls:<12} →  [无文件]')
        continue
    pred, prob = predict(test[0])
    ok = '✅' if pred == cls else '❌'
    if ok == '✅': correct += 1
    print(f'{cls:<12} → {pred:<12}  {prob*100:5.1f}%  {ok}')

print(f'\n正确率: {correct}/10')

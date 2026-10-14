"""
对比: 10类 freeze 模型 vs Pandeya model_best 官方模型
在同一批 40 条音频上各自推理，看结果是否一致
# 模型交叉对比：HuBERT+SVM vs Pandeya model_best 官方权重
"""

import os, sys, json, re, warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from pathlib import Path
from collections import Counter
import numpy as np
import joblib
import torch
import librosa
from transformers import HubertModel, Wav2Vec2FeatureExtractor

# ── 路径 ──
BASE = Path(r'/path/to/dataset')
DATA_DIR = BASE / '新猫语分类/新情绪分类'
HUBERT_DIR = BASE / '模型代码/训练结果/hubert_v2/freeze_model_hubert_20260618_111717'
OUTPUT = BASE / '模型代码/结果_模型交叉对比_40条.json'

SR = 16000
CROP_SAMPLES = int(SR * 5.0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PANDEYA_LABELS = [
    'Angry', 'Defence', 'Fighting', 'Happy', 'HuntingMind',
    'Mating', 'MotherCall', 'Paining', 'Resting', 'Warning'
]

MERGE_6 = {
    'Angry':'Aggressive', 'Defence':'Aggressive',
    'Fighting':'Aggressive', 'Warning':'Aggressive',
    'Happy':'Happy', 'HuntingMind':'Happy',
    'Mating':'Mating', 'MotherCall':'MotherCall',
    'Paining':'Paining', 'Resting':'Resting',
}

print(f'设备: {DEVICE}')
print('═'*50)
print('模型对比: HuBERT+SVM(freeze 10类) vs Pandeya model_best')
print('═'*50)

# ═══════════════════════════════════
# 模型 1: HuBERT freeze SVM
# ═══════════════════════════════════
t0 = __import__('time').time()
print('\n[1/2] 加载 HuBERT+SVM...')
data = joblib.load(str(HUBERT_DIR / 'svm.pkl'))
svm = data['svm']
scaler = data['scaler']
le = data['label_encoder']
label_names = list(le.classes_)
print(f'  SVM {len(label_names)}类')

model = HubertModel.from_pretrained('facebook/hubert-base-ls960', local_files_only=True)
model.to(DEVICE)
model.eval()
for p in model.parameters():
    p.requires_grad = False
processor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/hubert-base-ls960', local_files_only=True)
print(f'  加载: {__import__("time").time()-t0:.1f}s')

# ═══════════════════════════════════
# 模型 2: Pandeya model_best
# ═══════════════════════════════════
print('\n[2/2] 加载 Pandeya model_best...')
t1 = __import__('time').time()
import tensorflow as tf
from tensorflow.keras import layers, models

def build_pandeya_model(input_shape=(None, 96, 1), num_classes=10):
    """重建 Pandeya CNN + Kapre melgram"""
    try:
        import kapre
        from kapre.composed import get_melspectrogram_layer
        inp = layers.Input(shape=(CROP_SAMPLES,))
        x = get_melspectrogram_layer(
            n_fft=1024, hop_length=512, n_mels=96,
            sr=SR, return_decibel=True, padding='same'
        )(inp)
        x = layers.Normalization(axis=-1)(x)
        # (time, freq, 1)
        x = layers.Reshape((-1, 96, 1))(x)
    except ImportError:
        print('  [WARN] kapre 未安装, fallback: 外部mel输入')
        inp = layers.Input(shape=input_shape)
        x = inp
    
    # Conv2D×5 (同 model_best.hdf5)
    x = layers.Conv2D(32, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2,4))(x)
    
    x = layers.Conv2D(32, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((3,4))(x)
    
    x = layers.Conv2D(32, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2,5))(x)
    
    x = layers.Conv2D(32, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2,4))(x)
    
    x = layers.Conv2D(32, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((4,4))(x)
    
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dense(50)(x)
    out = layers.Dense(num_classes, activation='sigmoid')(x)
    
    return models.Model(inputs=inp, outputs=out)

MODEL_PATH = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X/model_best.hdf5'
pandeya_model = None
try:
    pandeya_model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    print(f'  ✅ 直接加载 hdf5 成功 ({__import__("time").time()-t1:.1f}s)')
except Exception as e:
    print(f'  ⚠️ 直接加载失败: {e}')
    print(f'  → 尝试重建架构加载权重')
    try:
        pandeya_model = build_pandeya_model()
        pandeya_model.load_weights(str(MODEL_PATH))
        print(f'  ✅ 架构重建+加载权重成功 ({__import__("time").time()-t1:.1f}s)')
    except Exception as e2:
        print(f'  ❌ 加载失败: {e2}')

# ═══════════════════════════════════
# 收集所有文件
# ═══════════════════════════════════
all_files = []
for subdir in sorted(DATA_DIR.iterdir()):
    if not subdir.is_dir():
        continue
    for f in sorted(subdir.glob('*')):
        all_files.append((f, subdir.name))

print(f'\n共 {len(all_files)} 条音频，开始推理...')

# ═══════════════════════════════════
# 逐条推理
# ═══════════════════════════════════
results = []
agree = 0

for idx, (path, cn_label) in enumerate(all_files):
    # 加载音频
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    peak = np.max(np.abs(y))
    if peak > 1e-10:
        y = y / peak
    if len(y) > CROP_SAMPLES:
        start = (len(y) - CROP_SAMPLES) // 2
        y = y[start:start + CROP_SAMPLES]
    else:
        pad = CROP_SAMPLES - len(y)
        y = np.pad(y, (pad // 2, pad - pad // 2))

    # ── 模型1: HuBERT ──
    with torch.no_grad():
        inputs = processor([y], sampling_rate=SR, return_tensors='pt', padding=True).to(DEVICE)
        outputs = model(inputs.input_values, output_hidden_states=True)
        hs = outputs.hidden_states
        emb = torch.stack([hs[i] for i in [3, 4, 5]], dim=0).mean(dim=0).mean(dim=1).cpu().numpy()
    emb_scaled = scaler.transform(emb)
    h_scores = svm.decision_function(emb_scaled)[0]
    h_scores = h_scores - h_scores.max()
    h_probs = np.exp(h_scores) / np.exp(h_scores).sum()
    h_top_idx = np.argsort(h_probs)[::-1]
    h_top_label = label_names[h_top_idx[0]]
    h_top_6 = MERGE_6.get(h_top_label, h_top_label)

    # ── 模型2: Pandeya ──
    p_top_label = 'N/A'
    p_top_6 = 'N/A'
    p_top_prob = 0.0
    p_top5 = []
    if pandeya_model is not None:
        # 计算 mel（如果模型用kapre则传入raw）
        try:
            # Try with kapre internal
            p_inp = np.expand_dims(y, axis=0).astype(np.float32)
            p_out = pandeya_model.predict(p_inp, verbose=0)[0]  # (10,)
        except:
            # Fallback: compute mel externally
            mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=1024, hop_length=512, n_mels=96)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            # (time, freq) -> (1, time, freq, 1)
            mel_db = np.expand_dims(mel_db.T, axis=(0, -1)).astype(np.float32)
            p_out = pandeya_model.predict(mel_db, verbose=0)[0]
        
        p_scores = p_out
        p_top_idx = np.argsort(p_scores)[::-1]
        p_top_label = PANDEYA_LABELS[p_top_idx[0]]
        p_top_6 = MERGE_6.get(p_top_label, p_top_label)
        p_top_prob = float(p_scores[p_top_idx[0]])
        for i in p_top_idx[:5]:
            p_top5.append({
                'label': PANDEYA_LABELS[i],
                'merged': MERGE_6.get(PANDEYA_LABELS[i], PANDEYA_LABELS[i]),
                'prob': round(float(p_scores[i]), 4)
            })

    # 判断两个模型是否一致（按6类TOP1）
    match = '✅' if h_top_6 == p_top_6 else '⚠️'
    if h_top_6 == p_top_6:
        agree += 1

    print(f'\n({idx+1:2d}) {path.name[:35]}...')
    print(f'    用户标签: {cn_label}')
    print(f'    HuBERT:  {h_top_label:<12s} → {h_top_6:<12s} ({h_probs[h_top_idx[0]]*100:.1f}%)')
    print(f'    Pandeya: {p_top_label:<12s} → {p_top_6:<12s} ({p_top_prob*100:.1f}%)')
    print(f'    一致? {match}')

    results.append({
        'file': path.name,
        'cn_label': cn_label,
        'hubert_top1': {'label': h_top_label, 'merged': h_top_6, 'prob': round(float(h_probs[h_top_idx[0]]), 4)},
        'hubert_top5': [{'label': label_names[i], 'merged': MERGE_6.get(label_names[i], label_names[i]),
                         'prob': round(float(h_probs[i]), 4)} for i in h_top_idx[:5]],
        'pandeya_top1': {'label': p_top_label, 'merged': p_top_6, 'prob': round(p_top_prob, 4)},
        'pandeya_top5': p_top5,
        'agree': h_top_6 == p_top_6,
    })

# ── 汇总 ──
total = len(all_files)
print(f'\n{"═"*60}')
print(f'总文件: {total}')
print(f'两模型一致: {agree}/{total} ({agree/total*100:.1f}%)')
print(f'{"═"*60}')

# 按用户类别统计一致性
from collections import defaultdict
agree_by_cn = defaultdict(lambda: {'agree': 0, 'total': 0})
for r in results:
    c = r['cn_label']
    agree_by_cn[c]['total'] += 1
    if r['agree']:
        agree_by_cn[c]['agree'] += 1

print('\n按用户标签统计两模型一致性:')
for c in sorted(agree_by_cn.keys()):
    info = agree_by_cn[c]
    rate = info['agree'] / info['total'] * 100
    print(f'  {c:8s}: {info["agree"]}/{info["total"]} ({rate:.0f}%)')

with open(OUTPUT, 'w', encoding='utf-8') as f:
    json.dump({
        'total': total,
        'agree_count': agree,
        'agree_pct': round(agree/total, 4),
        'agree_by_category': {k: {'agree': v['agree'], 'total': v['total'],
                                   'rate': round(v['agree']/v['total'], 4)}
                              for k, v in agree_by_cn.items()},
        'results': results,
    }, f, ensure_ascii=False, indent=2)

print(f'\n完整结果 → {OUTPUT}')

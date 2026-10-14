"""
测试 Pandeya model_best.hdf5 — 直接加载权重推理
对比 HuBERT+SVM 在跨数据集上的表现

结构: melgram(12kHz,96mel)→Conv2D×5→GlobalMaxPool→Dense(50)→Sigmoid(10)
# Pandeya model_best.hdf5（原始 Keras 权重）→ 推理验证
"""

import os, sys, time, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from pathlib import Path
import numpy as np
import librosa
import h5py

# ── 路径 ──
BASE = Path(r'/path/to/dataset')
MODEL_PATH = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X/model_best.hdf5'
DATA_DIR = BASE / '原猫语分类/情绪分类'
HUBERT_MODEL_DIR = BASE / '模型代码/训练结果/hubert_v2/freeze_model_hubert_20260618_111717'

# Pandeya 10类 (与 CSV 一致)
PANDEYA_LABELS = [
    "Angry", "Defence", "Fighting", "Happy", "HuntingMind",
    "Mating", "MotherCall", "Paining", "Resting", "Warning"
]

# 中文目录 → Pandeya 标签
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


# =========================================================
# Reconstruct Pandeya model (without custom melgram layer)
# =========================================================
def build_pandeya_model(input_shape=(None, 96, 1), num_classes=10):
    """
    重建 Pandeya 的 CNN 架构。
    不含自定义 Melspectrogram 层——外部计算 mel 后输入。
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models
    
    inp = layers.Input(shape=input_shape)
    x = layers.Normalization(axis=-1)(inp)
    
    # Conv block 1
    x = layers.Conv2D(32, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2,4))(x)
    
    # Conv block 2
    x = layers.Conv2D(32, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((3,4))(x)
    
    # Conv block 3
    x = layers.Conv2D(32, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2,5))(x)
    
    # Conv block 4
    x = layers.Conv2D(32, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2,4))(x)
    
    # Conv block 5
    x = layers.Conv2D(32, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((4,4))(x)
    
    # Global max pool + classifier
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dense(50, activation='sigmoid')(x)
    x = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = models.Model(inputs=inp, outputs=x)
    return model


def load_pandeya_weights(model, h5_path):
    """
    从 model_best.hdf5 加载权重。
    跳过 melgram 和 normalization 层（我们在外部做）。
    """
    f = h5py.File(h5_path, 'r')
    weights_group = f['model_weights']
    
    layer_map = {
        'conv2d_1': model.layers[2],  # 0=inp, 1=norm, 2=conv2d_1
        'batch_normalization_1': model.layers[3],
        'conv2d_2': model.layers[5],
        'batch_normalization_2': model.layers[6],
        'conv2d_3': model.layers[8],
        'batch_normalization_3': model.layers[9],
        'conv2d_4': model.layers[11],
        'batch_normalization_4': model.layers[12],
        'conv2d_5': model.layers[14],
        'batch_normalization_5': model.layers[15],
        'dense_1': model.layers[17],
        'dense_2': model.layers[18],  # final classifier
    }
    
    loaded = 0
    for layer_name in weights_group:
        if layer_name not in layer_map:
            continue
        target = layer_map[layer_name]
        w_names = weights_group[layer_name].attrs['weight_names']
        weights = []
        for wn in w_names:
            w = weights_group[layer_name][layer_name][wn.decode()][:]
            weights.append(w)
        target.set_weights(weights)
        loaded += 1
    
    f.close()
    print(f'  权重加载: {loaded} 层')
    return model


def compute_melgram(y, sr=12000):
    """
    匹配 Pandeya 的 melgram 参数:
      sr=12000, n_fft=512, hop=256, n_mels=96, fmin=0, fmax=6000
      power=2.0, decibel scale
    """
    # 重采样到 12kHz (Pandeya 的 sr)
    if sr != 12000:
        y = librosa.resample(y, orig_sr=sr, target_sr=12000)
        sr = 12000
    
    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=512, hop_length=256,
        n_mels=96, fmin=0, fmax=6000,
        power=2.0
    )  # (96, T)
    
    # dB scale (match return_decibel_melgram=True)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Normalize per sample (Normalization2D layer effect)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    
    # Add channels dim: (T, 96, 1)
    mel_db = mel_db.T[:, :, None]  # (T, 96, 1)
    
    return mel_db.astype(np.float32)


# =========================================================
# 测试: 原猫语分类 163 条
# =========================================================
def test_cross_dataset():
    print('═' * 60)
    print('  Pandeya model_best.hdf5 跨数据集测试')
    print('  数据: 原猫语分类/情绪分类')
    print('═' * 60)
    
    # 重建模型并加载权重
    print('\n[1/3] 重建 Pandeya CNN...')
    model = build_pandeya_model()
    try:
        model = load_pandeya_weights(model, MODEL_PATH)
    except Exception as e:
        print(f'  ❌ 权重加载失败: {e}')
        return
    
    print(f'\n[2/3] 推理 {DATA_DIR}...')
    total, correct_10, correct_6 = 0, 0, 0
    per_class = {}  # cls -> {total, correct_10, correct_6}
    
    t0 = time.time()
    for cn_dir in sorted(DATA_DIR.iterdir()):
        if not cn_dir.is_dir():
            continue
        cn_name = cn_dir.name
        en_label = CN_MAP.get(cn_name)
        if en_label is None:
            continue
        merged_6 = MERGE_6[en_label]
        
        files = sorted(cn_dir.glob('*.wav')) + sorted(cn_dir.glob('*.mp3'))
        if not files:
            continue
        
        if merged_6 not in per_class:
            per_class[merged_6] = {'total': 0, 'correct_10': 0, 'correct_6': 0}
        
        for fp in files:
            total += 1
            per_class[merged_6]['total'] += 1
            
            # 加载 & 预处理
            y, orig_sr = librosa.load(str(fp), sr=None, mono=True)
            mel = compute_melgram(y, orig_sr)
            
            # 推理
            inp = np.expand_dims(mel, axis=0)  # (1, T, 96, 1)
            logits = model.predict(inp, verbose=0)[0]  # (10,)
            
            # 10 类预测 (sigmoid → argmax)
            pred_idx = np.argmax(logits)
            pred_label = PANDEYA_LABELS[pred_idx]
            pred_6 = MERGE_6[pred_label]
            
            # 10 类准确率
            if pred_label == en_label:
                correct_10 += 1
                per_class[merged_6]['correct_10'] += 1
            
            # 6 类准确率
            if pred_6 == merged_6:
                correct_6 += 1
                per_class[merged_6]['correct_6'] += 1
    
    elapsed = time.time() - t0
    
    print(f'\n[3/3] 结果')
    print(f'  时间: {elapsed:.1f}s ({elapsed/total:.2f}s/条)')
    print(f'\n  ── 各类准确率 ──')
    print(f'  {"类别":<14} {"总数":>5} {"10类准确":>10} {"6类合并":>10}')
    print(f'  {"─"*41}')
    for cls in sorted(per_class.keys()):
        d = per_class[cls]
        a10 = d['correct_10'] / d['total']
        a6 = d['correct_6'] / d['total']
        print(f'  {cls:<14} {d["total"]:>5}  {d["correct_10"]:>3}/{d["total"]}={a10:>5.0%}  {d["correct_6"]:>3}/{d["total"]}={a6:>5.0%}')
    
    print(f'  {"─"*41}')
    print(f'  总计: {total} 条')
    print(f'  10 类准确率: {correct_10}/{total} = {correct_10/total:.1%}')
    print(f'  6 类合并:     {correct_6}/{total} = {correct_6/total:.1%}')
    
    return {'10_class': correct_10/total, '6_class': correct_6/total, 'total': total}


# =========================================================
# 快速对比 HuBERT+SVM
# =========================================================
def compare_with_hubert():
    """加载现有 HuBERT+SVM 模型，跑同样数据对比"""
    import torch
    import joblib
    from transformers import HubertModel, Wav2Vec2FeatureExtractor
    
    print(f'\n{"="*60}')
    print(f'  HuBERT+SVM 对比')
    print(f'{"="*60}')
    
    # 加载
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    svm_data = joblib.load(str(HUBERT_MODEL_DIR / 'svm.pkl'))
    svm, scaler, le = svm_data['svm'], svm_data['scaler'], svm_data['label_encoder']
    
    hubert = HubertModel.from_pretrained('facebook/hubert-base-ls960', local_files_only=True)
    hubert.to(device); hubert.eval()
    for p in hubert.parameters(): p.requires_grad = False
    processor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/hubert-base-ls960', local_files_only=True)
    
    total, correct_10, correct_6 = 0, 0, 0
    layers = [3, 4, 5]
    
    t0 = time.time()
    for cn_dir in sorted(DATA_DIR.iterdir()):
        if not cn_dir.is_dir():
            continue
        en_label = CN_MAP.get(cn_dir.name)
        if en_label is None:
            continue
        merged_6 = MERGE_6[en_label]
        
        files = sorted(cn_dir.glob('*.wav')) + sorted(cn_dir.glob('*.mp3'))
        for fp in files:
            total += 1
            y, _ = librosa.load(str(fp), sr=16000, mono=True)
            peak = np.max(np.abs(y))
            y = y / peak if peak > 1e-10 else np.zeros_like(y)
            
            with torch.no_grad():
                inputs = processor([y], sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(device)
                outputs = hubert(inputs, output_hidden_states=True)
                hs = outputs.hidden_states
                emb = torch.stack([hs[i] for i in layers], dim=0).mean(dim=0).mean(dim=1).cpu().numpy()
            
            emb_s = scaler.transform(emb)
            scores = svm.decision_function(emb_s)[0]
            scores -= scores.max()
            probs = np.exp(scores) / np.exp(scores).sum()
            top_idx = np.argmax(probs)
            pred_label = le.classes_[top_idx]
            pred_6 = MERGE_6.get(pred_label, pred_label)
            
            if pred_label == en_label:
                correct_10 += 1
            if pred_6 == merged_6:
                correct_6 += 1
    
    elapsed = time.time() - t0
    print(f'  时间: {elapsed:.1f}s')
    print(f'  10 类准确率: {correct_10}/{total} = {correct_10/total:.1%}')
    print(f'  6 类合并:     {correct_6}/{total} = {correct_6/total:.1%}')


if __name__ == '__main__':
    test_cross_dataset()
    compare_with_hubert()

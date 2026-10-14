"""
CNN 新猫语 40条推理 (快速验证)
用 fold3_best.pth (F1=68.9%) 跑跨数据集泛化
"""
import os, sys, warnings, time
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

SR = 16000
N_MELS = 128
N_FFT = 512
HOP_LENGTH = 256
MAX_T = 300
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 路径
BASE = Path('/mnt/c/Users/陈项目负责人/Desktop')
DATA_DIR = BASE / '新猫语分类/新情绪分类'
MODEL_PATH = BASE / '模型代码/训练结果/fdap_cnn_v1/mel_cache/fold3_best.pth'

# 中文目录名 → Pandeya 10类
CN_TO_LABEL = {
    '友好': 'Happy',
    '吸引注意': 'Mating',
    '愤怒示威': 'Angry',
    '母体呼唤': 'MotherCall',
    '渴求': 'Mating',
    '疼痛惊吓': 'Paining',
    '警告驱逐': 'Warning',
}
LABELS = [
    "Angry", "Defence", "Fighting", "Happy", "HuntingMind",
    "Mating", "MotherCall", "Paining", "Resting", "Warning"
]

class FDAPMobileCatCNN(nn.Module):
    """必须与训练时定义一致"""
    def __init__(self, num_classes=10, fdap_bands=4):
        super().__init__()
        self.fdap_bands = fdap_bands
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16),
            nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d((1, 2)))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d((1, 2)))
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(), nn.Identity())
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * fdap_bands, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        b, c, f, t = x.shape
        bs = f // self.fdap_bands
        fdap = []
        for i in range(self.fdap_bands):
            end = (i + 1) * bs if i < self.fdap_bands - 1 else f
            fdap.append(x[:, :, i * bs:end, :].mean(dim=[2, 3]))
        x = torch.cat(fdap, dim=1)
        return self.classifier(x)

def audio_to_mel(path):
    """单条音频 → Mel 频谱 [1, 128, 300]"""
    y, _ = librosa.load(path, sr=SR, mono=True)
    peak = np.max(np.abs(y))
    y = y / peak if peak > 1e-10 else np.zeros_like(y)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    if mel_norm.shape[1] < MAX_T:
        mel_norm = np.pad(mel_norm, ((0,0), (0, MAX_T - mel_norm.shape[1])))
    else:
        mel_norm = mel_norm[:, :MAX_T]
    return torch.FloatTensor(mel_norm).unsqueeze(0).unsqueeze(0)  # [1, 1, 128, 300]

def main():
    print(f'设备: {DEVICE}')
    if DEVICE == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    
    # 加载模型
    print(f'\n加载模型: {MODEL_PATH}')
    model = FDAPMobileCatCNN(num_classes=10).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f'  参数量: {sum(p.numel() for p in model.parameters()):,}')
    
    # 6类合并映射
    MERGE_6 = {
        'Aggressive': {'Angry', 'Defence', 'Fighting', 'Warning'},
        'Happy': {'Happy', 'HuntingMind'},
        'Mating': {'Mating'},
        'MotherCall': {'MotherCall'},
        'Paining': {'Paining'},
        'Resting': {'Resting'},
    }
    # 中文目录 → 6类
    CN_TO_6 = {
        '友好': 'Happy',
        '吸引注意': 'Mating',
        '愤怒示威': 'Aggressive',
        '母体呼唤': 'MotherCall',
        '渴求': 'Mating',
        '疼痛惊吓': 'Paining',
        '警告驱逐': 'Aggressive',
    }
    # 10类 → 6类
    LABEL_TO_6 = {}
    for six, tens in MERGE_6.items():
        for t in tens:
            LABEL_TO_6[t] = six
    
    # 扫描音频
    audio_files = []
    for cn_dir in sorted(DATA_DIR.iterdir()):
        if not cn_dir.is_dir() or cn_dir.name not in CN_TO_LABEL:
            continue
        label = CN_TO_LABEL[cn_dir.name]
        for f in cn_dir.iterdir():
            if f.suffix.lower() in ('.wav', '.mp3', '.m4a', '.ogg'):
                audio_files.append((str(f), label, cn_dir.name))
    
    print(f'\n共 {len(audio_files)} 条音频, {len(set(l for _,l,_ in audio_files))} 类')
    
    # 推理 (10类原始 + 6类合并)
    results_10 = defaultdict(lambda: {'correct': 0, 'total': 0, 'preds': []})
    results_6 = defaultdict(lambda: {'correct': 0, 'total': 0, 'preds': []})
    t0 = time.time()
    
    for path, true_label, cn_name in audio_files:
        mel = audio_to_mel(path).to(DEVICE)
        with torch.no_grad():
            logits = model(mel)
            pred_idx = logits.argmax(1).item()
            pred_label = LABELS[pred_idx]
        
        # 10类评价
        correct_10 = pred_label == true_label
        results_10[true_label]['total'] += 1
        results_10[true_label]['correct'] += correct_10
        results_10[true_label]['preds'].append(pred_label)
        
        # 6类评价
        true_6 = CN_TO_6[cn_name]
        pred_6 = LABEL_TO_6[pred_label]
        results_6[true_6]['total'] += 1
        results_6[true_6]['correct'] += pred_6 == true_6
        results_6[true_6]['preds'].append(pred_6)
    
    elapsed = time.time() - t0
    
    # 打印10类结果
    print(f'\n{"═"*50}')
    print(f' 10类 × 数据集对比 (fold3: F1=68.9%)')
    print(f'{"═"*50}')
    total_c = 0
    total_a = 0
    for label in sorted(results_10.keys()):
        r = results_10[label]
        acc = r['correct'] / r['total'] if r['total'] > 0 else 0
        total_c += r['correct']
        total_a += r['total']
        dist = Counter(r['preds']).most_common(3)
        print(f'  {label:12s}  {r["correct"]:2d}/{r["total"]:2d} = {acc:.1%}  → {", ".join(f"{k}={v}" for k,v in dist)}')
    print(f'  总体: {total_c}/{total_a} = {total_c/total_a:.1%}')
    
    # 打印6类结果
    print(f'\n{"═"*50}')
    print(f' 6类合并 (CNN 预测 10类→映射→6类)')
    print(f'{"═"*50}')
    total_c = 0
    total_a = 0
    for six_label in ['Aggressive', 'Happy', 'Mating', 'MotherCall', 'Paining', 'Resting']:
        if six_label not in results_6:
            continue
        r = results_6[six_label]
        acc = r['correct'] / r['total'] if r['total'] > 0 else 0
        total_c += r['correct']
        total_a += r['total']
        dist = Counter(r['preds']).most_common(3)
        print(f'  {six_label:12s}  {r["correct"]:2d}/{r["total"]:2d} = {acc:.1%}  → {", ".join(f"{k}={v}" for k,v in dist)}')
    print(f'  总体: {total_c}/{total_a} = {total_c/total_a:.1%}')
    print(f'  耗时: {elapsed:.1f}s')

if __name__ == '__main__':
    main()

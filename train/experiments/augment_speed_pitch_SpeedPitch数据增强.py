"""
Speed/Pitch 数据增强 — Pandeya论文原始方法
对 CatSound_V2 每条音频生成：
  - Speed +10%, Speed -10%
  - Pitch +2 半音, Pitch -2 半音
然后缓存 Mel 频谱

用法: python3 augment_speed_pitch.py
输出: Desktop/模型代码/训练结果/aug_sp/ 目录下的 .npy 文件
"""
import os, sys, csv, hashlib, warnings, time
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import librosa
import torch

SR = 16000
N_MELS = 128
N_FFT = 512
HOP_LENGTH = 256
VAD_FRAMES = 100
VAD_TOP_DB = 25

BASE = Path('/mnt/c/Users/陈项目负责人/Desktop')
DATA_DIR = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
CACHE_DIR = BASE / '模型代码/训练结果/aug_sp_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["Angry","Defence","Fighting","Happy","HuntingMind",
          "Mating","MotherCall","Paining","Resting","Warning"]

# 增强参数（Pandeya论文: speed ±10%, pitch ±2半音）
AUGMENTATIONS = [
    ('sp10', lambda y, sr: librosa.effects.time_stretch(y=y, rate=1.1)),     # +10%
    ('sm10', lambda y, sr: librosa.effects.time_stretch(y=y, rate=0.9)),     # -10%
    ('pp2',  lambda y, sr: librosa.effects.pitch_shift(y=y, sr=sr, n_steps=2)),   # +2半音
    ('pm2',  lambda y, sr: librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-2)),  # -2半音
]

def extract_longest_vocal(y):
    segs = librosa.effects.split(y, top_db=VAD_TOP_DB, frame_length=512, hop_length=HOP_LENGTH)
    if len(segs) == 0: return y
    longest = max(segs, key=lambda s: s[1] - s[0])
    return y[longest[0]:longest[1]]

def audio_to_mel(audio_path):
    """加载音频 → VAD → 归一化 → Mel频谱 → 固定100帧"""
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    y = extract_longest_vocal(y)
    peak = np.max(np.abs(y))
    y = y / peak if peak > 1e-10 else np.zeros_like(y)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    if mel_norm.shape[1] < VAD_FRAMES:
        mel_norm = np.pad(mel_norm, ((0, 0), (0, VAD_FRAMES - mel_norm.shape[1])))
    else:
        s = (mel_norm.shape[1] - VAD_FRAMES) // 2
        mel_norm = mel_norm[:, s:s + VAD_FRAMES]
    return mel_norm

def main():
    # 加载文件列表
    csv_path = DATA_DIR / 'Catmood_NAYA_1xAug.csv'
    file_list = []
    label_list = []
    cache_hits = 0
    cache_misses = 0

    with open(csv_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        try: next(reader)
        except: pass
        for row in reader:
            if len(row) < 3: continue
            li = int(row[2])
            fn = os.path.basename(row[1].strip())
            lp = DATA_DIR / LABELS[li] / fn
            if lp.exists():
                file_list.append(str(lp))
                label_list.append(li)

    print(f'加载 {len(file_list)} 条原始音频')
    print(f'增强类型: {len(AUGMENTATIONS)} 种')
    print(f'预计总 Mel 数: {len(file_list) * (1 + len(AUGMENTATIONS))}')
    print()

    # 对每条音频生成增强 Mel 并缓存
    for i, fp in enumerate(file_list):
        base_name = hashlib.md5(fp.encode()).hexdigest()
        base_cache = CACHE_DIR / f'{base_name}.npy'
        
        # 原始 Mel（从已有缓存或即时计算）
        if base_cache.exists():
            cache_hits += 1
        else:
            mel = audio_to_mel(fp)
            np.save(base_cache, mel)
            cache_misses += 1

        # 4种增强 Mel
        for aug_name, aug_fn in AUGMENTATIONS:
            aug_cache = CACHE_DIR / f'{base_name}_{aug_name}.npy'
            if aug_cache.exists():
                cache_hits += 1
            else:
                try:
                    y, sr = librosa.load(fp, sr=SR, mono=True)
                    y_aug = aug_fn(y, sr)
                    # 增强后重新VAD+Mel
                    y_aug = extract_longest_vocal(y_aug)
                    peak = np.max(np.abs(y_aug))
                    y_aug = y_aug / peak if peak > 1e-10 else np.zeros_like(y_aug)
                    mel = librosa.feature.melspectrogram(y=y_aug, sr=SR, n_mels=N_MELS,
                                                         n_fft=N_FFT, hop_length=HOP_LENGTH)
                    mel_db = librosa.power_to_db(mel, ref=np.max)
                    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
                    if mel_norm.shape[1] < VAD_FRAMES:
                        mel_norm = np.pad(mel_norm, ((0,0), (0, VAD_FRAMES-mel_norm.shape[1])))
                    else:
                        s = (mel_norm.shape[1] - VAD_FRAMES) // 2
                        mel_norm = mel_norm[:, s:s+VAD_FRAMES]
                    np.save(aug_cache, mel_norm)
                    cache_misses += 1
                except Exception as e:
                    print(f'  增强失败 {os.path.basename(fp)} [{aug_name}]: {e}')

        if (i + 1) % 500 == 0 or i == 0:
            eta = (time.time() - t0) / (i + 1) * (len(file_list) - i - 1) if i > 0 else 0
            print(f'  [{i+1}/{len(file_list)}] 命中={cache_hits} 新增={cache_misses} 预计剩余={eta:.0f}s')

    print(f'\n完成! 缓存目录: {CACHE_DIR}')
    print(f'缓存命中: {cache_hits}, 新增: {cache_misses}')

    # 输出训练用的文件列表 + 标签
    out_path = CACHE_DIR / 'aug_file_list.txt'
    with open(out_path, 'w') as f:
        for i, fp in enumerate(file_list):
            base_name = hashlib.md5(fp.encode()).hexdigest()
            for aug_name, _ in AUGMENTATIONS:
                f.write(f'{base_name}_{aug_name}.npy\t{label_list[i]}\n')

if __name__ == '__main__':
    t0 = time.time()
    main()

"""
音视频Speed/Pitch增强 — 生成增强WAV文件供管线使用
对 CatSound_V2 每条非增强音频生成 Speed ±10%, Pitch ±2半音

输出: DATA_DIR / 'aug_SP/' 下按类别存放的 .wav 文件
"""

import os, sys, csv, warnings, time, concurrent.futures
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import librosa

SR = 16000
BASE = Path('/mnt/c/Users/陈项目负责人/Desktop')
DATA_DIR = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
OUT_DIR = DATA_DIR / 'aug_SP'

LABELS = ["Angry","Defence","Fighting","Happy","HuntingMind",
          "Mating","MotherCall","Paining","Resting","Warning"]

AUGS = [
    ('sp10', lambda y, sr: librosa.effects.time_stretch(y=y, rate=1.1)),
    ('sm10', lambda y, sr: librosa.effects.time_stretch(y=y, rate=0.9)),
    ('pp2',  lambda y, sr: librosa.effects.pitch_shift(y=y, sr=sr, n_steps=2)),
    ('pm2',  lambda y, sr: librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-2)),
]

def process_file(args):
    fp, label, label_idx, is_aug = args
    if is_aug:
        return None  # 跳过已增强的文件
    try:
        y, _ = librosa.load(fp, sr=SR, mono=True)
    except:
        return None
    
    stem = Path(fp).stem
    results = []
    for suffix, fn in AUGS:
        try:
            y_aug = fn(y, SR)
            y_aug = np.nan_to_num(y_aug)
        except:
            continue
        out_name = f'{stem}_{suffix}.wav'
        out_dir = OUT_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_fp = out_dir / out_name
        # librosa.output.write_wav was removed in newer versions
        import soundfile as sf
        sf.write(str(out_fp), y_aug, SR)
        results.append((out_name, label, label_idx, f'{stem}_{suffix}'))
    return results

def main():
    csv_path = DATA_DIR / 'Catmood_NAYA_1xAug.csv'
    files, labels, is_aug = [], [], []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 3: continue
            label_idx = int(row[2])
            filename = os.path.basename(row[1].strip())
            local_path = DATA_DIR / LABELS[label_idx] / filename
            if local_path.exists():
                files.append(str(local_path))
                labels.append(LABELS[label_idx])
                is_aug.append('_aug1(1)' in filename)
    
    print(f'加载 {len(files)} 条，其中增强 {sum(is_aug)} 条，原始 {len(files)-sum(is_aug)} 条')
    t0 = time.time()
    
    args_list = [(f, labels[i], _, is_aug[i]) for i, (f, _) in enumerate(zip(files, labels))]
    total = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        for result in executor.map(process_file, args_list):
            if result is None: continue
            for r in result:
                total += 1
    
    print(f'\n完成！生成 {total} 条增强音频, 耗时 {time.time()-t0:.0f}s')
    print(f'输出目录: {OUT_DIR}')

if __name__ == '__main__':
    main()

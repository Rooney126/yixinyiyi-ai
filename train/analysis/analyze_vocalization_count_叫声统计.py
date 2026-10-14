"""
CatSound_V2 各类每文件猫叫段数统计
验证假设: Defence 每文件叫声数远少于 Happy 等类
用 librosa.effects.split 做静音分割 (非手写 VAD)
# 统计每段音频中叫声事件数量分布
"""

import numpy as np
import librosa
from pathlib import Path

SR = 16000
HOP = 256
DATA_DIR = Path(r'/path/to/dataset\CatSound_DataSet_V2\NAYA_DATA_AUG1X')

LABELS = {
    'Angry': 0, 'Defence': 1, 'Fighting': 2, 'Happy': 3,
    'HuntingMind': 4, 'Mating': 5, 'MotherCall': 6,
    'Paining': 7, 'Resting': 8, 'Warning': 9,
}

# 对每个文件: 检测非静音段数 + 总时长
# librosa.effects.split : top_db=30 为默认, 低于阈值30dB 视为静音

def count_segments(y, sr, top_db=30):
    """
    librosa.effects.split 能量阈值分割, 返回非静音段数和每个片段的帧数
    """
    # split 返回 [(start, end), ...] 帧索引
    intervals = librosa.effects.split(y, top_db=top_db,
                                       frame_length=2048, hop_length=HOP)
    if len(intervals) == 0:
        return 0, []
    
    durations = [(end - start) * HOP / sr for start, end in intervals]
    
    # 过滤极短段(<0.05s, 可能是噪声)
    valid_intervals = [(s, e) for (s, e), d in zip(intervals, durations) if d > 0.05]
    valid_durations = [d for d in durations if d > 0.05]
    
    return len(valid_intervals), valid_durations

print('═' * 65)
print('  CatSound_V2 每文件非静音段数统计')
print('  (librosa.effects.split)')
print('═' * 65)

results = {}
for label in LABELS:
    dir_path = DATA_DIR / label
    if not dir_path.exists():
        print(f'  [!] {label}: 目录不存在')
        continue
    
    files = sorted(dir_path.glob('*.mp3'))
    counts = []
    all_durations = []
    zeros = 0
    
    for i, f in enumerate(files):
        try:
            y, sr_orig = librosa.load(str(f), sr=SR, mono=True)
        except Exception as e:
            zeros += 1
            continue
        
        n, durations = count_segments(y, sr_orig)
        counts.append(n)
        all_durations.extend(durations)
        if n == 0:
            zeros += 1
    
    counts = np.array(counts)
    results[label] = {
        'n_files': len(counts),
        'mean': counts.mean(),
        'median': np.median(counts),
        'std': counts.std(),
        'min': counts.min(),
        'max': counts.max(),
        'sum': int(counts.sum()),
        'zeros': zeros,
        'seg_dur_mean': np.mean(all_durations) if all_durations else 0,
        'seg_dur_std': np.std(all_durations) if all_durations else 0,
    }
    
    print(f'\n  {label:-<15s} {len(counts):>4d} 文件')
    print(f'    每文件非静音段: {results[label]["mean"]:.2f} ± {results[label]["std"]:.2f}')
    print(f'    中位数 {results[label]["median"]:.1f}, 范围 {results[label]["min"]:.0f}-{results[label]["max"]:.0f}')
    print(f'    总段数: {results[label]["sum"]}, 静音文件: {results[label]["zeros"]}')
    print(f'    平均段长: {results[label]["seg_dur_mean"]:.2f}s')

# 汇总
print(f'\n{"="*65}')
print(f'  对比 (按平均段数降序)')
print(f'{"="*65}')
print(f'  {"类名":<15s} {"文件数":>5s} {"平均段数":>7s} {"  ±std":<6s} {"总段数":>6s} {"静音文件":>7s} {"段长":>6s}')
print(f'  {"─"*55}')
for label, s in sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True):
    print(f'  {label:<15s} {s["n_files"]:>5d} {s["mean"]:>7.2f} ±{s["std"]:<5.2f} {s["sum"]:>6d} {s["zeros"]:>7d} {s["seg_dur_mean"]:>5.2f}s')

print(f'\n  如果 Defence=1 段而 Happy=3+ 段 → 叫声数量偏差坐实')

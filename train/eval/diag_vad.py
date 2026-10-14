"""
诊断: 检查 VAD 分段为什么把多声合成一段
"""
import os
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stdout.reconfigure(encoding='utf-8')


import librosa
import numpy as np

SR = 16000
FRAME_LEN = 2048
HOP_LENGTH = 256

audio_path = "/Desktop/cat_test.wav"
y, _ = librosa.load(audio_path, sr=SR, mono=True)

# 帧RMS
n = len(y)
frames = []
for start in range(0, n - FRAME_LEN + 1, HOP_LENGTH):
    frame = y[start:start + FRAME_LEN]
    rms = np.sqrt(np.mean(frame ** 2))
    frames.append(rms)
rms = np.array(frames, dtype=np.float32)

# 安静值 & 阈值
sorted_rms = np.sort(rms)
n_bottom = max(1, int(len(rms) * 0.15))
quiet_value = float(np.median(sorted_rms[:n_bottom]))
onset_thresh = quiet_value * 1.5
offset_thresh = quiet_value * 1.2

print(f"音频长度: {len(y)/SR:.2f}s")
print(f"安静值: {quiet_value:.6f}")
print(f"onset阈值: {onset_thresh:.6f}")
print(f"offset阈值: {offset_thresh:.6f}")
print()

# 找到所有超过 onset 的帧区间
in_call = False
cur_onset = None
segments = []
for i in range(len(rms)):
    if not in_call:
        if rms[i] > onset_thresh:
            cur_onset = i
            in_call = True
    else:
        if rms[i] < offset_thresh:
            segments.append((cur_onset, i))
            in_call = False
            cur_onset = None
if in_call and cur_onset is not None:
    segments.append((cur_onset, len(rms) - 1))

print("VAD 检测到的叫声段(加缓冲前):")
for onset, offset in segments:
    dur = (offset - onset) * HOP_LENGTH / SR
    max_rms = float(np.max(rms[onset:offset+1]))
    print(f"  帧 {onset:4d}-{offset:4d}  时长 {dur:.2f}s  maxRMS {max_rms:.4f}")

print()

# RMS 曲线分段打印(仅打印叫声附近)
print("RMS 曲线(峰值>onset_thresh的帧标注):")
# 取一个范围看看
for i in range(0, len(rms)):
    marker = ""
    if rms[i] > onset_thresh:
        marker = " ↑ONSET"
    elif rms[i] > offset_thresh and rms[i] <= onset_thresh:
        marker = " 介于"
    if rms[i] > offset_thresh:
        time_s = i * HOP_LENGTH / SR
        print(f"  {time_s:5.2f}s  RMS={rms[i]:.6f}{marker}")

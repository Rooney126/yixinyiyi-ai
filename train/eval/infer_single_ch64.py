"""
译芯怡意 — ch64 CNN 单条音频推理
用 A_aug fold3 模型（部署版，87.9% F1）
用法: python infer_single_ch64.py <音频文件路径>
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn as nn
import librosa
from pathlib import Path

# ── 与训练管线完全一致的参数 ──
SR = 16000
MAX_DUR = 30.0
FRAME_LEN = 2048
HOP_LENGTH = 256
N_MELS = 128
FMIN = 50
FMAX = 8000
N_FFT = 512
MEL_FRAMES = 80
PROP_DECREASE = 0.8
MIN_RMS_THRESH = 0.03
MIN_NOISE_SAMPLES = 2048
BUF_SEC = 0.075
MIN_SEG_DUR = 0.25
ONSET_MULT = 1.5
OFFSET_MULT = 1.2

import noisereduce as nr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABELS = [
    "Angry","Defence","Fighting","Happy","HuntingMind",
    "Mating","MotherCall","Paining","Resting","Warning"
]

CN_LABELS = [
    "生气","防御","战斗","开心","狩猎之心",
    "渴求","寻找妈妈","疼痛","休息中","警告"
]

# ── 模型定义 ──
class FDAPMobileCatCNN(nn.Module):
    def __init__(self, num_classes=10, fdap_bands=4):
        super().__init__()
        self.fdap_bands = fdap_bands
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((1, 2)))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 48, 3, padding=1),
            nn.BatchNorm2d(48), nn.ReLU(), nn.MaxPool2d((1, 2)))
        self.conv4 = nn.Sequential(
            nn.Conv2d(48, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.Identity())
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 96), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(96, num_classes))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        b, c, f, t = x.shape
        bs = f // self.fdap_bands
        fdap = [
            x[:, :, i * bs:(i + 1) * bs if i < 3 else f, :].mean(dim=[2, 3])
            for i in range(self.fdap_bands)
        ]
        x = torch.cat(fdap, dim=1)
        return self.classifier(x)

# ── 预处理管线 ──
def compute_frame_rms(y):
    n = len(y)
    frames = []
    for start in range(0, n - FRAME_LEN + 1, HOP_LENGTH):
        frame = y[start:start + FRAME_LEN]
        rms = np.sqrt(np.mean(frame ** 2))
        frames.append(rms)
    return np.array(frames, dtype=np.float32)

def find_bottom_longest_continuous(rms, bottom_pct=0.15):
    n_frames = len(rms)
    n_bottom = max(1, int(n_frames * bottom_pct))
    sorted_indices = np.argsort(rms)
    bottom_indices = set(sorted_indices[:n_bottom])
    longest_start, longest_end = 0, 0
    cur_start = None
    for i in range(n_frames):
        if i in bottom_indices:
            if cur_start is None: cur_start = i
        else:
            if cur_start is not None:
                if i - cur_start > longest_end - longest_start:
                    longest_start, longest_end = cur_start, i
                cur_start = None
    if cur_start is not None:
        if n_frames - cur_start > longest_end - longest_start:
            longest_start, longest_end = cur_start, n_frames
    n_samples = (longest_end - longest_start) * HOP_LENGTH
    if n_samples < MIN_NOISE_SAMPLES and bottom_pct < 0.25:
        return find_bottom_longest_continuous(rms, min(bottom_pct + 0.05, 0.25))
    return longest_start * HOP_LENGTH, longest_end * HOP_LENGTH, bottom_pct

def detect_segments(rms, quiet_pct=0.15):
    n_frames = len(rms)
    if n_frames < 2: return []
    n_bottom = max(1, int(n_frames * quiet_pct))
    sorted_rms = np.sort(rms)
    quiet_value = float(np.median(sorted_rms[:n_bottom]))
    if quiet_value < 1e-10:
        quiet_value = np.percentile(rms, quiet_pct * 100)
    onset_thresh = quiet_value * ONSET_MULT
    offset_thresh = quiet_value * OFFSET_MULT
    if onset_thresh < 1e-8 or onset_thresh <= offset_thresh:
        if quiet_pct < 0.25:
            return detect_segments(rms, quiet_pct + 0.05)
        return []
    segments = []
    in_call = False
    cur_onset = None
    for i in range(n_frames):
        if not in_call:
            if rms[i] > onset_thresh:
                cur_onset = i; in_call = True
        else:
            if rms[i] < offset_thresh:
                segments.append((cur_onset, i))
                in_call = False; cur_onset = None
    if in_call and cur_onset is not None:
        segments.append((cur_onset, n_frames - 1))
    buf_frames = int(BUF_SEC * SR / HOP_LENGTH)
    results = []
    for onset_frame, offset_frame in segments:
        orig_dur = (offset_frame - onset_frame) * HOP_LENGTH / SR
        if orig_dur < MIN_SEG_DUR: continue
        seg_max_rms = float(np.max(rms[onset_frame:offset_frame + 1]))
        if seg_max_rms < 0.01: continue
        buf_onset = max(0, onset_frame - buf_frames)
        buf_offset = min(n_frames - 1, offset_frame + buf_frames)
        results.append({
            'onset_sample': int(buf_onset * HOP_LENGTH),
            'offset_sample': int((buf_offset + 1) * HOP_LENGTH),
            'orig_dur': orig_dur,
        })
    return results

def split_overlapping(segments, rms):
    if len(segments) <= 1: return segments
    result = [segments[0]]
    for seg in segments[1:]:
        prev = result[-1]
        prev_end = prev['offset_sample']
        cur_start = seg['onset_sample']
        if cur_start < prev_end:
            overlap_start_frame = prev_end // HOP_LENGTH
            overlap_end_frame = cur_start // HOP_LENGTH
            overlap_frames = rms[overlap_start_frame:overlap_end_frame + 1]
            if len(overlap_frames) > 0:
                split_frame = overlap_start_frame + int(np.argmin(overlap_frames))
            else:
                split_frame = (overlap_start_frame + overlap_end_frame) // 2
            split_sample = split_frame * HOP_LENGTH
            prev['offset_sample'] = split_sample
            if split_sample > seg['onset_sample']:
                seg['onset_sample'] = split_sample
        result.append(seg)
    return result

def process_audio(file_path):
    y, _ = librosa.load(file_path, sr=SR, mono=True)
    max_samples = int(MAX_DUR * SR)
    if len(y) > max_samples: y = y[:max_samples]
    if len(y) < N_FFT: return []
    rms = compute_frame_rms(y)
    if len(rms) < 2: return []
    noise_start, noise_end, _ = find_bottom_longest_continuous(rms)
    y_noise = y[noise_start:noise_end]
    if len(y_noise) >= FRAME_LEN:
        min_rms_val = float(np.min(rms)) if len(rms) > 0 else 0.0
        stationary = min_rms_val < MIN_RMS_THRESH
        try:
            y_denoised = nr.reduce_noise(y=y, sr=SR, y_noise=y_noise,
                stationary=stationary, prop_decrease=PROP_DECREASE, n_fft=FRAME_LEN)
            y_denoised = np.nan_to_num(y_denoised)
        except: y_denoised = y
    else: y_denoised = y
    rms_denoised = compute_frame_rms(y_denoised)
    if len(rms_denoised) < 2: return []
    raw_segments = detect_segments(rms_denoised)
    if not raw_segments: return []
    clean_segments = split_overlapping(raw_segments, rms_denoised)
    results = []
    for seg in clean_segments:
        onset_s = seg['onset_sample']
        offset_s = min(seg['offset_sample'], len(y_denoised))
        seg_y = y_denoised[onset_s:offset_s]
        if len(seg_y) < N_FFT: continue
        peak = np.max(np.abs(seg_y))
        if peak > 1e-10: seg_y = seg_y / peak
        else: continue
        mel = librosa.feature.melspectrogram(y=seg_y, sr=SR, n_mels=N_MELS,
            fmin=FMIN, fmax=FMAX, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_min, mel_max = mel_db.min(), mel_db.max()
        if mel_max - mel_min < 1e-10: continue
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min)
        if mel_norm.shape[1] < MEL_FRAMES:
            mel_norm = np.pad(mel_norm, ((0, 0), (0, MEL_FRAMES - mel_norm.shape[1])))
        else:
            start = (mel_norm.shape[1] - MEL_FRAMES) // 2
            mel_norm = mel_norm[:, start:start + MEL_FRAMES]
        results.append({'mel': mel_norm.astype(np.float32), 'orig_dur': seg['orig_dur']})
    return results

def main():
    if len(sys.argv) < 2:
        print("用法: python infer_single_ch64.py <音频文件路径>")
        print("支持: .wav .mp3 .mp4 .m4a .ogg 等 librosa 支持的格式")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"❌ 文件不存在: {audio_path}")
        sys.exit(1)

    # 确定模型路径
    script_dir = Path(__file__).resolve().parent
    model_dir = script_dir.parent / "output" / "fdap_cnn_ch64_A_aug"
    model_path = model_dir / "fold3_best.pth"

    if not model_path.exists():
        # 备选: app/model 下
        alt_path = script_dir.parent.parent / "app" / "model" / "fold3_best.pth"
        if alt_path.exists():
            model_path = alt_path
        else:
            print(f"❌ 找不到模型文件")
            print(f"   尝试路径1: {model_path}")
            print(f"   尝试路径2: {alt_path}")
            sys.exit(1)

    print(f"模型: {model_path}")
    print(f"设备: {DEVICE}")
    print(f"音频: {os.path.basename(audio_path)}")
    print()

    # 加载模型
    model = FDAPMobileCatCNN(num_classes=10).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("✅ 模型加载成功")

    # 预处理
    print("⏳ 预处理中...")
    segments = process_audio(audio_path)
    if not segments:
        print("❌ 未检测到猫叫声段")
        sys.exit(1)
    print(f"✅ 检测到 {len(segments)} 个声音段")
    print()

    # 推理
    all_probs = []
    with torch.no_grad():
        for i, seg in enumerate(segments):
            mel_t = torch.from_numpy(seg['mel']).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            logits = model(mel_t)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            all_probs.append((probs, seg['orig_dur']))
            top_idx = int(np.argmax(probs))
            print(f"  段{i+1} (时长{seg['orig_dur']:.2f}s): "
                  f"{CN_LABELS[top_idx]}({LABELS[top_idx]}) {probs[top_idx]*100:.1f}%")

    # 等权投票
    print()
    n_segments = len(all_probs)
    votes = np.zeros(10)
    for probs, _ in all_probs:
        votes += probs  # 等权
    final_probs = votes / n_segments
    top5_idx = np.argsort(final_probs)[::-1][:5]

    print("═══ 最终结果（等权投票）═══")
    for idx in top5_idx:
        bar = '█' * int(final_probs[idx] * 30)
        print(f"  {CN_LABELS[idx]:6s} ({LABELS[idx]:12s})  {final_probs[idx]*100:.1f}% {bar}")
    print()
    top = top5_idx[0]
    print(f"👉 结论: {CN_LABELS[top]}({LABELS[top]}) 置信度 {final_probs[top]*100:.1f}%")

if __name__ == '__main__':
    main()

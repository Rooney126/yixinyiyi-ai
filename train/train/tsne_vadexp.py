r"""
t-SNE 可视化 — FDAP ch64 嵌入空间 (适配 train_cnn_pipeline_v2_vadexp.py)

用法:
  cd PROJECT_ROOT\train\train
  /path/to/user\AppData\Local\Programs\Python\Python310\python.exe tsne_vadexp.py

输出: PROJECT_ROOT\train\output\tsne_result.png
"""

import csv
import os
import sys
import time
import warnings

# 强制 UTF-8 输出 (Windows GBK 终端无法打印 emoji)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 中文字体修复（Windows: 微软雅黑/黑体）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
import librosa
import noisereduce as nr
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn

# ═══════════════════════════════════════
#  配置 (与 train_cnn_pipeline_v2_vadexp.py 一致)
# ═══════════════════════════════════════

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
MIN_SEG_DUR = 0.25
ONSET_MULT = 3.5
OFFSET_MULT = 3.0
BUF_SEC = 0.075

LABELS = ["Angry","Defence","Fighting","Happy","HuntingMind",
          "Mating","MotherCall","Paining","Resting","Warning"]

DATA_DIR = Path(r'/path/to/dataset\CatSound_DataSet_V2\NAYA_DATA_AUG1X')
CACHE_DIR = Path(r'/path/to/user\AppData\Local\Temp\seg_cache_full')
MODEL_PATH = Path(r'PROJECT_ROOT\train\output\fdap_cnn_ch64_A_aug\fold3_best.pth')
OUTPUT_DIR = Path(r'PROJECT_ROOT\train\output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'设备: {DEVICE}')
if DEVICE == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name(0)}')

# ═══════════════════════════════════════
#  模型定义 (与训练脚本一致)
# ═══════════════════════════════════════

class FDAPMobileCatCNN_Embed(nn.Module):
    """与训练脚本一致, 但返回 256-dim 嵌入 (在 classifier 之前)"""
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
        self.fdap_norm = nn.BatchNorm1d(256)  # 训练中没有, 但推理时加上会平移嵌入

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        b, c, f, t = x.shape
        bs = f // self.fdap_bands
        bands = []
        for i in range(self.fdap_bands):
            if i < self.fdap_bands - 1:
                band = x[:, :, i * bs:(i + 1) * bs, :].mean(dim=[2, 3])
            else:
                band = x[:, :, i * bs:, :].mean(dim=[2, 3])
            bands.append(band)
        embed = torch.cat(bands, dim=1)  # (B, 256)
        return embed


# ═══════════════════════════════════════
#  管线函数 (从训练脚本复制)
# ═══════════════════════════════════════

def compute_frame_rms(y, frame_len=FRAME_LEN, hop=HOP_LENGTH):
    n = len(y)
    frames = []
    for start in range(0, n - frame_len + 1, hop):
        frame = y[start:start + frame_len]
        rms = np.sqrt(np.mean(frame ** 2))
        frames.append(rms)
    return np.array(frames, dtype=np.float32)


def collect_noise_samples(rms, y, bottom_pct=0.15, min_samples=MIN_NOISE_SAMPLES, hop=HOP_LENGTH):
    n_frames = len(rms)
    n_bottom = max(1, int(n_frames * bottom_pct))
    sorted_indices = np.argsort(rms)[:n_bottom]
    noise_pieces = []
    total_samples = 0
    for idx in sorted_indices:
        start_s = idx * hop
        end_s = min((idx + 1) * hop, len(y))
        if end_s - start_s > 0:
            noise_pieces.append(y[start_s:end_s])
            total_samples += (end_s - start_s)
    if total_samples < min_samples and bottom_pct < 0.40:
        new_pct = min(bottom_pct + 0.05, 0.40)
        return collect_noise_samples(rms, y, new_pct, min_samples, hop)
    y_noise = np.concatenate(noise_pieces) if len(noise_pieces) > 0 else np.array([])
    return y_noise, bottom_pct


def detect_segments(rms, quiet_pct=0.40, hop=HOP_LENGTH, sr=SR):
    n_frames = len(rms)
    if n_frames < 2:
        return []
    n_bottom = max(1, int(n_frames * quiet_pct))
    sorted_rms = np.sort(rms)
    quiet_value = float(np.median(sorted_rms[:n_bottom]))
    if quiet_value < 1e-10:
        quiet_value = np.percentile(rms, quiet_pct * 100)
    onset_thresh = quiet_value * ONSET_MULT
    offset_thresh = quiet_value * OFFSET_MULT
    if onset_thresh < 1e-8 or (onset_thresh <= offset_thresh):
        if quiet_pct < 0.50:
            return detect_segments(rms, quiet_pct + 0.05, hop, sr)
        return []
    segments = []
    in_call = False
    cur_onset = None
    for i in range(n_frames):
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
        segments.append((cur_onset, n_frames - 1))
    buf = BUF_SEC
    buf_frames = int(buf * sr / hop)
    results = []
    for onset_frame, offset_frame in segments:
        orig_dur = (offset_frame - onset_frame) * hop / sr
        if orig_dur < MIN_SEG_DUR:
            continue
        seg_max_rms = float(np.max(rms[onset_frame:offset_frame + 1]))
        if seg_max_rms < 0.01:
            continue
        buf_onset = max(0, onset_frame - buf_frames)
        buf_offset = min(n_frames - 1, offset_frame + buf_frames)
        results.append({
            'onset_sample': int(buf_onset * hop),
            'offset_sample': int((buf_offset + 1) * hop),
            'orig_dur': orig_dur,
        })
    return results


def split_overlapping(segments, rms, hop=HOP_LENGTH):
    if len(segments) <= 1:
        return segments
    result = [segments[0]]
    for seg in segments[1:]:
        prev = result[-1]
        prev_end = prev['offset_sample']
        cur_start = seg['onset_sample']
        if cur_start < prev_end:
            overlap_start_frame = prev_end // hop
            overlap_end_frame = cur_start // hop
            overlap_frames = rms[overlap_start_frame:overlap_end_frame + 1]
            if len(overlap_frames) > 0:
                min_frame_offset = int(np.argmin(overlap_frames))
                split_frame = overlap_start_frame + min_frame_offset
            else:
                split_frame = (overlap_start_frame + overlap_end_frame) // 2
            split_sample = split_frame * hop
            prev['offset_sample'] = split_sample
            if split_sample > seg['onset_sample']:
                seg['onset_sample'] = split_sample
        result.append(seg)
    return result


def process_audio(file_path):
    """返回 list of mel (128, 80)"""
    try:
        y, _ = librosa.load(file_path, sr=SR, mono=True)
    except Exception as e:
        return []
    max_samples = int(MAX_DUR * SR)
    if len(y) > max_samples:
        y = y[:max_samples]
    rms = compute_frame_rms(y)
    if len(rms) < 2:
        return []
    # 降噪
    y_noise, _ = collect_noise_samples(rms, y)
    if len(y_noise) >= FRAME_LEN:
        min_rms_val = float(np.min(rms)) if len(rms) > 0 else 0.0
        stationary = min_rms_val < MIN_RMS_THRESH
        try:
            y_denoised = nr.reduce_noise(
                y=y, sr=SR, y_noise=y_noise, stationary=stationary,
                prop_decrease=PROP_DECREASE, n_fft=FRAME_LEN,
            )
            y_denoised = np.nan_to_num(y_denoised)
        except Exception:
            y_denoised = y
    else:
        y_denoised = y
    rms_denoised = compute_frame_rms(y_denoised)
    if len(rms_denoised) < 2:
        return []
    raw_segments = detect_segments(rms_denoised)
    if not raw_segments:
        return []
    clean_segments = split_overlapping(raw_segments, rms_denoised)
    if not clean_segments:
        return []
    mels = []
    for seg in clean_segments:
        onset_s = seg['onset_sample']
        offset_s = min(seg['offset_sample'], len(y_denoised))
        seg_y = y_denoised[onset_s:offset_s]
        if len(seg_y) < N_FFT:
            continue
        peak = np.max(np.abs(seg_y))
        if peak > 1e-10:
            seg_y = seg_y / peak
        else:
            continue
        mel = librosa.feature.melspectrogram(
            y=seg_y, sr=SR, n_mels=N_MELS,
            fmin=FMIN, fmax=FMAX, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_min, mel_max = mel_db.min(), mel_db.max()
        if mel_max - mel_min < 1e-10:
            continue
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min)
        if mel_norm.shape[1] < MEL_FRAMES:
            mel_norm = np.pad(mel_norm, ((0, 0), (0, MEL_FRAMES - mel_norm.shape[1])))
        else:
            start = (mel_norm.shape[1] - MEL_FRAMES) // 2
            mel_norm = mel_norm[:, start:start + MEL_FRAMES]
        mels.append(mel_norm.astype(np.float32))
    return mels


def load_file_list(data_dir, max_files_per_class=200):
    """加载 CatSound_V2, 每个类别最多 N 条 (控制 t-SNE 点数)"""
    csv_path = data_dir / 'Catmood_NAYA_1xAug.csv'
    all_files, all_labels = [], []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return [], []
        for row in reader:
            if len(row) < 3:
                continue
            label_idx = int(row[2])
            filename = os.path.basename(row[1].strip())
            local_path = data_dir / LABELS[label_idx] / filename
            if local_path.exists():
                all_files.append(str(local_path))
                all_labels.append(LABELS[label_idx])
    
    # 每个类取样
    from collections import defaultdict
    class_files = defaultdict(list)
    for fp, lb in zip(all_files, all_labels):
        class_files[lb].append(fp)
    
    sampled_files, sampled_labels = [], []
    for lb in LABELS:
        names = class_files.get(lb, [])
        if max_files_per_class > 0:
            names = np.random.RandomState(42).choice(
                names, size=min(max_files_per_class, len(names)), replace=False).tolist()
        sampled_files.extend(names)
        sampled_labels.extend([lb] * len(names))
    
    print(f'  文件总加载: {len(all_files)}, 取样后: {len(sampled_files)} '
          f'(每类最多 {max_files_per_class})')
    return np.array(sampled_files), np.array(sampled_labels)


# ═══════════════════════════════════════
#  Main: t-SNE
# ═══════════════════════════════════════

def main():
    print(f'\n{"█" * 50}')
    print('  t-SNE: FDAP ch64 嵌入空间 (fold3_best.pth)')
    print(f'  {DEVICE=}')
    print(f'{"█" * 50}\n')

    # ── 加载模型 ──
    print('[0] 加载模型...')
    model = FDAPMobileCatCNN_Embed().to(DEVICE)
    model.eval()
    if not MODEL_PATH.exists():
        print(f'  ❌ 模型不存在: {MODEL_PATH}')
        return
    state = torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=False)
    print('  ✅ fold3_best.pth 加载完成')

    # ── 加载数据 ──
    print('[1] 加载数据取样 (每类 ≤200 条, 控点)...')
    files, labels = load_file_list(DATA_DIR, max_files_per_class=200)

    # ── 管线处理 + 嵌入提取 ──
    print('[2] 管线处理 + 嵌入提取...')
    t0 = time.time()
    all_embeds = []
    all_labels_list = []
    n_processed = 0
    n_empty = 0

    for i, fp in enumerate(files):
        mels = process_audio(fp)
        if not mels:
            n_empty += 1
            continue
        for mel in mels:
            mel_t = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                embed = model(mel_t).cpu().numpy().squeeze()  # (256,)
            all_embeds.append(embed)
            all_labels_list.append(labels[i])
            n_processed += 1
        if (i + 1) % 200 == 0:
            et = time.time() - t0
            print(f'  {i+1}/{len(files)}  files_ok={n_processed} segs={len(all_embeds)} empty={n_empty} {et:.0f}s')

    if len(all_embeds) == 0:
        print('  ❌ 无有效嵌入!')
        return

    embeds = np.stack(all_embeds, axis=0)
    le = LabelEncoder()
    label_idx = le.fit_transform(all_labels_list)
    print(f'\n  完成: {len(embeds)} 段嵌入, {et:.0f}s')
    print(f'  形状: {embeds.shape}')
    unique_labels, counts = np.unique(all_labels_list, return_counts=True)
    print(f'  类别分布: {dict(zip(unique_labels, counts))}')

    # ── t-SNE ──
    print('\n[3] t-SNE 降维 (n_components=2, perplexity=30)...')
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=min(30, len(embeds) // 3),
                random_state=42, max_iter=1000, verbose=1)
    embed_2d = tsne.fit_transform(embeds)
    print(f'  t-SNE 完成: {embed_2d.shape}')

    # ── 绘图 ──
    print('\n[4] 绘图...')
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    fig, ax = plt.subplots(figsize=(14, 10))

    for i, label in enumerate(unique_labels):
        mask = label_idx == i
        ax.scatter(embed_2d[mask, 0], embed_2d[mask, 1],
                   c=[colors[i]], label=label, alpha=0.6, s=30, edgecolors='none')

    ax.set_title(f't-SNE: FDAP ch64 嵌入空间 (fold3 best)\n'
                 f'{len(embeds)} 段, 10类', fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / 'tsne_vadexp_fold3.png'
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    print(f'  ✅ 保存: {out_path}')
    plt.close()

    # ── 额外: 可视化每个类在嵌入空间中的散点密度区别 ──
    # 按类绘制单独的散点图, 方便看类内弥散
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for i, label in enumerate(unique_labels):
        ax = axes[i]
        mask = label_idx == i
        # 该类点
        ax.scatter(embed_2d[mask, 0], embed_2d[mask, 1],
                   c=[colors[i]], label=label, alpha=0.6, s=8)
        # 其他类淡灰色
        other = ~mask
        ax.scatter(embed_2d[other, 0], embed_2d[other, 1],
                   c='gray', alpha=0.1, s=4)
        ax.set_title(f'{label}\n({mask.sum()} 段)', fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    out_path2 = OUTPUT_DIR / 'tsne_vadexp_per_class.png'
    fig.savefig(str(out_path2), dpi=150)
    print(f'  ✅ 逐类图: {out_path2}')
    plt.close()

    print(f'\n{"=" * 50}')
    print('  t-SNE 完成! 输出:')
    print(f'    {OUTPUT_DIR / "tsne_vadexp_fold3.png"}')
    print(f'    {OUTPUT_DIR / "tsne_vadexp_per_class.png"}')
    print(f'{"=" * 50}')


if __name__ == '__main__':
    main()

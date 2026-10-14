"""
train_binary_gateway.py — 猫/人二分类门卫 CNN 训练脚本

猫正样本: CatSound_DataSet V2 (5925条, 全部标记为1)
人负样本: MagicData-RAMC (180h中文对话, 抽~6000段, 标记为0)

预处理管线完全对齐 `音频数据预处理流程.txt`:
  降噪 → VAD 切分 → 峰值归一化 → Mel 80帧 [0,1] → 标签

模型: MiniCatBinary ~20K 参数, 输出 2 类 (猫/非猫)
"""

import os
import sys
import json
import random
import hashlib
import argparse

# Windows GBK 控制台无法输出 emoji，强制 UTF-8（重定向/管道场景也适用）
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ─── 配置 ───────────────────────────────────────────────────────────

CATSOUND_ROOT = r"/path/to/dataset\CatSound_DataSet V2\NAYA_DATA_AUG1X"
# 负样本多源：每个源独立扫描，带采样模式
#   mode="file"   → 文件级 shuffle，每文件最多 quota 段（适合长对话文件）
#   mode="prefix" → 前缀轮转采样，每前缀内部 shuffle 后轮流取文件，每文件最多 quota 段（适合 THCHS-30 按说话人前缀）
#   元组: (root, mode, quota, max_secs, max_segs)
#     max_secs=0  → 不截断；>0 → 每文件只取前 max_secs 秒（长对话提速）
#     max_segs=0  → 不限制；>0 → 每文件最多取前 max_segs 条 VAD 段
HUMAN_SOURCES = [
    (r"E:\MagicData-RAMC\MDT2021S003\WAV", "file", 15, 60, 5),   # 663人 男女均衡 手机对话（前60秒+前5段，28min长文件提速）
    (r"E:\data_thchs30\data_thchs30\data", "prefix", 2, 0, 0),    # 60前缀 女声朗读（前缀轮转保证全覆盖，每文件2段）
]
CACHE_DIR = r"/path/to/user\AppData\Local\Temp\seg_cache_binary"

N_CAT = 99999       # 全量正样本（目录扫描）
N_HUMAN = 9000      # 抽取负样本数（平衡猫正样本 8983 段）

# 管线参数（对齐 `音频数据预处理流程.txt`）
SAMPLE_RATE = 16000
N_FFT = 2048          # 帧长（RMS 计算）
HOP_LENGTH = 256      # 帧步进
MIN_SEG_DURATION = 0.25
MAX_SEG_DURATION = 30.0

# Mel 参数
MEL_N_FFT = 512
MEL_HOP_LENGTH = 256
MEL_N_MELS = 128
MEL_FMIN = 50
MEL_FMAX = 8000
MEL_N_FRAMES = 80

# 降噪
PROP_DECREASE = 0.8
NOISE_BOTTOM_PCT = 0.15     # 起始底部比例 15%
NOISE_MIN_SAMPLES = 2048    # 最小噪声采样点

# 安静值
QUIET_BOTTOM_PCT = 0.40     # 降噪后底部 40%
ONSET_MULT = 3.5
OFFSET_MULT = 3

# 缓冲
BUFFER_SEC = 0.075          # 单侧缓冲

# 训练参数
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 20
N_FOLDS = 5
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[设备] {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


# ─── MiniCatBinary 模型 ─────────────────────────────────────────────

class MiniCatBinary(nn.Module):
    """
    猫/人二分类 MiniCatCNN
    Input: (B, 1, 128, 80) — Mel 频谱
    Output: (B, 2) — [非猫, 猫]
    Params: ~20K
    """
    def __init__(self, in_channels=1, n_mels=128, n_frames=80, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 64×40
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 32×20
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 16×10
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def print_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  [MiniCatBinary] 总参数: {total:,} / 可训练: {trainable:,}")


# ─── 预处理管线 ─────────────────────────────────────────────────────

# ---------- 步骤4/9: Mel 计算（对齐管线步骤9）----------

def compute_mel_from_audio(y: np.ndarray) -> np.ndarray:
    """
    峰值归一化 → Mel 频谱 → dB → [0,1] → 80帧
    返回 (128, 80) float32
    """
    import librosa
    # 峰值归一化（管线第9步）
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    # Mel 频谱（n_fft=512, 50-8000Hz）
    S = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=MEL_N_FFT,
        hop_length=MEL_HOP_LENGTH, n_mels=MEL_N_MELS,
        fmin=MEL_FMIN, fmax=MEL_FMAX
    )
    # 转 dB
    S_db = librosa.power_to_db(S, ref=np.max)
    # 缩放到 [0, 1]
    S_min, S_max = S_db.min(), S_db.max()
    if S_max > S_min:
        S_db = (S_db - S_min) / (S_max - S_min)
    else:
        S_db = np.zeros_like(S_db)

    # 固定到 80 帧
    if S_db.shape[1] < MEL_N_FRAMES:
        pad_width = MEL_N_FRAMES - S_db.shape[1]
        S_db = np.pad(S_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    elif S_db.shape[1] > MEL_N_FRAMES:
        start = (S_db.shape[1] - MEL_N_FRAMES) // 2
        S_db = S_db[:, start:start + MEL_N_FRAMES]

    return S_db.astype(np.float32)


# ---------- 步骤1-8: VAD 切分（对齐管线步骤1-8）----------

def _compute_rms(y: np.ndarray) -> np.ndarray:
    """逐帧 RMS 能量（步骤1/4）"""
    import librosa
    return librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]


def _collect_noise(y: np.ndarray, rms: np.ndarray) -> np.ndarray:
    """
    步骤2: 收集底部 15% 帧→回退到 40%，拼接为 y_noise
    返回 y_noise 数组
    """
    bottom_pct = NOISE_BOTTOM_PCT
    while bottom_pct <= 0.40:
        n_noise_frames = max(1, int(len(rms) * bottom_pct))
        sorted_indices = np.argsort(rms)
        noise_idx = sorted_indices[:n_noise_frames]
        noise_idx.sort()  # 保持时间序
        # 拼接对应帧的音频
        noise_pieces = []
        for idx in noise_idx:
            start = idx * HOP_LENGTH
            end = min(len(y), start + HOP_LENGTH)
            noise_pieces.append(y[start:end])
        y_noise = np.concatenate(noise_pieces)
        if len(y_noise) >= NOISE_MIN_SAMPLES:
            return y_noise
        bottom_pct += 0.05
    return y_noise


def _apply_denoise(y: np.ndarray, rms: np.ndarray) -> np.ndarray:
    """
    步骤2→3: 收集噪声 → 降噪
    返回降噪后音频
    """
    y_noise = _collect_noise(y, rms)
    if len(y_noise) < NOISE_MIN_SAMPLES:
        return y  # 跳过降噪
    try:
        import noisereduce as nr
        min_rms = float(np.min(rms[rms > 0])) if np.any(rms > 0) else 0.0
        stationary = min_rms < 0.03
        return nr.reduce_noise(
            y=y, sr=SAMPLE_RATE, y_noise=y_noise,
            stationary=stationary, prop_decrease=PROP_DECREASE
        )
    except ImportError:
        return y
    except Exception as e:
        print(f"  [降噪] 跳过: {e}")
        return y


def _vad_detect(rms: np.ndarray, hop: int, sr: int):
    """
    步骤5: 安静值(bottom 40%) → Onset/Offset 检测
    返回 [(onset_frame, offset_frame), ...]
    """
    sorted_rms = np.sort(rms)
    quiet_val = np.median(sorted_rms[:max(1, int(len(sorted_rms) * QUIET_BOTTOM_PCT))])

    onset_thresh = quiet_val * ONSET_MULT
    offset_thresh = quiet_val * OFFSET_MULT

    segments = []
    in_call = False
    start_frame = 0

    for i in range(len(rms)):
        if not in_call:
            if rms[i] > onset_thresh:
                start_frame = i
                in_call = True
        else:
            if rms[i] < offset_thresh:
                segments.append((start_frame, i))
                in_call = False

    # 扫描结束时仍在叫声状态 → 补偿最后帧为 offset
    if in_call:
        segments.append((start_frame, len(rms) - 1))

    return segments, quiet_val, onset_thresh, offset_thresh


def _filter_segments(segments: list, rms: np.ndarray, hop: int, sr: int):
    """
    步骤6: 有效段判断
    - 时长 < 0.25s → 丢弃
    - 段内最大 RMS < 0.01 → 丢弃
    """
    valid = []
    for onset_f, offset_f in segments:
        dur = (offset_f - onset_f) * hop / sr
        if dur < MIN_SEG_DURATION:
            continue
        seg_rms = rms[onset_f:offset_f + 1]
        if np.max(seg_rms) < 0.01:
            continue
        valid.append((onset_f, offset_f, dur))
    return valid


def _add_buffer(segments: list, total_frames: int, hop: int, sr: int):
    """
    步骤7: 0.075s 缓冲，边界夹到音频起止
    """
    buf_frames = int(BUFFER_SEC * sr / hop)
    buffered = []
    for onset_f, offset_f, dur in segments:
        s = max(0, onset_f - buf_frames)
        e = min(total_frames - 1, offset_f + buf_frames)
        buffered.append((s, e))
    return buffered


def _resolve_overlaps(segments: list, rms: np.ndarray):
    """
    步骤8: 重叠段处理 — 重叠区找 RMS 最低帧切分
    """
    if len(segments) <= 1:
        return segments
    resolved = [segments[0]]
    for i in range(1, len(segments)):
        prev_s, prev_e = resolved[-1]
        cur_s, cur_e = segments[i]
        if cur_s < prev_e:
            # 重叠，找 RMS 最低帧在 overlap 区
            overlap_start = cur_s
            overlap_end = min(prev_e, cur_e)
            overlap_rms = rms[overlap_start:overlap_end + 1]
            split_idx = overlap_start + np.argmin(overlap_rms)
            # 切分：前一段到 split_idx，后一段从 split_idx+1 起
            resolved[-1] = (prev_s, split_idx)
            segments[i] = (split_idx + 1, cur_e)
        resolved.append(segments[i])
    return resolved


def vad_segment(wav_path: str, skip_denoise: bool = False, max_secs: float = 0) -> list:
    """
    完整 VAD 切分管线（步骤1-8）
    返回 [(audio_segment_ndarray, start_sec, end_sec), ...]
    skip_denoise=True 时跳过降噪步骤2-3，直接进入 VAD
    max_secs>0 时只取音频前 max_secs 秒做 VAD（长对话提速，适合负样本源）
    """
    import librosa
    try:
        y, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"  [VAD] 加载失败 {wav_path}: {e}")
        return []

    # 长音频截断：只保留前 max_secs 秒（加速预处理）
    if max_secs > 0:
        cut = int(max_secs * SAMPLE_RATE)
        if len(y) > cut:
            y = y[:cut]

    # 步骤1: 逐帧 RMS（第一次）
    rms = _compute_rms(y)

    # 步骤2-3: 降噪（人声跳过）
    if skip_denoise:
        y_clean = y
    else:
        y_clean = _apply_denoise(y, rms)

    # 步骤4: 逐帧 RMS（第二次）
    rms_clean = _compute_rms(y_clean)

    # 步骤5: VAD 检测
    raw_segments, quiet_val, onset_th, offset_th = _vad_detect(rms_clean, HOP_LENGTH, SAMPLE_RATE)

    # 步骤6: 有效段过滤
    valid_segments = _filter_segments(raw_segments, rms_clean, HOP_LENGTH, SAMPLE_RATE)

    if not valid_segments:
        return []

    # 步骤7: 缓冲
    buffered_segments = _add_buffer(valid_segments, len(rms_clean), HOP_LENGTH, SAMPLE_RATE)

    # 步骤8: 重叠处理
    final_segments = _resolve_overlaps(buffered_segments, rms_clean)

    # 转成音频数组
    result = []
    for s, e in final_segments:
        start_sample = int(s * HOP_LENGTH)
        end_sample = min(len(y_clean), int((e + 1) * HOP_LENGTH))
        seg_y = y_clean[start_sample:end_sample]
        start_sec = start_sample / SAMPLE_RATE
        end_sec = end_sample / SAMPLE_RATE
        result.append((seg_y, start_sec, end_sec))

    return result


# ─── 数据集 ─────────────────────────────────────────────────────────

class BinaryMelDataset(Dataset):
    def __init__(self, mel_files, labels):
        print(f"  [加载] {len(mel_files)} 段到内存...", end=" ", flush=True)
        self.mels = []
        for f in mel_files:
            mel = np.load(f)
            self.mels.append(torch.from_numpy(mel).unsqueeze(0))
        self.labels = labels
        print("完成")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.mels[idx], torch.tensor(self.labels[idx], dtype=torch.long)


def preprocess_and_cache(root_dir, cache_dir, label=1, max_files=None):
    """
    扫描目录下所有 MP3/WAV 做 VAD → Mel → 缓存
    """
    os.makedirs(cache_dir, exist_ok=True)
    audio_files = sorted(Path(root_dir).rglob("*.mp3"))
    if not audio_files:
        audio_files = sorted(Path(root_dir).rglob("*.wav"))
    print(f"  [正样本] 找到 {len(audio_files)} 个音频文件，全部标为猫")

    if max_files and max_files < len(audio_files):
        random.seed(SEED)
        audio_files = random.sample(audio_files, max_files)

    mel_files = []
    labels = []
    cat_ids = []
    reused = 0

    for idx, wav_path in enumerate(audio_files):
        wav_path = str(wav_path)
        parent_dir = os.path.basename(os.path.dirname(wav_path))
        fname = os.path.splitext(os.path.basename(wav_path))[0]
        cat_id_part = fname.replace('_aug1(1)', '').replace('_aug1(2)', '')
        cat_id = f"{parent_dir}_{'_'.join(cat_id_part.split('_')[:3])}"

        segments = vad_segment(wav_path)
        for seg_y, start_sec, end_sec in segments:
            rel = os.path.relpath(wav_path, root_dir)
            cache_key = hashlib.md5(f"cat_{rel}_{start_sec:.3f}".encode()).hexdigest()
            mel_path = os.path.join(cache_dir, f"{cache_key}.npy")
            meta_path = mel_path.replace('.npy', '_meta.json')
            if os.path.exists(mel_path) and os.path.exists(meta_path):
                # 缓存命中：跳过 VAD/Mel 重算（rel 与 start 一致 → 内容一致）
                reused += 1
                mel_files.append(mel_path)
                labels.append(label)
                cat_ids.append(cat_id)
                continue
            mel = compute_mel_from_audio(seg_y)
            np.save(mel_path, mel)

            meta = {
                'source': rel, 'start': start_sec, 'end': end_sec,
                'label': label, 'cat_id': cat_id,
            }
            with open(os.path.join(cache_dir, f"{cache_key}_meta.json"), 'w') as f:
                json.dump(meta, f)

            mel_files.append(mel_path)
            labels.append(label)
            cat_ids.append(cat_id)

        if (idx + 1) % 500 == 0:
            print(f"  [正样本] 已处理 {idx + 1}/{len(audio_files)} 条")

    print(f"  [正样本] 完成：{len(mel_files)} 段（缓存命中 {reused}）")
    return mel_files, labels, cat_ids


def build_prefix_round_robin(wav_files):
    """THCHS-30 前缀轮转：按文件名前3字符分组，每前缀内部 shuffle，
    然后轮转产出文件（第 i 轮取每前缀第 i 个文件，耗尽的前缀跳过）。
    返回生成器，产出 (wav_path, prefix)。
    """
    groups = defaultdict(list)
    for w in wav_files:
        prefix = os.path.basename(str(w))[:3]
        groups[prefix].append(w)
    for g in groups.values():
        random.shuffle(g)

    max_len = max(len(g) for g in groups.values())
    for i in range(max_len):
        for prefix in sorted(groups.keys()):
            if i < len(groups[prefix]):
                yield groups[prefix][i], prefix


def preprocess_human_roots(human_sources, cache_dir, n_target=9000, label=0):
    """多源人声负样本 → VAD → Mel → 缓存

    human_sources: [(root_dir, mode, quota, max_secs, max_segs)]
      mode="file"   → 文件级 shuffle，每文件最多 quota 段
      mode="prefix" → 前缀轮转采样，每文件最多 quota 段
      max_secs>0    → 每文件只取前 max_secs 秒做 VAD（长对话提速）
      max_segs>0    → 每文件最多取前 max_segs 条 VAD 段
    跨源轮询取文件，直到集满 n_target 或所有源扫完。
    """
    import librosa
    os.makedirs(cache_dir, exist_ok=True)

    # 每个源准备一个文件迭代器
    source_iters = []  # [(root_dir, mode, quota, max_secs, max_segs, iterator)]
    for root_dir, mode, quota, max_secs, max_segs in human_sources:
        wav_files = sorted(Path(root_dir).rglob("*.wav"))
        extra = f"max_secs={max_secs}" if max_secs > 0 else "full"
        extra += f", max_segs={max_segs}" if max_segs > 0 else ""
        print(f"  [负样本源] {root_dir}: {len(wav_files)} 个 WAV (mode={mode}, quota={quota}段/文件, {extra})")
        if not wav_files:
            continue
        random.seed(SEED)
        if mode == "prefix":
            it = build_prefix_round_robin(wav_files)
        else:
            random.shuffle(wav_files)
            it = ((w, '') for w in wav_files)
        source_iters.append((root_dir, mode, quota, max_secs, max_segs, it))

    if not source_iters:
        print("  [⚠️] 没有找到任何负样本源")
        return [], []

    mel_files = []
    labels = []
    total = 0

    # 跨源轮询：每轮每个源取 1 个文件，保证多源均匀（而不是先扫完一个源）
    while total < n_target:
        progressed = False
        for root_dir, mode, quota, max_secs, max_segs, it in source_iters:
            if total >= n_target:
                break
            try:
                wav_path, prefix = next(it)
            except StopIteration:
                continue
            progressed = True

            segments = vad_segment(str(wav_path), max_secs=max_secs)   # 与猫一致，过降噪
            if not segments:
                continue

            # 每文件最多取前 max_segs 条（保持对话开头双方都说话）
            if max_segs > 0:
                segments = segments[:max_segs]

            file_count = 0
            for seg_y, start_sec, end_sec in segments:
                if total >= n_target:
                    break
                if file_count >= quota:
                    break  # 该文件已到配额

                mel = compute_mel_from_audio(seg_y)
                rel = str(wav_path.relative_to(root_dir))
                cache_key = hashlib.md5(f"human_{rel}_{start_sec:.3f}".encode()).hexdigest()
                mel_path = os.path.join(cache_dir, f"{cache_key}.npy")
                meta_path = mel_path.replace('.npy', '_meta.json')
                if os.path.exists(mel_path) and os.path.exists(meta_path):
                    # 缓存命中：跳过 VAD/Mel 重算
                    mel_files.append(mel_path)
                    labels.append(label)
                    total += 1
                    file_count += 1
                    continue

                np.save(mel_path, mel)
                meta = {
                    'source': rel, 'start': start_sec, 'end': end_sec, 'label': label,
                }
                with open(meta_path, 'w') as f:
                    json.dump(meta, f)

                mel_files.append(mel_path)
                labels.append(label)
                total += 1
                file_count += 1

            if (total % 500 == 0 or total >= n_target) and total > 0:
                print(f"  [负样本] 已有 {total}/{n_target} 段")

        if not progressed:
            break  # 所有源都扫完了

    if total < n_target:
        print(f"  [⚠️] 负样本仅 {total} 段，不足 {n_target}")
    return mel_files, labels


# ─── 训练 ───────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for mel, label in loader:
        mel, label = mel.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(mel)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * mel.size(0)
        preds = output.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for mel, label in loader:
            mel, label = mel.to(device), label.to(device)
            output = model(mel)
            loss = criterion(output, label)
            total_loss += loss.item() * mel.size(0)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, acc, prec, rec, f1, all_preds, all_labels


def run_cv(all_mel_files, all_labels, all_cat_ids=None):
    n_total = len(all_mel_files)
    all_indices = list(range(n_total))
    cat_indices = [i for i in all_indices if all_labels[i] == 1]
    human_indices = [i for i in all_indices if all_labels[i] == 0]
    print(f"  总样本: {n_total} (猫: {len(cat_indices)}, 人: {len(human_indices)})")

    cat_id_to_indices = defaultdict(list)
    for i in cat_indices:
        meta_path = all_mel_files[i].replace('.npy', '_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            cid = meta.get('cat_id', 'unknown')
        else:
            cid = 'unknown'
        cat_id_to_indices[cid].append(i)

    cat_id_list = list(cat_id_to_indices.keys())
    print(f"  猫: {len(cat_indices)} 段, {len(cat_id_list)} 只猫")

    random.seed(SEED)
    random.shuffle(human_indices)
    human_folds = np.array_split(human_indices, N_FOLDS)

    fold_results = []

    for fold in range(N_FOLDS):
        print(f"\n═══ Fold {fold + 1}/{N_FOLDS} ═══")
        cat_id_split = np.array_split(np.array(cat_id_list), N_FOLDS)
        val_cat_ids = set(cat_id_split[fold])
        train_cat_ids = set(cat_id_list) - val_cat_ids

        train_cat_indices = []
        val_cat_indices = []
        for cid, idxs in cat_id_to_indices.items():
            if cid in val_cat_ids:
                val_cat_indices.extend(idxs)
            else:
                train_cat_indices.extend(idxs)

        val_human_indices = list(human_folds[fold])
        train_human_indices = []
        for f in range(N_FOLDS):
            if f != fold:
                train_human_indices.extend(list(human_folds[f]))

        train_idx = train_cat_indices + train_human_indices
        val_idx = val_cat_indices + val_human_indices
        print(f"  Train: {len(train_cat_indices)} 猫 + {len(train_human_indices)} 人 = {len(train_idx)}")
        print(f"  Val:   {len(val_cat_indices)} 猫 + {len(val_human_indices)} 人 = {len(val_idx)}")

        train_dataset = BinaryMelDataset(
            [all_mel_files[i] for i in train_idx],
            [all_labels[i] for i in train_idx])
        val_dataset = BinaryMelDataset(
            [all_mel_files[i] for i in val_idx],
            [all_labels[i] for i in val_idx])

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = MiniCatBinary().to(device)
        model.print_params()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_val_f1 = 0.0
        best_state = None

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc, val_prec, val_rec, val_f1, preds, gt = eval_epoch(model, val_loader, criterion)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:2d}/{EPOCHS} | "
                      f"Train L:{train_loss:.4f} A:{train_acc:.4f} | "
                      f"Val L:{val_loss:.4f} A:{val_acc:.4f} P:{val_prec:.4f} R:{val_rec:.4f} F1:{val_f1:.4f}")

        model.load_state_dict(best_state)
        _, val_acc, val_prec, val_rec, val_f1, preds, gt = eval_epoch(model, val_loader, criterion)
        cm = confusion_matrix(gt, preds)
        print(f"  → Fold {fold+1} | A:{val_acc:.4f} P:{val_prec:.4f} R:{val_rec:.4f} F1:{val_f1:.4f}")
        print(f"    CM: [[{cm[0,0]:4d} {cm[0,1]:4d}] [{cm[1,0]:4d} {cm[1,1]:4d}]]")

        fold_results.append({
            'fold': fold + 1, 'acc': val_acc, 'precision': val_prec,
            'recall': val_rec, 'f1': val_f1, 'confusion_matrix': cm.tolist(),
        })

    print("\n" + "=" * 50)
    print("5折交叉验证汇总")
    print("=" * 50)
    avg_f1 = np.mean([r['f1'] for r in fold_results])
    avg_acc = np.mean([r['acc'] for r in fold_results])
    avg_prec = np.mean([r['precision'] for r in fold_results])
    avg_rec = np.mean([r['recall'] for r in fold_results])
    print(f"  Acc:   {avg_acc:.4f} (±{np.std([r['acc'] for r in fold_results]):.4f})")
    print(f"  Prec:  {avg_prec:.4f} (±{np.std([r['precision'] for r in fold_results]):.4f})")
    print(f"  Rec:   {avg_rec:.4f} (±{np.std([r['recall'] for r in fold_results]):.4f})")
    print(f"  F1:    {avg_f1:.4f} (±{np.std([r['f1'] for r in fold_results]):.4f})")
    for r in fold_results:
        print(f"  Fold {r['fold']}: Acc={r['acc']:.4f} F1={r['f1']:.4f}")
    return fold_results, best_state


# ─── 入口 ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="训练猫/人二分类门卫 CNN")
    parser.add_argument('--preprocess', action='store_true', help="只做预处理和缓存")
    parser.add_argument('--skip-preprocess', action='store_true', help="跳过预处理，加载已有缓存")
    parser.add_argument('--human-target', type=int, default=N_HUMAN)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    args = parser.parse_args()

    print("=" * 60)
    print("猫/人二分类门卫 CNN 训练")
    print(f"  正样本: CatSound_DataSet V2 ({CATSOUND_ROOT})")
    for r, mode, q, ms, mseg in HUMAN_SOURCES:
        print(f"  负样本: {r}  (mode={mode}, {q}段/文件)")
    print(f"  缓存:   {CACHE_DIR}")
    print(f"  管线:   对齐 `音频数据预处理流程.txt`")
    print("=" * 60)

    if not args.skip_preprocess:
        print("\n[预处理] 正样本...")
        cat_mel_files, cat_labels, cat_ids = preprocess_and_cache(
            CATSOUND_ROOT, os.path.join(CACHE_DIR, 'cat'),
            label=1, max_files=N_CAT)
        print(f"  ✅ 猫正样本: {len(cat_mel_files)} 段")

        print("\n[预处理] 负样本（多源）...")
        human_mel_files, human_labels = preprocess_human_roots(
            HUMAN_SOURCES, os.path.join(CACHE_DIR, 'human'),
            n_target=args.human_target, label=0)
        print(f"  ✅ 人负样本: {len(human_mel_files)} 段")

        all_mel_files = cat_mel_files + human_mel_files
        all_labels = cat_labels + human_labels
        all_cat_ids = cat_ids + ['human'] * len(human_labels)

        import pickle as pkl
        with open(os.path.join(CACHE_DIR, 'dataset_index.pkl'), 'wb') as f:
            pkl.dump({'mel_files': all_mel_files, 'labels': all_labels, 'cat_ids': all_cat_ids}, f)
        print(f"\n[数据] 总样本: {len(all_mel_files)} 段")

        if args.preprocess:
            print("\n预处理完成，退出。")
            return
    else:
        print("\n[加载] 从缓存加载...")
        import pickle as pkl
        with open(os.path.join(CACHE_DIR, 'dataset_index.pkl'), 'rb') as f:
            index_data = pkl.load(f)
        all_mel_files = index_data['mel_files']
        all_labels = index_data['labels']
        all_cat_ids = index_data['cat_ids']
        print(f"  ✅ 加载 {len(all_mel_files)} 段")

    print("\n[训练] 5 折交叉验证...")
    fold_results, best_state = run_cv(all_mel_files, all_labels, all_cat_ids)

    print("\n[保存] 最终模型...")
    full_dataset = BinaryMelDataset(all_mel_files, all_labels)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    final_model = MiniCatBinary().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=LR)
    for epoch in range(args.epochs):
        loss, acc = train_epoch(final_model, full_loader, optimizer, criterion)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{args.epochs} | Loss: {loss:.4f} Acc: {acc:.4f}")

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(final_model.state_dict(), os.path.join(save_dir, 'minicat_binary.pt'))
    print(f"  ✅ 模型保存至: {save_dir}\\minicat_binary.pt")

    try:
        final_model.eval()
        dummy = torch.randn(1, 1, 128, 80).to(device)
        torch.onnx.export(final_model, dummy, os.path.join(save_dir, 'minicat_binary.onnx'),
                          input_names=['mel'], output_names=['scores'],
                          opset_version=11,
                          dynamic_axes={'mel': {0: 'batch_size'}, 'scores': {0: 'batch_size'}})
        print(f"  ✅ ONNX 保存至: {save_dir}\\minicat_binary.onnx")
    except Exception as e:
        print(f"  ⚠️ ONNX 导出失败: {e}")

    print("\n✅ 完成！")


if __name__ == '__main__':
    main()

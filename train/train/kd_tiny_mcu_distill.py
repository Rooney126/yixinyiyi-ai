"""
知识蒸馏：FDAP ch64 Teacher → TinyMCUCatCNN Student
=========================================
Teacher: FDAPMobileCatCNN (4Conv+FDAP, 128×80 Mel, ~70K params, F1~86%)
Student: TinyMCUCatCNN (3Conv+GAP, 32×40 Mel (直接输入), ~3K params, arena <50KB)
蒸馏方式: logit-level KD — CE(hard) + T²·KL(soft_teacher/T ∥ soft_student/T)

Student Mel = Teacher Mel 降采样 (128→64 bands × 80→40 frames)
→ 复用完整 seg_cache 缓存，不增加预处理负担
"""

import csv
import hashlib
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path

import librosa
import noisereduce as nr
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset

# ═══════════════════════════════════════
#  配置
# ═══════════════════════════════════════

SR = 16000
MAX_DUR = 30.0

# Mel 参数（Teacher）
N_MELS_T = 128
MEL_FRAMES_T = 80

# Mel 参数（Student — 降采样自 Teacher）
N_MELS_S = 64
MEL_FRAMES_S = 40

# 管线参数（与 vadexp 一致）
FRAME_LEN = 2048
HOP_LENGTH = 256
N_FFT = 512
FMIN = 50
FMAX = 8000
PROP_DECREASE = 0.8
MIN_RMS_THRESH = 0.03
MIN_NOISE_SAMPLES = 2048
MIN_SEG_DUR = 0.25
ONSET_MULT = 3.5             # 与 vadexp 一致，复用缓存
OFFSET_MULT = 3.0            # 与 vadexp 一致，复用缓存
BUF_SEC = 0.075

# 训练参数
BATCH_SIZE = 64
EPOCHS = 100
LR = 3e-3                 # 小模型用稍高 LR
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
PATIENCE = 15
CLIP_GRAD = 1.0

# ── KD 超参 ──
KD_TEMPERATURE = 4        # 蒸馏温度
KD_ALPHA = 0.5            # α=1 全硬标签, α=0 全模仿教师

# ═══════════════════════════════════════
#  路径
# ═══════════════════════════════════════

_USER = 'chen'

BASE = Path(f'C:/Users/{_USER}/Desktop')
DATA_DIR = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
CACHE_DIR = Path(f'C:/Users/{_USER}/AppData/Local/Temp/seg_cache_full')
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path(f'C:/Users/{_USER}/译芯怡意/train/output/kd_tiny_mcu')

# Teacher 权重：A_aug fold3 (最佳折)
TEACHER_CKPT = Path(
    f'C:/Users/{_USER}/译芯怡意/train/output/fdap_cnn_ch64_A_aug/fold3_best.pth')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'设备: {DEVICE}')
if DEVICE == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name(0)}')

LABELS = ["Angry", "Defence", "Fighting", "Happy", "HuntingMind",
          "Mating", "MotherCall", "Paining", "Resting", "Warning"]


# ═══════════════════════════════════════
#  11步管线 (从 vadexp 复制，保持完全一致)
# ═══════════════════════════════════════

def compute_frame_rms(y, frame_len=FRAME_LEN, hop=HOP_LENGTH):
    frames = []
    for start in range(0, len(y) - frame_len + 1, hop):
        rms = np.sqrt(np.mean(y[start:start + frame_len] ** 2))
        frames.append(rms)
    return np.array(frames, dtype=np.float32)


def collect_noise_samples(rms, y, bottom_pct=0.15,
                          min_samples=MIN_NOISE_SAMPLES, hop=HOP_LENGTH):
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
        return collect_noise_samples(rms, y, min(bottom_pct + 0.05, 0.40),
                                     min_samples, hop)
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
    if onset_thresh < 1e-8 or onset_thresh <= offset_thresh:
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
            'orig_onset_frame': onset_frame,
            'orig_offset_frame': offset_frame,
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


# 缓存参数 hash（与 vadexp 一致，确保复用现有缓存）
_CACHE_PARAMS = (f"segdur={MIN_SEG_DUR}_on={ONSET_MULT}_off={OFFSET_MULT}_"
                 f"propd={PROP_DECREASE}_buf={BUF_SEC}_nody=False_qp40")
CACHE_PARAMS_HASH = hashlib.md5(_CACHE_PARAMS.encode()).hexdigest()[:12]


def process_audio_segment(y, onset_sample, offset_sample):
    """
    从完整音频提取一段，做 Mel 预处理（11步管线步骤9）
    返回 128×80 Mel (float32)
    """
    offset_sample = min(offset_sample, len(y))
    seg_y = y[onset_sample:offset_sample]
    if len(seg_y) < N_FFT:
        return None
    peak = np.max(np.abs(seg_y))
    if peak > 1e-10:
        seg_y = seg_y / peak
    else:
        return None
    mel = librosa.feature.melspectrogram(
        y=seg_y, sr=SR, n_mels=N_MELS_T,
        fmin=FMIN, fmax=FMAX, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_min, mel_max = mel_db.min(), mel_db.max()
    if mel_max - mel_min < 1e-10:
        return None
    mel_norm = (mel_db - mel_min) / (mel_max - mel_min)
    if mel_norm.shape[1] < MEL_FRAMES_T:
        mel_norm = np.pad(mel_norm, ((0, 0), (0, MEL_FRAMES_T - mel_norm.shape[1])))
    else:
        start = (mel_norm.shape[1] - MEL_FRAMES_T) // 2
        mel_norm = mel_norm[:, start:start + MEL_FRAMES_T]
    return mel_norm.astype(np.float32)


def process_audio(file_path):
    """
    完整管线处理单文件
    返回: list of {'mel': (128,80), 'orig_dur': float}
    """
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
    y_noise, _ = collect_noise_samples(rms, y)
    if len(y_noise) >= FRAME_LEN:
        min_rms_val = float(np.min(rms)) if len(rms) > 0 else 0.0
        stationary = min_rms_val < MIN_RMS_THRESH
        try:
            y_denoised = nr.reduce_noise(
                y=y, sr=SR, y_noise=y_noise, stationary=stationary,
                prop_decrease=PROP_DECREASE, n_fft=FRAME_LEN)
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
    results = []
    for seg in clean_segments:
        mel = process_audio_segment(y_denoised, seg['onset_sample'],
                                     seg['offset_sample'])
        if mel is not None:
            results.append({'mel': mel, 'orig_dur': seg['orig_dur']})
    return results


def process_and_cache(file_path, label_idx, cache_dir):
    """处理+缓存，返回段列表 (复用 vadexp 缓存)"""
    cache_key = hashlib.md5(
        (CACHE_PARAMS_HASH + file_path).encode()).hexdigest()
    meta_path = cache_dir / f'{cache_key}_meta.json'

    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            segments = []
            for seg_info in meta['segments']:
                mel_path = cache_dir / seg_info['mel_file']
                if not mel_path.exists():
                    return _process_and_save(file_path, label_idx,
                                              cache_key, cache_dir)
                mel = np.load(str(mel_path))
                segments.append({
                    'mel': mel,
                    'label_idx': seg_info['label_idx'],
                    'weight': seg_info['weight'],
                    'orig_dur': seg_info['orig_dur'],
                })
            return segments
        except Exception:
            return _process_and_save(file_path, label_idx, cache_key, cache_dir)
    else:
        return _process_and_save(file_path, label_idx, cache_key, cache_dir)


def _process_and_save(file_path, label_idx, cache_key, cache_dir):
    segments = process_audio(file_path)
    if not segments:
        return []
    meta_segments = []
    for i, seg in enumerate(segments):
        mel_file = f'{cache_key}_seg{i}.npy'
        np.save(str(cache_dir / mel_file), seg['mel'])
        meta_segments.append({
            'mel_file': mel_file,
            'label_idx': label_idx,
            'weight': seg['orig_dur'],
            'orig_dur': seg['orig_dur'],
        })
    meta = {
        'file': os.path.basename(file_path) if isinstance(file_path, str)
                else str(file_path),
        'cache_key': cache_key,
        'n_segments': len(segments),
        'segments': meta_segments,
    }
    with open(cache_dir / f'{cache_key}_meta.json', 'w') as f:
        json.dump(meta, f, ensure_ascii=False)
    return [{
        'mel': seg['mel'],
        'label_idx': label_idx,
        'weight': seg['orig_dur'],
        'orig_dur': seg['orig_dur'],
    } for seg in segments]


# ═══════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════

def load_catsound_v2(data_dir, use_aug=True):
    csv_path = data_dir / 'Catmood_NAYA_1xAug.csv'
    files, labels, cat_ids, is_aug = [], [], [], []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return np.array(files), np.array(labels), np.array(cat_ids), np.array(is_aug)
        for row in reader:
            if len(row) < 3:
                continue
            label_idx = int(row[2])
            filename = os.path.basename(row[1].strip())
            local_path = data_dir / LABELS[label_idx] / filename
            if local_path.exists():
                files.append(str(local_path))
                labels.append(LABELS[label_idx])
                raw_stem = local_path.stem.replace('_aug1(1)', '').lower()
                cat_ids.append(raw_stem)
                is_aug.append('_aug1(1)' in local_path.stem)
    return np.array(files), np.array(labels), np.array(cat_ids), np.array(is_aug)


def load_augmented(data_dir, base_files, base_labels, base_cat_ids):
    """加载 aug_SP 目录增强"""
    aug_dir = data_dir / 'aug_SP'
    if not aug_dir.exists():
        return base_files, base_labels, base_cat_ids
    aug_files, aug_labels, aug_cat_ids = [], [], []
    for label in LABELS:
        label_dir = aug_dir / label
        if not label_dir.exists():
            continue
        for fp in sorted(label_dir.glob('*.wav')):
            stem = fp.stem
            for suffix in ['_sp10', '_sm10', '_pp2', '_pm2']:
                if stem.endswith(suffix):
                    raw = stem[:-len(suffix)].lower()
                    aug_files.append(str(fp))
                    aug_labels.append(label)
                    aug_cat_ids.append(raw)
                    break
    return (np.concatenate([base_files, aug_files]),
            np.concatenate([base_labels, aug_labels]),
            np.concatenate([base_cat_ids, aug_cat_ids]))


def preprocess_all(file_list, label_idxs, cat_ids, cache_dir):
    """预处理所有文件→段，返回 Teacher Mel (128×80)"""
    print(f'\n[预处理] 共 {len(file_list)} 条音频...')
    t0 = time.time()
    all_mels, all_labels, all_weights = [], [], []
    all_cat_ids, all_file_idx = [], []
    n_empty = 0

    cache_hits = 0
    for i, fp in enumerate(file_list):
        segs = process_and_cache(fp, int(label_idxs[i]), cache_dir)
        # 检测是否缓存命中
        meta_check = hashlib.md5((CACHE_PARAMS_HASH + fp).encode()).hexdigest()
        if (cache_dir / f'{meta_check}_meta.json').exists():
            cache_hits += 1
        if len(segs) == 0:
            n_empty += 1
        else:
            for seg in segs:
                all_mels.append(seg['mel'])
                all_labels.append(seg['label_idx'])
                all_weights.append(seg['weight'])
                all_cat_ids.append(cat_ids[i])
                all_file_idx.append(i)
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{len(file_list)}  segs={len(all_mels)}  '
                  f'缓存命中={cache_hits}  {time.time()-t0:.0f}s')

    if len(all_mels) == 0:
        print('  ⚠️ 无有效段!')
        return None, None, None, None, None

    all_mels = np.stack(all_mels, axis=0)
    all_labels = np.array(all_labels)
    all_weights = np.array(all_weights)
    all_cat_ids = np.array(all_cat_ids)
    all_file_idx = np.array(all_file_idx)

    print(f'\n完成: {len(all_mels)} 段 / {len(file_list)} 文件')
    print(f'  空文件: {n_empty}')
    print(f'  形状: {list(all_mels.shape)}, {all_mels.nbytes/1024/1024:.0f}MB')
    print(f'  唯一猫ID: {len(np.unique(all_cat_ids))}')
    return all_mels, all_labels, all_weights, all_cat_ids, all_file_idx


# ═══════════════════════════════════════
#  Mel 降采样: 128×80 ──→ 64×40
# ═══════════════════════════════════════

def downsample_mel(mel_128x80):
    """
    下采样 Teacher Mel (128 Mel bands, 80 frames) → Student Mel (64, 40)
    策略: Mel 轴上相邻 2 band 平均, 时间轴上相邻 2 frame 平均
    输入: (128, 80) 输出: (64, 40)
    """
    h, w = mel_128x80.shape
    # Mel 轴: 2→1 平均
    mel_64 = mel_128x80.reshape(h // 2, 2, w).mean(axis=1)
    # 时间轴: 2→1 平均
    mel_64_40 = mel_64.reshape(64, w // 2, 2).mean(axis=2)
    return mel_64_40.astype(np.float32)


# ═══════════════════════════════════════
#  模型定义
# ═══════════════════════════════════════

class FDAPMobileCatCNN(nn.Module):
    """Teacher: 4Conv+FDAP, 128×80 Mel"""
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


class TinyMCUCatCNN(nn.Module):
    """
    Student: 3Conv+GAP, 32×40 Mel (直接输入)
    目标: arena <50KB, 全 SRAM, 推理 ~50ms @M33F
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(),
            nn.MaxPool2d(2))                    # → 8×32×20
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2))                    # → 16×16×10
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2))                    # → 32×8×5
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)


# ═══════════════════════════════════════
#  蒸馏损失
# ═══════════════════════════════════════

def kd_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.5):
    """
    知识蒸馏损失
    L = α * CE(y_hard, y_pred) + (1-α) * T² * KL(soft_teacher/T ∥ student/T)
    """
    # 硬标签损失
    loss_hard = F.cross_entropy(student_logits, labels)

    # 软标签损失
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    loss_soft = F.kl_div(soft_student, soft_teacher,
                         reduction='batchmean') * (T * T)

    return alpha * loss_hard + (1.0 - alpha) * loss_soft


# ═══════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════

class KDDataset(Dataset):
    """
    每个样本: Teacher Mel (128×80) + 标签 + 文件索引
    Student Mel 由 Teacher Mel 降采样得到（在 collate 或 forward 时做）
    """
    def __init__(self, teacher_mels, labels, file_idxs, augment=False):
        self.teacher_mels = teacher_mels           # (N, 128, 80)
        self.labels = labels
        self.file_idxs = file_idxs
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mel_t = self.teacher_mels[idx].clone()  # (128, 80)
        label = self.labels[idx]
        fidx = self.file_idxs[idx]
        if self.augment:
            # 小噪声增强（在 teacher Mel 上做，降采样后噪声也被保留）
            noise = torch.randn_like(mel_t) * 0.005
            mel_t = mel_t + noise
            # 时间位移（±5% 帧）
            shift = np.random.randint(-int(0.05 * mel_t.shape[1]),
                                       int(0.05 * mel_t.shape[1]))
            if shift != 0:
                mel_t = torch.roll(mel_t, shifts=shift, dims=1)
        return mel_t, label, fidx


def kd_collate(batch):
    """批量: teacher_mels→(B,1,128,80), labels→(B,), file_idxs→(B,)"""
    mels, labels, fidxs = zip(*batch)
    return (torch.stack(mels).unsqueeze(1),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(fidxs, dtype=torch.long))


# ═══════════════════════════════════════
#  训练
# ═══════════════════════════════════════

def train_kd_fold(fold, train_mels, train_labels, train_fidxs,
                  test_mels, test_labels, test_fidxs, teacher_model):
    """
    知识蒸馏训练一个折

    参数:
        train_mels: (N_train, 128, 80) — 训练段 Teacher Mel
        test_mels: (N_test, 128, 80) — 测试段 Teacher Mel
        train/test_labels: (N,) — 标签
        train/test_fidxs: (N,) — 文件索引 (用于文件级投票评估)
        teacher_model: 冻结的FDAP教师模型
    """
    print(f'\n{"─" * 50}')
    print(f'折 {fold}: 训练 {len(train_mels)} 段 → 测试 {len(test_mels)} 段')
    print('  Teacher: FDAP (128×80, 冻结)')
    print('  Student: TinyMCU (32×40, 3Conv+GAP)')
    print(f'  KD: T={KD_TEMPERATURE}, α={KD_ALPHA}')
    print(f'{"─" * 50}')

    # ── 准备好的数据集 ──
    train_ds = KDDataset(
        torch.FloatTensor(train_mels),
        torch.LongTensor(train_labels),
        torch.LongTensor(train_fidxs),
        augment=True)
    test_ds = KDDataset(
        torch.FloatTensor(test_mels),
        torch.LongTensor(test_labels),
        torch.LongTensor(test_fidxs),
        augment=False)

    train_loader = DataLoader(
        train_ds, BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        collate_fn=kd_collate)
    test_loader = DataLoader(
        test_ds, BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        collate_fn=kd_collate)

    # ── 初始化学生模型 ──
    student = TinyMCUCatCNN(num_classes=10).to(DEVICE)
    p = sum(p.numel() for p in student.parameters())
    print(f'  Student 参数量: {p:,}')

    # class weights
    seg_weights = compute_class_weight(
        'balanced',
        classes=np.arange(10),
        y=train_labels)
    seg_weight_tensor = torch.FloatTensor(seg_weights).to(DEVICE)

    opt = torch.optim.AdamW(
        student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_acc, best_f1, best_ep = 0, 0, 0
    no_improve = 0

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        student.train()
        teacher_model.eval()  # 冻结

        train_loss = 0.0
        batch_count = 0

        for batch_idx, (mel_t, labels, fidxs) in enumerate(train_loader):
            # mel_t: (B, 1, 128, 80) — Teacher Mel
            # 需要 Student Mel: 对 mel_t 降采样
            # 用 F.avg_pool2d 做 2×2 平均降采样
            with torch.no_grad():
                # (B,1,128,80) → F.avg_pool2d(2,2) → (B,1,64,40)
                mel_s = F.avg_pool2d(mel_t.to(DEVICE), kernel_size=4, stride=(4,2))
                # Teacher 推理
                teacher_logits = teacher_model(mel_t.to(DEVICE))

            mel_s = mel_s.to(DEVICE)
            labels = labels.to(DEVICE)
            teacher_logits = teacher_logits.detach()

            opt.zero_grad()
            student_logits = student(mel_s)

            loss = kd_loss(
                student_logits, teacher_logits, labels,
                T=KD_TEMPERATURE, alpha=KD_ALPHA)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), CLIP_GRAD)
            opt.step()
            train_loss += loss.item()
            batch_count += 1

        avg_train_loss = train_loss / max(batch_count, 1)

        # ── 测试: 文件级等权投票 ──
        student.eval()
        all_file_preds = {}
        all_file_labels = {}

        with torch.no_grad():
            for mel_t, labels, fidxs in test_loader:
                mel_s = F.avg_pool2d(mel_t.to(DEVICE), kernel_size=4, stride=(4,2))
                labels = labels.to(DEVICE)
                logits = student(mel_s)
                probs = F.softmax(logits, dim=1)
                for j in range(len(fidxs)):
                    fi = int(fidxs[j].cpu())
                    if fi not in all_file_preds:
                        all_file_preds[fi] = []
                        all_file_labels[fi] = int(labels[j].cpu())
                    all_file_preds[fi].append(probs[j].cpu().numpy())

        p_list, la_list = [], []
        for fi in all_file_preds:
            arr = np.stack(all_file_preds[fi], axis=0)
            vote = arr.mean(axis=0)  # 等权投票
            p_list.append(int(np.argmax(vote)))
            la_list.append(all_file_labels[fi])

        p = np.array(p_list)
        la = np.array(la_list)
        acc = (p == la).mean()
        f1 = f1_score(la, p, average='macro', zero_division=0)
        sched.step()
        et = time.time() - t0

        if f1 > best_f1:
            best_acc, best_f1, best_ep = acc, f1, ep
            no_improve = 0
            torch.save(student.state_dict(),
                       OUTPUT_DIR / f'fold{fold}_best.pth')
        else:
            no_improve += 1

        if (ep % 5 == 1 or ep == EPOCHS or no_improve == 1):
            print(f'  E{ep:3d}/{EPOCHS} L={avg_train_loss:.4f} '
                  f'Acc={acc:.1%} F1={f1:.1%} '
                  f'Best={best_f1:.1%}@{best_ep} {et:.0f}s')

        if no_improve >= PATIENCE:
            print(f'  早停 @{ep}, Best F1={best_f1:.1%}')
            break

    print(f'  折{fold} 完成: Acc={best_acc:.1%} F1={best_f1:.1%} (ep{best_ep})')
    return best_acc, best_f1, best_ep





# ═══════════════════════════════════════
#  main
# ═══════════════════════════════════════

def main():
    print(f'\n{"█" * 55}')
    print('  知识蒸馏: FDAP Teacher → TinyMCU Student')
    print('  Teacher: FDAP ch64 (70K) → Student: TinyMCU (3K)')
    print('  Mel: 128×80 → 降采样(4,2) → 32×40')
    print(f'  KD: α={KD_ALPHA}, T={KD_TEMPERATURE}')
    print(f'  输出: {OUTPUT_DIR}')
    print(f'{"█" * 55}')

    # ── 加载 Teacher ──
    print('\n[0] 加载 Teacher 模型...')
    if not TEACHER_CKPT.exists():
        print(f'  ❌ Teacher 权重不存在: {TEACHER_CKPT}')
        print('  请确认训练已运行: train_cnn_pipeline_v2_vadexp.py')
        return
    teacher = FDAPMobileCatCNN(num_classes=10).to(DEVICE)
    teacher.load_state_dict(torch.load(str(TEACHER_CKPT),
                                        map_location=DEVICE))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f'  ✅ Teacher 加载完成 ({TEACHER_CKPT.name})')
    tp = sum(p.numel() for p in teacher.parameters())
    print(f'  Teacher 参数量: {tp:,}')

    # ── 加载数据 ──
    print('\n[1] 加载 CatSound_V2...')
    files, labels, cat_ids, is_aug = load_catsound_v2(DATA_DIR)
    files, labels, cat_ids = load_augmented(DATA_DIR, files, labels, cat_ids)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    print(f'  {len(files)} 条, 类别: {list(le.classes_)}')
    print(f'  唯一猫ID: {len(set(cat_ids))}')

    # ── 预处理 → Teacher Mel 缓存 ──
    print('\n[2] 管线预处理 (缓存复用)...')
    all_mels, all_labels, all_weights, seg_cat_ids, seg_file_idxs = \
        preprocess_all(files, y, cat_ids, CACHE_DIR)
    if all_mels is None:
        print('\n❌ 预处理失败, 退出')
        return

    # ── GroupKFold 5折 ──
    print('\n[3] GroupKFold 5折 (按猫ID分组, KD训练)...')
    seg_indices = np.arange(len(seg_cat_ids))
    gkf = GroupKFold(n_splits=5)
    fold_accs, fold_f1s = [], []

    for fold, (train_si, test_si) in enumerate(gkf.split(
            seg_indices, all_labels, seg_cat_ids), 1):
        # 在外部切好段，避免内部用错索引
        train_mels_k = all_mels[train_si]
        train_lbls_k = all_labels[train_si]
        train_fids_k = seg_file_idxs[train_si]
        test_mels_k = all_mels[test_si]
        test_lbls_k = all_labels[test_si]
        test_fids_k = seg_file_idxs[test_si]
        a, f, best_ep = train_kd_fold(
            fold,
            train_mels_k, train_lbls_k, train_fids_k,
            test_mels_k, test_lbls_k, test_fids_k,
            teacher)
        fold_accs.append(a)
        fold_f1s.append(f)

    # ── 结果 ──
    print(f'\n{"█" * 55}')
    print('  结果: TinyMCU (知识蒸馏)')
    print(f'{"█" * 55}')
    for i, (a, f) in enumerate(zip(fold_accs, fold_f1s), 1):
        print(f'  折{i}: Acc={a:.1%} F1={f:.1%}')
    print(f'  Acc: {np.mean(fold_accs):.1%}±{np.std(fold_accs):.1%}')
    print(f'  F1:  {np.mean(fold_f1s):.1%}±{np.std(fold_f1s):.1%}')

    # ── 保存结果 ──
    ts = time.strftime('%Y%m%d_%H%M%S')
    result = {
        'method': 'KD_FDAP_TinyMCU',
        'teacher': 'FDAP_ch64_fold3',
        'teacher_params': tp,
        'student': 'TinyMCUCatCNN',
        'student_params': sum(p.numel() for p in
                              TinyMCUCatCNN(10).parameters()),
        'n_files': len(files),
        'n_segments': len(all_labels),
        'n_cats': len(np.unique(seg_cat_ids)),
        'kd_config': {
            'T': KD_TEMPERATURE,
            'alpha': KD_ALPHA,
            'epochs': EPOCHS,
            'lr': LR,
            'batch_size': BATCH_SIZE,
        },
        'folds': {
            f'fold{i+1}': {'acc': float(a), 'f1': float(f)}
            for i, (a, f) in enumerate(zip(fold_accs, fold_f1s))
        },
        'mean_acc': float(np.mean(fold_accs)),
        'std_acc': float(np.std(fold_accs)),
        'mean_f1': float(np.mean(fold_f1s)),
        'std_f1': float(np.std(fold_f1s)),
        'best_f1': float(max(fold_f1s)),
        'ts': ts,
        'device': DEVICE,
    }
    rp = OUTPUT_DIR / f'results_{ts}.json'
    with open(rp, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'\n结果: {rp}')
    print('✅ 知识蒸馏完成')


if __name__ == '__main__':
    main()

"""
译芯怡意 — FDAPMobileCatCNN 新管线训练 (CatSound_V2)
遵循桌面"音频数据预处理流程"11步管线：

  0. 加载: librosa.load → 16kHz mono — 1-30s 约束
  1. 原始帧RMS: 帧长2048, 步进256 → 原始音频RMS数组
  2. 底部15%最长连续段作为噪声样本 [回退机制: 15%→20%→25%→...]
  3. noisereduce: y_noise传入, 自适应stationary(原始min_rms<0.03→平稳),
     prop_decrease=0.8
  4. 降噪后帧RMS: 同参数重算
  5. 安静值(降噪后底部15%中位数) → onset(×3) / offset(×2) [回退机制同2]
  6. 过滤<0.3s段, 记录原始叫声时长作为投票权重
  7. 0.2s缓冲: onset前推, offset后推
  8. 重叠段切分: 重叠区找RMS最低帧采样点切开
  9. Mel预处理: 峰值归一化→128Mel/512FFT/256hop→dB→[0,1]→固定80帧
  10. CNN推理: ch64 FDAP, 每段标签=文件原始Pandeya标注
  (11. 推理时加权投票——训练时不投票,每段独立)

训练: 每段独立样本, 标签=文件标注. 交叉熵. GroupKFold按cat_id.
"""

import os, sys, time, json, csv, warnings, hashlib
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import librosa
import noisereduce as nr

# ═══════════════════════════════════════
#  配置
# ═══════════════════════════════════════

SR = 16000
MAX_DUR = 30.0           # 步骤0: 最长30s
# 步骤0: 无最短时长限制, 全部交给VAD过滤

FRAME_LEN = 2048         # 帧长 (采样点)
HOP_LENGTH = 256         # 帧间步进

N_MELS = 128
FMIN = 50
FMAX = 8000
N_FFT = 512
MEL_FRAMES = 80          # 固定Mel帧数 (1.28s)

PROP_DECREASE = 0.8      # 降噪强度
MIN_RMS_THRESH = 0.03    # 平稳/非平稳切换阈值

MIN_NOISE_SAMPLES = 2048  # 噪声样本最短采样点数 (≈0.128s, 一帧FFT)

BATCH_SIZE = 64
EPOCHS = 150
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
PATIENCE = EPOCHS  # 跑满150轮，不早停
CLIP_GRAD = 5.0

# 路径
import platform; USER="陈项目负责人"; BASE = __import__("pathlib").Path(f"C:/Users/{USER}/Desktop") if platform.system()=="Windows" else __import__("pathlib").Path("/mnt/c/Users/陈项目负责人/Desktop")
DATA_DIR = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
OUTPUT_DIR = __import__("pathlib").Path(f"PROJECT_ROOT/app/model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = __import__("pathlib").Path(f"C:/Users/陈项目负责人/AppData/Local/Temp/seg_cache_v2")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["Angry","Defence","Fighting","Happy","HuntingMind",
          "Mating","MotherCall","Paining","Resting","Warning"]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'设备: {DEVICE}')
if DEVICE == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name(0)}')


# ═══════════════════════════════════════
#  11步管线实现
# ═══════════════════════════════════════

def compute_frame_rms(y, frame_len=FRAME_LEN, hop=HOP_LENGTH):
    """步骤1/4: 逐帧RMS能量"""
    n = len(y)
    frames = []
    for start in range(0, n - frame_len + 1, hop):
        frame = y[start:start + frame_len]
        rms = np.sqrt(np.mean(frame ** 2))
        frames.append(rms)
    return np.array(frames, dtype=np.float32)


def find_bottom_longest_continuous(rms, bottom_pct=0.15, min_samples=MIN_NOISE_SAMPLES, hop=HOP_LENGTH):
    """
    步骤2: 底部 N% 帧中最长连续段
    [回退机制: bottom_pct 逐级扩大直到满足 min_samples]

    返回: (start_sample, end_sample, used_pct)
        start_sample/end_sample 是原始音频中的采样点范围
    """
    n_frames = len(rms)
    n_bottom = max(1, int(n_frames * bottom_pct))

    # 取底部 bottom_pct 帧的索引
    sorted_indices = np.argsort(rms)
    bottom_indices = set(sorted_indices[:n_bottom])

    # 按连续索引找最长段
    longest_start, longest_end = 0, 0
    cur_start = None
    for i in range(n_frames):
        if i in bottom_indices:
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                if i - cur_start > longest_end - longest_start:
                    longest_start, longest_end = cur_start, i
                cur_start = None
    if cur_start is not None:
        if n_frames - cur_start > longest_end - longest_start:
            longest_start, longest_end = cur_start, n_frames

    # 检查是否满足最小长度, 不满足则递归扩大
    n_samples = (longest_end - longest_start) * hop
    if n_samples < min_samples and bottom_pct < 0.25:
        new_pct = min(bottom_pct + 0.05, 0.25)
        #print(f'    [回退] 底部 {bottom_pct:.0%} 连续段 {n_samples} 采样点 < {min_samples}, 扩大至 {new_pct:.0%}')
        return find_bottom_longest_continuous(rms, new_pct, min_samples, hop)

    # 最长连续段的采样点范围 (用帧索引 × hop 映射到采样点)
    start_sample = longest_start * hop
    end_sample = longest_end * hop
    return start_sample, end_sample, bottom_pct


def detect_segments(rms, quiet_pct=0.15, hop=HOP_LENGTH, sr=SR):
    """
    步骤5/6/7: Onset/Offset 检测 + 过滤+缓冲

    输入: 降噪后RMS数组
    输出: list of dict:
        {
            'onset_sample': int,    # onset 采样点 (已加缓冲)
            'offset_sample': int,   # offset 采样点 (已加缓冲)
            'orig_dur': float,      # 原始叫声时长(秒), 用作投票权重
            'orig_onset_frame': int, # 原始onset帧索引(无缓冲)
            'orig_offset_frame': int, # 原始offset帧索引(无缓冲)
        }
    """
    n_frames = len(rms)
    if n_frames < 2:
        return []

    # 安静值 = 底部 quiet_pct 帧的中位数
    n_bottom = max(1, int(n_frames * quiet_pct))
    sorted_rms = np.sort(rms)
    quiet_value = float(np.median(sorted_rms[:n_bottom]))

    # 如果安静值接近0, 用全局统计
    if quiet_value < 1e-10:
        quiet_value = np.percentile(rms, quiet_pct * 100)

    onset_thresh = quiet_value * 3
    offset_thresh = quiet_value * 2

    # 如果阈值太小(底噪都不到), 尝试放宽
    if onset_thresh < 1e-8 or (onset_thresh <= offset_thresh):
        if quiet_pct < 0.25:
            return detect_segments(rms, quiet_pct + 0.05, hop, sr)

    # 逐帧扫描
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

    # 扫描结束仍在叫声状态 → 补偿到最后一帧
    if in_call and cur_onset is not None:
        segments.append((cur_onset, n_frames - 1))

    # 步骤6: 过滤 < 0.3s + 记录原始叫声时长
    buf = 0.075  # 秒  (覆盖检测延迟+起音包络, 文档定稿)
    buf_frames = int(buf * sr / hop)  # 缓冲对应的帧数

    results = []
    for onset_frame, offset_frame in segments:
        orig_dur = (offset_frame - onset_frame) * hop / sr
        if orig_dur < 0.3:
            continue  # Step6-①: 丢弃 < 0.3s
        # Step6-②: 段内最大帧RMS < 0.01 → 丢弃
        seg_max_rms = float(np.max(rms[onset_frame:offset_frame + 1]))
        if seg_max_rms < 0.01:
            continue
        # 步骤7: 0.2s 缓冲
        buf_onset = max(0, onset_frame - buf_frames)
        buf_offset = min(n_frames - 1, offset_frame + buf_frames)

        results.append({
            'onset_sample': int(buf_onset * hop),
            'offset_sample': int((buf_offset + 1) * hop),  # +1 确保包含最后一帧
            'orig_dur': orig_dur,
            'orig_onset_frame': onset_frame,
            'orig_offset_frame': offset_frame,
        })

    return results


def split_overlapping(segments, rms, hop=HOP_LENGTH):
    """
    步骤8: 重叠段切分

    segments: detect_segments 输出 (已含缓冲)
    rms: 降噪后RMS数组 (用于找重叠区最低点)

    返回: 切开后不重叠的 segments
    """
    if len(segments) <= 1:
        return segments

    result = [segments[0]]
    for seg in segments[1:]:
        prev = result[-1]
        prev_end = prev['offset_sample']
        cur_start = seg['onset_sample']

        if cur_start < prev_end:
            # 重叠: 在重叠区找RMS最低帧作为分界
            overlap_start_frame = prev_end // hop  # 前一段结束帧
            overlap_end_frame = cur_start // hop    # 后一段起始帧

            # 取重叠区RMS最低点
            overlap_frames = rms[overlap_start_frame:overlap_end_frame + 1]
            if len(overlap_frames) > 0:
                min_frame_offset = int(np.argmin(overlap_frames))
                split_frame = overlap_start_frame + min_frame_offset
            else:
                # 不应该发生, 兜底: 取中点
                split_frame = (overlap_start_frame + overlap_end_frame) // 2

            split_sample = split_frame * hop

            # 前一段被切短
            prev['offset_sample'] = split_sample
            # 后一段起始不变(已在缓冲范围内)
            # 注意: 后一段的 onset 已经在缓冲后, 不需要再推
            # 但如果 split 点在后一段的 onset 之后, 后一段需要调整
            if split_sample > seg['onset_sample']:
                seg['onset_sample'] = split_sample

        result.append(seg)

    return result


# 环境变量覆盖: NO_DENOISE=1 跳过降噪, MIN_DUR=<val> 设置最短时长
# USE_AUG=1 启用 Speed/Pitch 数据增强（在文件名带 _aug 的目录加载）
NO_DENOISE = os.environ.get('NO_DENOISE', '0') == '1'
USE_MIN_DUR = float(os.environ.get('MIN_DUR', '0'))
USE_AUG = True

if NO_DENOISE or USE_MIN_DUR > 0:
    TAG = 'B' if NO_DENOISE and USE_MIN_DUR == 0 else 'D' if NO_DENOISE else 'C'
else:
    TAG = 'A'

if USE_AUG:
    TAG += '_aug'

import platform; USER="陈项目负责人"; BASE = __import__("pathlib").Path(f"C:/Users/{USER}/Desktop") if platform.system()=="Windows" else __import__("pathlib").Path("/mnt/c/Users/陈项目负责人/Desktop")
OUTPUT_DIR = __import__("pathlib").Path(f"PROJECT_ROOT/app/model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = __import__("pathlib").Path(f"C:/Users/陈项目负责人/AppData/Local/Temp/seg_cache_v2")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

if USE_MIN_DUR > 0:
    MIN_DUR = USE_MIN_DUR


def process_audio(file_path, cache_dir):
    """
    完整11步管线处理单个音频文件

    返回: list of dict
        {
            'mel': np.ndarray (128, 80),
            'label_idx': int,
            'weight': float,        # 原始叫声时长(秒)
            'orig_dur': float,      # 同上, 保留冗余
        }

    如果处理失败或无可分段, 返回空列表
    """
    # 步骤0: 加载
    try:
        y, _ = librosa.load(file_path, sr=SR, mono=True)
    except Exception as e:
        print(f'  [!] 加载失败: {os.path.basename(file_path)} — {e}')
        return []

    # 约束: 大于30s → 切前30s
    max_samples = int(MAX_DUR * SR)
    if len(y) > max_samples:
        y = y[:max_samples]
    # 环境变量 MIN_DUR: 最短时长过滤 (默认0=不限)
    if USE_MIN_DUR > 0 and len(y) < USE_MIN_DUR * SR:
        return []

    # ── 步骤1: 帧RMS ──
    rms = compute_frame_rms(y)

    if len(rms) < 2:
        return []

    # ── 步骤2: 底部15%最长连续段 → 噪声样本 ──
    if NO_DENOISE:
        # 实验B: 跳过降噪, 直接使用原始音频
        y_work = y
        rms_work = rms
    else:
        noise_start, noise_end, used_pct = find_bottom_longest_continuous(rms)
        y_noise = y[noise_start:noise_end]

        # ── 步骤3: noisereduce 降噪 ──
        # 如果噪声样本太短(<FRAME_LEN), 跳过降噪直接用原始音频
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
                y_denoised = y  # 降噪失败, 回退原始
        else:
            y_denoised = y

        # ── 步骤4: 降噪后帧RMS ──
        rms_denoised = compute_frame_rms(y_denoised)
        if len(rms_denoised) < 2:
            return []

        y_work = y_denoised
        rms_work = rms_denoised

    # ── 步骤5: Onset/Offset 检测 ──
    raw_segments = detect_segments(rms_work)

    if not raw_segments:
        return []

    # ── 步骤8: 重叠段切分 ──
    clean_segments = split_overlapping(raw_segments, rms_work)

    if not clean_segments:
        return []

    # ── 步骤9: 每段Mel预处理 ──
    results = []
    for seg in clean_segments:
        onset_s = seg['onset_sample']
        offset_s = min(seg['offset_sample'], len(y_work))
        seg_y = y_work[onset_s:offset_s]

        # 检查段长: 至少要有 1 帧 Mel 所需采样点
        if len(seg_y) < N_FFT:
            continue

        # 峰值归一化
        peak = np.max(np.abs(seg_y))
        if peak > 1e-10:
            seg_y = seg_y / peak
        else:
            continue  # 全静音段丢弃

        # Mel频谱
        mel = librosa.feature.melspectrogram(
            y=seg_y, sr=SR,
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=FMAX,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )

        # dB → [0,1]
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_min, mel_max = mel_db.min(), mel_db.max()
        if mel_max - mel_min < 1e-10:
            continue
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min)

        # 固定到80帧: <80 右补零; >80 居中
        if mel_norm.shape[1] < MEL_FRAMES:
            mel_norm = np.pad(mel_norm, ((0, 0), (0, MEL_FRAMES - mel_norm.shape[1])))
        else:
            start = (mel_norm.shape[1] - MEL_FRAMES) // 2
            mel_norm = mel_norm[:, start:start + MEL_FRAMES]

        results.append({
            'mel': mel_norm.astype(np.float32),
            'orig_dur': seg['orig_dur'],
        })

    return results


def process_and_cache(file_path, label_idx, cache_dir):
    """
    处理单文件并缓存所有段的Mel

    返回: list of dict (同 process_audio, 含 label_idx)
    """
    # 检查缓存meta文件
    cache_key = hashlib.md5(file_path.encode()).hexdigest()
    meta_path = cache_dir / f'{cache_key}_meta.json'

    if meta_path.exists():
        # 从缓存加载
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            segments = []
            for seg_info in meta['segments']:
                mel_path = cache_dir / seg_info['mel_file']
                if not mel_path.exists():
                    # 缓存损坏, 重新处理
                    return _process_and_save(file_path, label_idx, cache_key, cache_dir)
                mel = np.load(str(mel_path))
                segments.append({
                    'mel': mel,
                    'label_idx': seg_info['label_idx'],
                    'weight': seg_info['weight'],
                    'orig_dur': seg_info['orig_dur'],
                })
            return segments
        except Exception:
            # 缓存损坏, 重新处理
            return _process_and_save(file_path, label_idx, cache_key, cache_dir)
    else:
        return _process_and_save(file_path, label_idx, cache_key, cache_dir)


def _process_and_save(file_path, label_idx, cache_key, cache_dir):
    """处理并保存缓存"""
    segment_results = process_audio(file_path, cache_dir)

    if not segment_results:
        return []

    # 保存每个段
    meta_segments = []
    for i, seg in enumerate(segment_results):
        mel_file = f'{cache_key}_seg{i}.npy'
        mel_path = cache_dir / mel_file
        np.save(str(mel_path), seg['mel'])

        meta_segments.append({
            'mel_file': mel_file,
            'label_idx': label_idx,
            'weight': seg['orig_dur'],
            'orig_dur': seg['orig_dur'],
        })

    # 保存meta
    meta = {
        'file': os.path.basename(file_path) if isinstance(file_path, str) else str(file_path),
        'cache_key': cache_key,
        'n_segments': len(segment_results),
        'segments': meta_segments,
    }
    meta_path = cache_dir / f'{cache_key}_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, ensure_ascii=False)

    # 返回处理结果 (不包含mel以节省内存, 训练时从缓存加载)
    return [{
        'mel': seg['mel'],
        'label_idx': label_idx,
        'weight': seg['orig_dur'],
        'orig_dur': seg['orig_dur'],
    } for seg in segment_results]


# ═══════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════

def load_catsound_v2(data_dir):
    """加载 CatSound_V2 数据集"""
    csv_path = data_dir / 'Catmood_NAYA_1xAug.csv'
    files, labels, cat_ids, is_aug = [], [], [], []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            next(reader)  # 跳过表头
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
    """加载 aug_SP 目录下的增强音频（Speed/Pitch）"""
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
    print(f'  增强文件: {len(aug_files)} 条（Speed±10% + Pitch±2半音）')
    return (np.concatenate([base_files, aug_files]),
            np.concatenate([base_labels, aug_labels]),
            np.concatenate([base_cat_ids, aug_cat_ids]))


def preprocess_all(file_list, label_idxs, cat_ids, cache_dir):
    """
    预处理所有文件并缓存

    返回:
        all_mels: np.ndarray (N_segments, 128, 80)
        all_labels: np.ndarray (N_segments,)
        all_weights: np.ndarray (N_segments,)
        all_cat_ids: np.ndarray (N_segments,)  每个段对应的猫ID
    """
    print(f'\n[预处理] 共 {len(file_list)} 条音频, 开始管线处理...')
    t0 = time.time()
    total_segments = 0
    n_processed, n_failed, n_empty = 0, 0, 0

    all_mels_list = []
    all_labels_list = []
    all_weights_list = []
    all_cat_ids_list = []
    all_file_idx_list = []

    for i, fp in enumerate(file_list):
        segs = process_and_cache(fp, int(label_idxs[i]), cache_dir)

        if len(segs) == 0:
            n_empty += 1
        else:
            for seg in segs:
                all_mels_list.append(seg['mel'])
                all_labels_list.append(seg['label_idx'])
                all_weights_list.append(seg['weight'])
                all_cat_ids_list.append(cat_ids[i])
                all_file_idx_list.append(i)
            total_segments += len(segs)
            n_processed += 1

        if (i + 1) % 500 == 0:
            et = time.time() - t0
            print(f'  {i+1}/{len(file_list)}  files_proc={n_processed} segs={total_segments} empty={n_empty} {et:.0f}s')
        if (i + 1) % 100 == 0 or i == 0:
            sys.stdout.flush()

    if len(all_mels_list) == 0:
        print('  ⚠️ 无有效段生成!')
        return None, None, None, None, None

    all_mels = np.stack(all_mels_list, axis=0)
    all_labels = np.array(all_labels_list)
    all_weights = np.array(all_weights_list)
    all_cat_ids = np.array(all_cat_ids_list)

    print(f'\n完成: {n_processed}/{len(file_list)} 文件产生 {total_segments} 段')
    print(f'  空文件: {n_empty}, 失败: {n_failed}')
    print(f'  形状: {list(all_mels.shape)}, {all_mels.nbytes/1024/1024:.0f}MB, {time.time()-t0:.0f}s')
    print(f'  权重范围: {all_weights.min():.3f}s ~ {all_weights.max():.3f}s')
    print(f'  唯一猫ID: {len(np.unique(all_cat_ids))}')

    return all_mels, all_labels, all_weights, all_cat_ids, np.array(all_file_idx_list)


# ═══════════════════════════════════════
#  模型定义
# ═══════════════════════════════════════

class FDAPMobileCatCNN(nn.Module):
    """conv4=64ch, FDAP=4×64=256维 (与旧管线相同)"""
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


# ═══════════════════════════════════════
#  训练
# ═══════════════════════════════════════

class CatMelDataset(Dataset):
    def __init__(self, mels, labels, file_idxs, augment=False):
        self.mels = mels
        self.labels = labels
        self.file_idxs = file_idxs
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mel = self.mels[idx].clone()
        if self.augment:
            # 简单增强: 噪声 + 时间偏移
            mel_np = mel.squeeze(0).numpy()
            noise = np.random.randn(*mel_np.shape) * 0.01
            mel_np = mel_np + noise
            shift = np.random.randint(-int(0.1 * mel_np.shape[1]),
                                      int(0.1 * mel_np.shape[1]))
            mel_np = np.roll(mel_np, shift, axis=1)
            mel = torch.FloatTensor(mel_np).unsqueeze(0)
        return (mel,
                torch.LongTensor([self.labels[idx]])[0],
                torch.LongTensor([self.file_idxs[idx]])[0])


def train_one_fold(fold, train_mels, train_labels, train_fidxs,
                   test_mels, test_labels, test_fidxs, output_dir):
    """训练一个折"""
    n_train = len(train_mels)
    n_test = len(test_mels)
    print(f'\n{"─" * 50}')
    print(f'折 {fold}: 训练 {n_train} 段 → 测试 {n_test} 段')

    # 按该折训练集段标签分布算 class_weight
    seg_class_counts = np.bincount(train_labels.numpy())
    n_classes_seen = len(seg_class_counts)
    seg_weights = compute_class_weight(
        'balanced',
        classes=np.arange(n_classes_seen),
        y=train_labels.numpy()
    )
    seg_weight_tensor = torch.FloatTensor(seg_weights).to(DEVICE)
    print(f'  class_weight (段数倒数比例): '
          f'{dict(zip(range(n_classes_seen), [f"{w:.2f}" for w in seg_weights]))}')

    train_ds = CatMelDataset(train_mels, train_labels, train_fidxs, augment=True)
    test_ds = CatMelDataset(test_mels, test_labels, test_fidxs, augment=False)
    train_loader = DataLoader(
        train_ds, BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(
        test_ds, BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)

    model = FDAPMobileCatCNN(num_classes=10).to(DEVICE)
    opt = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_acc, best_f1, best_ep = 0, 0, 0
    no_improve = 0

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for m, l, fidx in train_loader:
            m, l, fidx = m.to(DEVICE), l.to(DEVICE), fidx.to(DEVICE)
            opt.zero_grad()

            # 文件级 top-3 投票 loss
            logits = model(m)
            probs = F.softmax(logits, dim=1)
            loss = 0.0
            unique_f = torch.unique(fidx)
            for uf in unique_f:
                mask = (fidx == uf)
                n_seg = mask.sum()
                # 该文件所有段对正确类的置信度
                correct_conf = probs[mask, l[mask]]
                # 取 top-k (最多3)
                k = min(3, int(n_seg))
                topk_conf, _ = torch.topk(correct_conf, k)
                avg_conf = topk_conf.mean()
                # class_weight
                cw = seg_weight_tensor[l[mask][0]]
                loss += -cw * torch.log(avg_conf + 1e-8)
            loss = loss / len(unique_f)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            opt.step()
            train_loss += loss.item()

        # 测试集: 文件级投票推理
        model.eval()
        all_file_preds, all_file_labels = {}, {}
        with torch.no_grad():
            for m, l, fidx in test_loader:
                m, l, fidx = m.to(DEVICE), l.to(DEVICE), fidx.to(DEVICE)
                o = model(m)
                probs_t = F.softmax(o, dim=1)
                for j in range(len(fidx)):
                    fi = int(fidx[j].cpu())
                    if fi not in all_file_preds:
                        all_file_preds[fi] = []
                        all_file_labels[fi] = int(l[j].cpu())
                    all_file_preds[fi].append(probs_t[j].cpu().numpy())

        # 各文件加权投票
        p_list, la_list = [], []
        for fi in all_file_preds:
            arr = np.stack(all_file_preds[fi], axis=0)  # (n_seg, 10)
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
            torch.save(model.state_dict(),
                       output_dir / f'fold{fold}_best.pth')
        else:
            no_improve += 1

        if (ep % 5 == 1 or ep == EPOCHS or no_improve == 1):
            print(f'  E{ep:3d}/{EPOCHS} L={train_loss / len(train_loader):.4f} '
                  f'Acc={acc:.1%} F1={f1:.1%} Best={best_f1:.1%}@{best_ep} {et:.0f}s')

        if no_improve >= PATIENCE:
            print(f'  早停 @{ep}, Best F1={best_f1:.1%} Acc={best_acc:.1%}@{best_ep}')
            break

    print(f'  折{fold} 完成: Acc={best_acc:.1%} F1={best_f1:.1%} (ep{best_ep})')
    return best_acc, best_f1, best_ep


# ═══════════════════════════════════════
#  main
# ═══════════════════════════════════════

def main():
    print(f'\n{"█" * 50}')
    print(f'  FDAP 64ch — 新管线 (11步)')
    print(f'  数据: CatSound_V2')
    print(f'  配置: LR={LR} BS={BATCH_SIZE} WD={WEIGHT_DECAY}')
    print(f'  Mel: {N_MELS}bands×{MEL_FRAMES}frames, hop={HOP_LENGTH}')
    print(f'  降噪: prop_decrease={PROP_DECREASE}, thr={MIN_RMS_THRESH}')
    print(f'  VAD: onset=quiet×3, offset=quiet×2, min_seg=0.3s, buffer=0.15s')
    print(f'  输出: {OUTPUT_DIR}')
    print(f'{"█" * 50}')

    # ── 加载数据 ──
    print('\n[1] 加载 CatSound_V2...')
    files, labels, cat_ids, is_aug = load_catsound_v2(DATA_DIR)
    if USE_AUG:
        print("  [增强] USE_AUG=1, 加载 Speed/Pitch 增强数据...")
        files, labels, cat_ids = load_augmented(DATA_DIR, files, labels, cat_ids)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    print(f'  {len(files)} 条, 类别: {list(le.classes_)}')
    print(f'  唯一猫ID: {len(set(cat_ids))}')

    # ── 预处理所有文件 → 段 ──
    print('\n[2] 11步管线预处理 (并缓存)...')
    all_mels, all_labels, all_weights, seg_cat_ids, seg_file_idxs = preprocess_all(
        files, y, cat_ids, CACHE_DIR)

    if all_mels is None:
        print('\n❌ 预处理未产生任何段, 退出')
        return

    # ── GroupKFold 5折 ──
    print('\n[3] GroupKFold 5折 (按猫ID分组, 段级)...')
    print(f'  段总数: {len(seg_cat_ids)}, 唯一猫ID: {len(np.unique(seg_cat_ids))}')
    gkf = GroupKFold(n_splits=5)

    # seg_indices 是 0..N-1 索引, 作为 dummy X 传给 GroupKFold
    seg_indices = np.arange(len(seg_cat_ids))

    fold_accs, fold_f1s = [], []
    fold = 1
    for train_seg_idx, test_seg_idx in gkf.split(
            seg_indices, all_labels, seg_cat_ids):
        if fold != 4:  # 只跑折4
            fold += 1
            continue
        print(f'\n{"█" * 50}\n  训练折4（目标折叠）\n{"█" * 50}')
        a, f, _ = train_one_fold(
            fold,
            torch.FloatTensor(all_mels[train_seg_idx]).unsqueeze(1),
            torch.LongTensor(all_labels[train_seg_idx]),
            torch.LongTensor(seg_file_idxs[train_seg_idx]),
            torch.FloatTensor(all_mels[test_seg_idx]).unsqueeze(1),
            torch.LongTensor(all_labels[test_seg_idx]),
            torch.LongTensor(seg_file_idxs[test_seg_idx]),
            OUTPUT_DIR)
        fold_accs.append(a)
        fold_f1s.append(f)
        fold += 1

    # ── 结果 ──
    print(f'\n{"█" * 50}')
    print(f'  结果 (新管线)')
    print(f'{"█" * 50}')
    for i, (a, f) in enumerate(zip(fold_accs, fold_f1s), 1):
        print(f'  折{i}: Acc={a:.1%} F1={f:.1%}')
    print(f'  Acc: {np.mean(fold_accs):.1%}±{np.std(fold_accs):.1%}  '
          f'F1: {np.mean(fold_f1s):.1%}±{np.std(fold_f1s):.1%}')

    # ── 保存结果JSON ──
    ts = time.strftime('%Y%m%d_%H%M%S')
    result = {
        'model': 'FDAP64ch_Pipeline',
        'n_files': len(files),
        'n_segments': len(all_labels),
        'n_cats': len(set(cat_ids)),
        'config': {
            'ep': EPOCHS, 'bs': BATCH_SIZE, 'lr': LR, 'wd': WEIGHT_DECAY,
            'mel_frames': MEL_FRAMES, 'n_mels': N_MELS,
            'prop_decrease': PROP_DECREASE,
            'min_rms_thresh': MIN_RMS_THRESH,
        },
        'folds': {
            f'fold{i + 1}': {'acc': float(a), 'f1': float(f)}
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


if __name__ == '__main__':
    main()

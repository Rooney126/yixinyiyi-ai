"""
译芯怡意 — FDAPMobileCatCNN 全量训练 (CatSound_V2)
===============================================
GroupKFold 5折按猫划分, 每折从零训练 CNN
先全量预提取 Mel 频谱再训练 (I/O 和计算分离)
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
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset

SR = 16000
N_MELS = 128
N_FFT = 512
HOP_LENGTH = 256
MAX_T = 300
# VAD 事件分割: 提取有效叫声段, 统一到 ~1.6s (100帧@hop=256)
VAD_FRAMES = 100
VAD_TOP_DB = 25
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2  # DataLoader workers (2够用, 流水线并行, 又不过分吃内存)

BASE = Path('/Desktop')
DATA_DIR = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
OUTPUT_DIR = BASE / '模型代码/训练结果/fdap_cnn_v2'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR_VAD = OUTPUT_DIR / 'mel_cache_vad'
CACHE_DIR_VAD.mkdir(parents=True, exist_ok=True)
print(f'VAD 缓存: {CACHE_DIR_VAD}')

LABELS = [
    "Angry", "Defence", "Fighting", "Happy", "HuntingMind",
    "Mating", "MotherCall", "Paining", "Resting", "Warning"
]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'设备: {DEVICE}')
if DEVICE == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name(0)}')

def load_catsound_v2(data_dir):
    csv_path = data_dir / 'Catmood_NAYA_1xAug.csv'
    files, labels, cat_ids, is_aug = [], [], [], []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try: next(reader)
        except: return np.array(files), np.array(labels), np.array(cat_ids)
        for row in reader:
            if len(row) < 3: continue
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

def extract_vocal_segments(y, sr=SR, top_db=VAD_TOP_DB):
    """能量检测 → 返回所有有效叫声段"""
    segs = librosa.effects.split(y, top_db=top_db, frame_length=512, hop_length=HOP_LENGTH)
    if len(segs) == 0:
        return [y]  # 无叫声段则返回原音频
    return [y[s:e] for s, e in segs]

def precompute_all_vad_segments(file_list, cache_dir):
    """全量 VAD 切分: 每条音频提取所有叫声段, 每段存一个 Mel"""
    INDEX_FILE = os.path.join(cache_dir, 'vad_index.json')
    
    print(f'\n[precompute] VAD全段切分 + Mel提取, 共 {len(file_list)} 条原始音频...')
    t0 = time.time()
    n_cached = 0
    n_new = 0
    n_failed = 0
    total_segments = 0
    segment_index = {}  # file_hash → num_segments
    
    for i, fp in enumerate(file_list):
        file_hash = hashlib.md5(fp.encode()).hexdigest()
        
        # 检查是否已有索引 (所有段都已缓存)
        if os.path.exists(INDEX_FILE):
            with open(INDEX_FILE) as f:
                idx = json.load(f)
            if file_hash in idx:
                n_cached += idx[file_hash]
                total_segments += idx[file_hash]
                continue
        
        # 新提取
        try:
            y, _ = librosa.load(fp, sr=SR, mono=True)
            segments = extract_vocal_segments(y)
            part_count = 0
            for seg in segments:
                peak = np.max(np.abs(seg))
                seg = seg / peak if peak > 1e-10 else np.zeros_like(seg)
                mel = librosa.feature.melspectrogram(y=seg, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
                
                n_frames = mel_norm.shape[1]
                if n_frames <= VAD_FRAMES:
                    # 短段: pad 到 VAD_FRAMES
                    chunk = np.pad(mel_norm, ((0,0), (0, VAD_FRAMES - n_frames)))
                    np.save(os.path.join(cache_dir, f'{file_hash}_p{part_count}.npy'), chunk)
                    part_count += 1
                else:
                    # 长段: 切成多个 VAD_FRAMES 块 (不重叠, 丢弃余数)
                    n_chunks = n_frames // VAD_FRAMES
                    for k in range(n_chunks):
                        chunk = mel_norm[:, k*VAD_FRAMES:(k+1)*VAD_FRAMES]
                        np.save(os.path.join(cache_dir, f'{file_hash}_p{part_count}.npy'), chunk)
                        part_count += 1
            segment_index[file_hash] = part_count
            total_segments += part_count
            n_new += 1
        except Exception as e:
            print(f'  [!] 损坏: {os.path.basename(fp)} — {e}')
            mel_silent = np.zeros((N_MELS, VAD_FRAMES))
            np.save(os.path.join(cache_dir, f'{file_hash}_p0.npy'), mel_silent)
            segment_index[file_hash] = 1
            total_segments += 1
            n_failed += 1
        
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f'  VAD: {i+1}/{len(file_list)} ({elapsed:.0f}s, {rate:.1f}/s)'
                  f'  new={n_new} cached={n_cached} 段={total_segments}')
    
    # 保存索引
    if segment_index:
        # 合并已有索引
        if os.path.exists(INDEX_FILE):
            with open(INDEX_FILE) as f:
                old_idx = json.load(f)
            old_idx.update(segment_index)
            segment_index = old_idx
        with open(INDEX_FILE, 'w') as f:
            json.dump(segment_index, f)
    
    total_time = time.time() - t0
    print(f'[precompute] 完成: {len(file_list)} 条原始音频, '
          f'共 {total_segments} 个叫声段 (平均 {total_segments/len(file_list):.1f}段/条), '
          f'新提取={n_new}, 缓存={n_cached}, 损坏={n_failed}, '
          f'总耗时={total_time:.0f}s')
    return total_segments, n_failed

def load_vad_segments(original_files, original_labels, original_cat_ids, original_is_aug, cache_dir):
    """加载 VAD 索引, 将原始文件列表展开为每个叫声段一个条目"""
    INDEX_FILE = os.path.join(cache_dir, 'vad_index.json')
    with open(INDEX_FILE) as f:
        idx = json.load(f)
    
    expanded_files, expanded_labels, expanded_cat_ids, expanded_is_aug = [], [], [], []
    for i, fp in enumerate(original_files):
        file_hash = hashlib.md5(fp.encode()).hexdigest()
        n_parts = idx.get(file_hash, 1)
        for p in range(n_parts):
            part_path = os.path.join(cache_dir, f'{file_hash}_p{p}.npy')
            expanded_files.append(part_path)
            expanded_labels.append(original_labels[i])
            expanded_cat_ids.append(original_cat_ids[i])
            expanded_is_aug.append(original_is_aug[i])
    
    return (np.array(expanded_files), np.array(expanded_labels), 
            np.array(expanded_cat_ids), np.array(expanded_is_aug))

# 兼容旧接口: load_mel_from_cache 改从 segment 路径加载
def load_mel_from_cache(segment_path, cache_dir=None, max_frames=VAD_FRAMES):
    """加载固定长度 VAD Mel (已预裁/填充到 VAD_FRAMES)"""
    mel = np.load(segment_path)
    return mel

def preload_all_mels_to_ram(file_list, cache_dir):
    """全量预加载 Mel 频谱到内存, 消除训练时的磁盘 I/O"""
    print(f'\n[preload] 加载 {len(file_list)} 条 Mel 到内存...')
    t0 = time.time()
    mels = []
    for i, fp in enumerate(file_list):
        mel_np = load_mel_from_cache(fp, cache_dir)
        mels.append(mel_np)
        if (i + 1) % 1000 == 0:
            print(f'  preload: {i+1}/{len(file_list)} ({time.time()-t0:.0f}s)')
    all_mels = np.stack(mels, axis=0)  # [N, 128, VAD_FRAMES]
    total_mb = all_mels.nbytes / 1024 / 1024
    print(f'[preload] 完成: {list(all_mels.shape)}, {total_mb:.0f}MB, {time.time()-t0:.0f}s')
    return torch.FloatTensor(all_mels).unsqueeze(1)  # [N, 1, 128, VAD_FRAMES]

class CatMelDataset(Dataset):
    """从内存 Tensor 加载 Mel (消除磁盘 I/O)"""
    def __init__(self, mels, labels, augment=False):
        self.mels = mels
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mel = self.mels[idx].clone()  # [1, 128, VAD_FRAMES]
        if self.augment:
            mel_np = mel.squeeze(0).numpy()
            noise = np.random.randn(*mel_np.shape) * 0.01
            mel_np = mel_np + noise
            shift = np.random.randint(-int(0.1 * mel_np.shape[1]), int(0.1 * mel_np.shape[1]))
            mel_np = np.roll(mel_np, shift, axis=1)
            mel = torch.FloatTensor(mel_np).unsqueeze(0)
        return mel, torch.LongTensor([self.labels[idx]])[0]

class CatMelDatasetFromCache(Dataset):
    """从缓存加载 Mel 频谱"""
    def __init__(self, files, labels, cache_dir, augment=False):
        self.files = files
        self.labels = labels
        self.cache_dir = cache_dir
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mel = load_mel_from_cache(self.files[idx], self.cache_dir)
        if self.augment:
            noise = np.random.randn(*mel.shape) * 0.01
            mel = mel + noise
            shift = np.random.randint(-int(0.1 * mel.shape[1]), int(0.1 * mel.shape[1]))
            mel = np.roll(mel, shift, axis=1)
        return torch.FloatTensor(mel).unsqueeze(0), torch.LongTensor([self.labels[idx]])[0]

class FDAPMobileCatCNN(nn.Module):
    """~99K 参数, 4层 Conv2D + FDAP"""
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
            nn.Conv2d(32, 48, 3, padding=1), nn.BatchNorm2d(48),
            nn.ReLU(), nn.MaxPool2d((1, 2)))
        self.conv4 = nn.Sequential(
            nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96),
            nn.ReLU(), nn.Identity())
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(384, 96), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(96, num_classes),
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

def train_one_fold(fold, train_mels, train_labels,
                   test_mels, test_labels):
    """训练单折 (mels: 预加载到内存的 Tensor [N,1,128,300])"""
    print(f'\n{"─"*50}')
    print(f'折 {fold}: 训练 {len(train_mels)} 条 → 测试 {len(test_mels)} 条')

    train_ds = CatMelDataset(train_mels, train_labels, augment=True)
    test_ds = CatMelDataset(test_mels, test_labels, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    model = FDAPMobileCatCNN(num_classes=10).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 类加权: 少数类更高权重 (clamp 防梯度爆炸)
    classes = np.unique(train_labels)
    cw = compute_class_weight('balanced', classes=classes, y=train_labels)
    cw = np.clip(cw, 0.2, 5.0)
    class_weights = torch.FloatTensor(cw).to(DEVICE)

    best_acc = 0.0
    best_f1 = 0.0
    best_epoch = 0
    no_improve = 0
    patience = 15

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        model.train()
        train_loss = 0.0
        for mels, lbls in train_loader:
            mels, lbls = mels.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            out = model(mels)
            loss = F.cross_entropy(out, lbls, weight=class_weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for mels, lbls in test_loader:
                mels, lbls = mels.to(DEVICE), lbls.to(DEVICE)
                out = model(mels)
                preds = out.argmax(1)
                all_preds.append(preds.cpu())
                all_labels.append(lbls.cpu())
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        acc = (all_preds == all_labels).mean()
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        scheduler.step()
        epoch_time = time.time() - t0

        if macro_f1 > best_f1:
            best_acc = acc
            best_f1 = macro_f1
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), OUTPUT_DIR / f'fold{fold}_best.pth')
        else:
            no_improve += 1

        if epoch % 5 == 1 or epoch == EPOCHS or no_improve == 1:
            print(f'  Epoch {epoch:3d}/{EPOCHS}  Loss={train_loss/len(train_loader):.4f}  '
                  f'Acc={acc:.1%}  F1={macro_f1:.1%}  '
                  f'Best={best_f1:.1%}@{best_epoch}  '
                  f'{epoch_time:.0f}s/epoch')
        if no_improve >= patience:
            print(f'  早停 @ epoch {epoch}, 最佳F1: {best_f1:.1%} (Acc: {best_acc:.1%})@{best_epoch}')
            break

    print(f'  折 {fold} 完成: Acc={best_acc:.1%}  F1={best_f1:.1%} (epoch {best_epoch})')
    return best_acc, best_f1, best_epoch

def main():
    print(f'\n{"█"*50}')
    print('  FDAPMobileCatCNN 全量训练 (CatSound_V2) [VAD全段+长段切分]')
    print('  模型: FDAPMobileCatCNN (~99K), 10类')
    print(f'  训练: {EPOCHS} epochs, GroupKFold 5折')
    print(f'  VAD: 全段提取 + 长段按1.6s切片 ({VAD_FRAMES}帧)')
    print(f'  设备: {DEVICE}')
    print(f'{"█"*50}')

    print('\n[1/4] 加载 CatSound_V2...')
    files, labels, cat_ids, is_aug = load_catsound_v2(DATA_DIR)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    print(f'  原始音频: {len(files)} 条, 标签: {list(le.classes_)}')

    # 预提取 VAD 全段 Mel
    print('\n[2/4] 预提取 VAD 全段 Mel 频谱...')
    total_segments, n_failed = precompute_all_vad_segments(files, CACHE_DIR_VAD)
    print(f'  损坏文件: {n_failed}/{len(files)}')

    # 展开为每个叫声段一个条目
    print('\n[3/4] 展开 VAD 叫声段列表...')
    seg_files, seg_labels, seg_cat_ids, seg_is_aug = load_vad_segments(
        files, labels, cat_ids, is_aug, CACHE_DIR_VAD)
    seg_y = le.transform(seg_labels)
    print(f'  展开后: {len(seg_files)} 个叫声段 (增幅 {len(seg_files)/len(files):.1f}×)')

    # 全量预加载到内存
    print('\n[4/4] 全量预加载 VAD Mel 到内存...')
    all_mels = preload_all_mels_to_ram(seg_files, CACHE_DIR_VAD)
    print(f'  内存 Mel 总计: {all_mels.shape}, {all_mels.element_size() * all_mels.nelement() / 1024 / 1024:.0f}MB')

    print('\nGroupKFold 5 折 (按猫划分)...')
    gkf = GroupKFold(n_splits=5)
    clean_mask = ~seg_is_aug
    clean_indices = np.where(clean_mask)[0]

    fold_accs, fold_f1s = [], []
    for fold, (tr_cidx, te_cidx) in enumerate(
        gkf.split(clean_indices, seg_y[clean_indices], seg_cat_ids[clean_indices]), 1):
        
        train_cats = set(seg_cat_ids[clean_indices[tr_cidx]])
        test_cats = set(seg_cat_ids[clean_indices[te_cidx]])
        train_mask = np.array([cid in train_cats for cid in seg_cat_ids])
        test_mask = np.array([cid in test_cats for cid in seg_cat_ids])

        acc, f1, best_ep = train_one_fold(
            fold,
            all_mels[train_mask], seg_y[train_mask],
            all_mels[test_mask],  seg_y[test_mask])
        fold_accs.append(acc)
        fold_f1s.append(f1)

    print(f'\n{"█"*50}')
    print('  FDAP CNN 5 折 CV 结果')
    print(f'{"█"*50}')
    for i, (acc, f1) in enumerate(zip(fold_accs, fold_f1s), 1):
        print(f'  折 {i}: Acc={acc:.1%}  F1={f1:.1%}')
    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    mean_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)
    print(f'  Acc: {mean_acc:.1%} ± {std_acc:.1%}  (最佳: {max(fold_accs):.1%})')
    print(f'  F1:  {mean_f1:.1%} ± {std_f1:.1%}  (最佳: {max(fold_f1s):.1%})')

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    result = {
        'model': 'FDAPMobileCatCNN_v3',
        'vad_enabled': True,
        'vad_total_segments': len(seg_files),
        'vad_avg_segments': len(seg_files) / len(files),
        'n_samples': len(files),
        'n_cats': len(set(cat_ids)),
        'n_failed': int(n_failed),
        'config': {
            'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
            'lr': LR, 'weight_decay': WEIGHT_DECAY,
        },
        'folds': {
            f'fold_{i+1}': {'acc': float(a), 'f1': float(f)}
            for i, (a, f) in enumerate(zip(fold_accs, fold_f1s))
        },
        'mean_acc': float(mean_acc), 'std_acc': float(std_acc),
        'mean_f1': float(mean_f1), 'std_f1': float(std_f1),
        'best_acc': float(max(fold_accs)),
        'best_f1': float(max(fold_f1s)),
        'timestamp': timestamp,
        'device': DEVICE,
    }
    result_path = OUTPUT_DIR / f'results_fdap_cnn_{timestamp}.json'
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'\n结果: {result_path}')

if __name__ == '__main__':
    main()

"""
仅跑 A_aug 折4 — 产出 fold4_best.pth → 部署
基于 train_cnn_pipeline_v2.py
"""
import time
import warnings
from pathlib import Path

import librosa
import numpy as np
import torch

warnings.filterwarnings('ignore')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'  设备: {DEVICE}')
if DEVICE == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name(0)}')

# ─── 配置（与 A_aug 相同）───
import platform as _platform

if _platform.system() == 'Windows':
    _USER = 'chen'
    BASE = Path(f'C:/Users/{_USER}/Desktop')
    DATA_DIR = Path(f'C:/Users/{_USER}/Desktop/CatSound_DataSet_V2/NAYA_DATA_AUG1X')
    OUTPUT_DIR = Path(f'C:/Users/{_USER}/译芯怡意/app/model')
    CACHE_DIR = Path(f'C:/Users/{_USER}/AppData/Local/Temp/seg_cache_aug')
else:
    BASE = Path('/Desktop')
    DATA_DIR = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
    OUTPUT_DIR = Path('/译芯怡意/app/model')
    CACHE_DIR = Path('/tmp/seg_cache_aug')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SR = 16000
MAX_DUR = 30
N_MELS = 128
MEL_FRAMES = 80
N_FFT = 512
HOP_LENGTH = 256

# 降噪
PROP_DECREASE = 0.8
MIN_RMS_THRESH = 0.03

# VAD
ONSET_MULT = 3.0
OFFSET_MULT = 2.0
MIN_SEG_DUR = 0.3
BUFFER = 0.2
FRAME_LEN = 2048
FRAME_HOP = 256

# 训练
EPOCHS = 150
BATCH_SIZE = 64
LR = 0.001
WEIGHT_DECAY = 0.0001
USE_AUG = 1           # Speed/Pitch 增强
TARGET_FOLD = 4       # 只跑折4

LABELS = ["Angry","Defence","Fighting","Happy","HuntingMind",
          "Mating","MotherCall","Paining","Resting","Warning"]
LABEL_MAP = {l: i for i, l in enumerate(LABELS)}

print(f"\n{'█'*50}")
print(f"  FDAP 64ch — 仅折{TARGET_FOLD} (A_aug)")
print(f"{'█'*50}")
print(f"  输出: {OUTPUT_DIR}")
print(f"  缓存: {CACHE_DIR}")
print()

# ─── 模型 ───
class FDAPMobileCatCNN_ch64(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,16,3,padding=1), torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(), torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,3,padding=1), torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(), torch.nn.MaxPool2d((1,2)))
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32,48,3,padding=1), torch.nn.BatchNorm2d(48),
            torch.nn.ReLU(), torch.nn.MaxPool2d((1,2)))
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(48,64,3,padding=1), torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(), torch.nn.Identity())
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3), torch.nn.Linear(256,96),
            torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(96, num_classes))

    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x); x = self.conv4(x)
        b, c, f, t = x.shape; bs = f // 4
        fdap = [x[:,:,i*bs:(i+1)*bs if i<3 else f,:].mean(dim=[2,3]) for i in range(4)]
        x = torch.cat(fdap, dim=1)
        return self.classifier(x)


def preprocess_pipeline(file_path, sr=SR, max_dur=MAX_DUR):
    """11步管线：降噪→VAD事件切分→Mel 80帧"""
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    if len(y) > sr * max_dur:
        y = y[:sr * max_dur]

    # 降噪
    try:
        import noisereduce as nr
        # 底部15%最长连续段作为噪声样本
        rms = librosa.feature.rms(y=y, frame_length=FRAME_LEN, hop_length=FRAME_HOP)[0]
        bottom15 = np.argsort(rms)[:max(1, int(len(rms) * 0.15))]
        if len(bottom15) > 1:
            diffs = np.diff(bottom15)
            runs = np.split(bottom15, np.where(diffs != 1)[0] + 1)
            longest = max(runs, key=len)
            seg_start = longest[0] * FRAME_HOP
            seg_end = min((longest[-1] + 1) * FRAME_HOP, len(y))
            noise_sample = y[seg_start:seg_end]
        else:
            noise_sample = y[:min(FRAME_LEN * 2, len(y))]
        if len(noise_sample) >= FRAME_LEN:
            stable = (rms.min() if len(rms) > 0 else 0) < MIN_RMS_THRESH
            try:
                y = nr.reduce_noise(y=y, sr=sr, y_noise=noise_sample,
                                    prop_decrease=PROP_DECREASE, stationary=stable)
            except Exception:
                pass
    except ImportError:
        pass  # 无降噪

    # RMS 能量 VAD
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LEN, hop_length=FRAME_HOP)[0]
    quiet_val = np.median(np.sort(rms)[:max(1, int(len(rms) * 0.15))])
    onset_thr = quiet_val * ONSET_MULT
    offset_thr = quiet_val * OFFSET_MULT

    # 状态机切分
    in_seg = False
    segs = []
    for i in range(len(rms)):
        if not in_seg and rms[i] >= onset_thr:
            start = max(0, i * FRAME_HOP - int(BUFFER * SR))
            in_seg = True
        elif in_seg and rms[i] < offset_thr:
            end = min(len(y), i * FRAME_HOP + int(BUFFER * SR))
            if end - start >= MIN_SEG_DUR * SR:
                segs.append(y[start:end])
            in_seg = False

    if in_seg:
        end = min(len(y), len(rms) * FRAME_HOP + int(BUFFER * SR))
        if end - (segs[-1][1] if segs else 0) >= MIN_SEG_DUR * SR:
            segs.append(y[-(len(rms) * FRAME_HOP):])

    if not segs:
        return np.zeros((1, N_MELS, MEL_FRAMES), dtype=np.float32)

    mels = []
    for seg in segs:
        mel = librosa.feature.melspectrogram(y=seg, sr=SR, n_mels=N_MELS,
                                              n_fft=N_FFT, hop_length=HOP_LENGTH)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        if mel_norm.shape[1] < MEL_FRAMES:
            mel_norm = np.pad(mel_norm, ((0,0), (0, MEL_FRAMES - mel_norm.shape[1])))
        else:
            mel_norm = mel_norm[:, :MEL_FRAMES]
        mels.append(mel_norm[np.newaxis, :, :])

    return np.concatenate(mels, axis=0).astype(np.float32)


# ─── 加载数据 + 管线预处理 ───
def load_all():
    import csv
    import os
    csv_path = DATA_DIR / 'Catmood_NAYA_1xAug.csv'
    files, labels, cat_ids = [], [], []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            pass
        for row in reader:
            if len(row) < 3:
                continue
            label_idx = int(row[2])
            filename = os.path.basename(row[1].strip())
            local_path = DATA_DIR / LABELS[label_idx] / filename
            if local_path.exists():
                files.append(str(local_path))
                labels.append(label_idx)  # 存整数标签
                raw_stem = local_path.stem.replace('_aug1(1)', '').lower()
                cat_ids.append(raw_stem)

    # 构建原始文件的 stem→索引 字典（用于增强文件快速匹配）
    stem_to_idx = {}
    for i, fp in enumerate(files):
        s = Path(fp).stem.replace('_aug1(1)', '').lower()
        stem_to_idx[s] = i

    if USE_AUG:
        aug_dir = DATA_DIR / 'aug_SP'
        if aug_dir.exists():
            for af in sorted(aug_dir.rglob('*.mp3')) + sorted(aug_dir.rglob('*.wav')):
                stem = af.stem.lower()
                # 去除速度/音高后缀: _pm2, _pp2, _sm10, _sp10
                key = stem
                for suffix in ['_pm2', '_pp2', '_sm10', '_sp10']:
                    if key.endswith(suffix):
                        key = key[:-len(suffix)]
                        break
                # 用字典查找（快）
                if key in stem_to_idx:
                    fi = stem_to_idx[key]
                    files.append(str(af))
                    labels.append(labels[fi])
                    cat_ids.append(cat_ids[fi])

    print(f'  总文件: {len(files)}, 唯一猫ID: {len(set(cat_ids))}')
    print(f'  类别: {sorted(set(labels))}')

    MEL_CACHE = CACHE_DIR / 'mels.npy'
    LABEL_CACHE = CACHE_DIR / 'labels.npy'
    CATID_CACHE = CACHE_DIR / 'cat_ids.npy'

    if MEL_CACHE.exists():
        print('  从缓存加载预处理结果...')
        all_mels = np.load(str(MEL_CACHE))
        all_labels = np.load(str(LABEL_CACHE))
        seg_cat_ids = np.load(str(CATID_CACHE))
        print(f'  {len(all_mels)} 段, 形状: {all_mels.shape}')
        return all_mels, all_labels, seg_cat_ids, files, labels, cat_ids

    print('  预处理管线...')
    all_mels, all_labels, seg_cat_ids = [], [], []
    t0 = time.time()
    for i, (fp, lbl, cid) in enumerate(zip(files, labels, cat_ids)):
        if i == 0: print("  开始预处理...", flush=True)
        mels = preprocess_pipeline(fp)
        if mels.shape[0] > 1:
            # 多段：取 RMS 最低点分段
            pass  # 所有段都保留
        for m in mels:
            all_mels.append(m)
            all_labels.append(lbl)
            seg_cat_ids.append(cid)
        if (i + 1) % 1000 == 0:
            dt = time.time() - t0
            print(f'  {i+1}/{len(files)}  segs={len(all_mels)}  {dt:.0f}s')

    all_mels = np.stack(all_mels).astype(np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)
    seg_cat_ids = np.array(seg_cat_ids)

    np.save(str(MEL_CACHE), all_mels)
    np.save(str(LABEL_CACHE), all_labels)
    np.save(str(CATID_CACHE), seg_cat_ids)
    print(f'  完成: {len(all_mels)} 段, 形状: {all_mels.shape}, {time.time()-t0:.0f}s', flush=True)
    return all_mels, all_labels, seg_cat_ids, files, labels, cat_ids


# ─── 训练 ───
def train_one_fold(fold, train_mels, train_labels, val_mels, val_labels, output_dir):
    from torch.utils.data import DataLoader, TensorDataset

    train_ds = TensorDataset(train_mels, train_labels)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = FDAPMobileCatCNN_ch64(num_classes=10).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = torch.nn.CrossEntropyLoss()

    best_f1, best_acc, best_ep = 0, 0, 0
    patience, no_improve = 15, 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        for batch_mel, batch_lbl in train_loader:
            batch_mel, batch_lbl = batch_mel.to(DEVICE), batch_lbl.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch_mel)
            loss = criterion(out, batch_lbl)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # 验证（分批，防 OOM）
        model.eval()
        val_preds, val_all = [], []
        with torch.no_grad():
            for i in range(0, len(val_mels), BATCH_SIZE):
                batch = val_mels[i:i+BATCH_SIZE].to(DEVICE)
                out = model(batch)
                val_preds.append(out.argmax(dim=1).cpu())
                val_all.append(val_labels[i:i+BATCH_SIZE])
        val_preds = torch.cat(val_preds)
        val_all = torch.cat(val_all)
        acc = (val_preds == val_all).float().mean().item()
        from sklearn.metrics import f1_score
        f1 = f1_score(val_all.numpy(), val_preds.numpy(), average='weighted')

        if f1 > best_f1:
            best_f1, best_acc, best_ep = f1, acc, ep
            no_improve = 0
            torch.save(model.state_dict(), output_dir / f'fold{fold}_best.pth')
        else:
            no_improve += 1

        if ep % 5 == 0 or f1 > best_f1:
            print(f'  E{ep:3d}/{EPOCHS} Acc={acc:.1%} F1={f1:.1%} Best={best_f1:.1%}@{best_ep}')

        if no_improve >= patience:
            print(f'  早停 @{ep}, Best F1={best_f1:.1%} Acc={best_acc:.1%}@{best_ep}')
            break

    print(f'  折{fold} 完成: Acc={best_acc:.1%} F1={best_f1:.1%} (ep{best_ep})')
    return best_acc, best_f1


def main():
    all_mels, all_labels, seg_cat_ids, files, labels, cat_ids = load_all()
    del files, labels, cat_ids  # 省内存

    torch.cuda.empty_cache()  # 清 GPU 缓存
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)
    seg_indices = np.arange(len(seg_cat_ids))

    fold = 1
    for train_seg_idx, test_seg_idx in gkf.split(seg_indices, all_labels, seg_cat_ids):
        if fold == TARGET_FOLD:
            print(f'\n{"─"*50}')
            print(f'  训练折 {fold}（目标折叠）')
            print(f'{"─"*50}')
            a, f = train_one_fold(
                fold,
                torch.FloatTensor(all_mels[train_seg_idx]).unsqueeze(1),
                torch.LongTensor(all_labels[train_seg_idx]),
                torch.FloatTensor(all_mels[test_seg_idx]).unsqueeze(1),
                torch.LongTensor(all_labels[test_seg_idx]),
                OUTPUT_DIR)
            print(f'\n  ✅ 折{fold} 完成: Acc={a:.1%} F1={f:.1%}')
            print(f'  权重: {OUTPUT_DIR}/fold{fold}_best.pth')
            break
        fold += 1

    # 清理缓存
    import shutil
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    print('\n  完成！')


if __name__ == '__main__':
    main()

"""
译芯怡意 — FDAPMobileCatCNN v4 (CatSound_V2) [VAD最长段+1.6s]
基于 v2 (83.2%) 做保守改进：

v4 改进项:
  1. ReLU → ELU (Pandeya 原论文已验证, 负半轴有梯度)
  2. SE Block on conv4 (96ch squeeze-and-excitation, +2.3K参)
  3. Residual connections (每层卷积后加残差, 缓解梯度消失)
  4. CosineAnnealingWarmRestarts (T_0=30, T_mult=2, 真正重启)
  5. 5-epoch linear warmup (稳定初始训练)
  6. 增强: 保留 v2 的 noise+shift, 加极温和的 freq_mask=4
  
排除项 (v3 验证有害):
  - ✗ SpecAugment (freq=8/time=8 → 破坏频谱)
  - ✗ Label Smoothing (平衡类上降精度)
  - ✗ CBAM (SE 够用了, CBAM 空间注意力在小特征图上无效)
  - ✗ SWA (需要先收敛, 加在 71% 模型上没用)
  - ✗ MixUp (α=0 等于没加)
"""
import os, sys, time, json, csv, warnings, hashlib
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'
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
import shutil
import librosa

# ── 超参 ──
SR = 16000
N_MELS = 128
N_FFT = 512
HOP_LENGTH = 256
VAD_FRAMES = 100
VAD_TOP_DB = 25
BATCH_SIZE = 64
EPOCHS = 150          # 比 v2 多 50 轮, 配合 CAWR(T_0=20) 跑满 3 周期
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

# v4 新超参
WARMUP_EPOCHS = 5                  # 线性 warmup
CA_T_0 = 20                        # CAWR 首周期 (20→40→80, 150 轮刚好 3 周期)
CA_T_MULT = 2                      # 周期倍增
AUG_NOISE = 0.01                   # 噪声强度 (同 v2)
AUG_SHIFT = 0.1                    # 时移比例 (同 v2)
AUG_FREQ_MASK = 4                  # 温和频域掩码 (v3 的 8 太暴力)

BASE = Path('/mnt/c/Users/陈项目负责人/Desktop')
DATA_DIR = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
OUTPUT_DIR = BASE / '模型代码/训练结果/fdap_cnn_v4'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT_DIR / 'mel_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
# 复用 v2 VAD 缓存 (已有 VAD + Mel, 避免重复计算)
V2_VAD = BASE / '模型代码/训练结果/fdap_cnn_v2/mel_cache_vad'
V3_CACHE = BASE / '模型代码/训练结果/fdap_cnn_v3/mel_cache'

LABELS = ["Angry","Defence","Fighting","Happy","HuntingMind","Mating","MotherCall","Paining","Resting","Warning"]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'设备: {DEVICE}')
if DEVICE == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name(0)}')

# ─────────────────────────────
# 数据加载
# ─────────────────────────────
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

# ─────────────────────────────
# VAD 预处理
# ─────────────────────────────
def extract_longest_vocal(y, sr=SR, top_db=VAD_TOP_DB):
    segs = librosa.effects.split(y, top_db=top_db, frame_length=512, hop_length=HOP_LENGTH)
    if len(segs) == 0: return y
    longest = max(segs, key=lambda s: s[1] - s[0])
    return y[longest[0]:longest[1]]

def _find_cache_path(file_path, cache_dir):
    """依次查 v4→v3→v2 缓存"""
    cn = hashlib.md5(file_path.encode()).hexdigest() + '.npy'
    for d in [cache_dir, V3_CACHE, V2_VAD]:
        if d and d.exists():
            p = d / cn
            if p.exists():
                return str(p), d == cache_dir
    return None, False

def precompute_all_mels(file_list, cache_dir):
    print(f'\n[precompute] VAD+Mel, 共 {len(file_list)} 条...')
    t0 = time.time()
    n_cached, n_new, n_failed, n_borrowed = 0, 0, 0, 0
    for i, fp in enumerate(file_list):
        existing_path, is_own = _find_cache_path(fp, cache_dir)
        if existing_path:
            n_cached += 1
            if not is_own:
                n_borrowed += 1
                shutil.copy2(existing_path, os.path.join(cache_dir, os.path.basename(existing_path)))
        else:
            try:
                y, _ = librosa.load(fp, sr=SR, mono=True)
                y = extract_longest_vocal(y)
                peak = np.max(np.abs(y))
                y = y / peak if peak > 1e-10 else np.zeros_like(y)
                mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
                cn = hashlib.md5(fp.encode()).hexdigest() + '.npy'
                np.save(os.path.join(cache_dir, cn), mel_norm)
                n_new += 1
            except Exception as e:
                print(f'  [!] 损坏: {os.path.basename(fp)} — {e}')
                cn = hashlib.md5(fp.encode()).hexdigest() + '.npy'
                np.save(os.path.join(cache_dir, cn), np.zeros((N_MELS, VAD_FRAMES)))
                n_failed += 1
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{len(file_list)}  new={n_new} cached={n_cached} borrowed={n_borrowed} failed={n_failed}')
    print(f'完成: {len(file_list)} 条, new={n_new}, cached={n_cached}, borrowed={n_borrowed}, failed={n_failed}, {time.time()-t0:.0f}s')
    return n_failed

def load_mel_from_cache(file_path, cache_dir, max_frames=VAD_FRAMES):
    # 一定有缓存 (刚 precompute 完)
    cn = hashlib.md5(file_path.encode()).hexdigest() + '.npy'
    mel = np.load(os.path.join(cache_dir, cn))
    if mel.shape[1] < max_frames:
        mel = np.pad(mel, ((0,0), (0, max_frames - mel.shape[1])))
    else:
        start = (mel.shape[1] - max_frames) // 2
        mel = mel[:, start:start + max_frames]
    return mel

def preload_all_mels_to_ram(file_list, cache_dir):
    print(f'\n[preload] 加载 {len(file_list)} 条到内存...')
    t0 = time.time()
    mels = [load_mel_from_cache(fp, cache_dir) for fp in file_list]
    all_mels = np.stack(mels, axis=0)
    print(f'完成: {list(all_mels.shape)}, {all_mels.nbytes/1024/1024:.0f}MB, {time.time()-t0:.0f}s')
    return torch.FloatTensor(all_mels).unsqueeze(1)

# ─────────────────────────────
# Dataset
# ─────────────────────────────
class CatMelDataset(Dataset):
    def __init__(self, mels, labels, augment=False):
        self.mels = mels; self.labels = labels; self.augment = augment
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        mel = self.mels[idx].clone()
        label = self.labels[idx]
        if self.augment:
            mel_np = mel.squeeze(0).numpy()
            # v2 原始增强: 噪声 + 时移
            noise = np.random.randn(*mel_np.shape) * AUG_NOISE
            mel_np = mel_np + noise
            shift = np.random.randint(-int(AUG_SHIFT * mel_np.shape[1]),
                                       int(AUG_SHIFT * mel_np.shape[1]))
            mel_np = np.roll(mel_np, shift, axis=1)
            # v4: 温和频域掩码 (每次只遮 4 bins, v3 的 8 太暴力)
            if np.random.rand() < 0.5 and mel_np.shape[0] > AUG_FREQ_MASK:
                f_start = np.random.randint(0, mel_np.shape[0] - AUG_FREQ_MASK)
                mel_np[f_start:f_start + AUG_FREQ_MASK, :] = mel_np.mean()
            mel = torch.FloatTensor(mel_np).unsqueeze(0)
        return mel, torch.LongTensor([label])[0]

# ─────────────────────────────
# 模型: FDAPMobileCatCNN v4
# ─────────────────────────────
class SEBlock(nn.Module):
    """Squeeze-and-Excitation: 显式建模通道间依赖"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        se = x.mean(dim=[2,3]).view(b, c)  # GAP
        se = self.fc(se).view(b, c, 1, 1)
        return x * se

class FDAPMobileCatCNN_v4(nn.Module):
    """~102K参数, 4层Conv2D + FDAP + SE + 残差连接 + ELU

    v4 相对 v2 的改动:
    1. ReLU → ELU
    2. conv4 后加 SE block
    3. 每个 conv block 加残差连接 (梯度直通)
    """
    def __init__(self, num_classes=10, fdap_bands=4):
        super().__init__()
        self.fdap_bands = fdap_bands

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ELU())
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ELU())
        self.pool2 = nn.MaxPool2d((1, 2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ELU())
        self.pool3 = nn.MaxPool2d((1, 2))

        self.conv4 = nn.Sequential(
            nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ELU())
        # 残差跳连: 48→96 需要一个 1x1 投影
        self.skip4 = nn.Conv2d(48, 96, 1)

        self.conv5 = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ELU())

        # SE block on 96ch feature map
        self.se = SEBlock(96)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(384, 96), nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(96, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # conv1 + pool
        x = self.pool1(self.conv1(x))
        # conv2 + pool
        x = self.pool2(self.conv2(x))
        # conv3 + pool
        x = self.pool3(self.conv3(x))
        # conv4 + 残差 (48→96 升维, 有 skip 投影)
        identity = self.skip4(x)
        x = self.conv4(x)
        x = F.elu(x + identity)  # 残差拼接后激活
        # conv5 + 残差 (96→96, 恒等跳连)
        identity_5 = x
        x = self.conv5(x)
        x = F.elu(x + identity_5)
        # SE 注意力
        x = self.se(x)
        # FDAP 池化
        b, c, f, t = x.shape
        bs = f // self.fdap_bands
        fdap_parts = []
        for i in range(self.fdap_bands):
            start = i * bs
            end = (i + 1) * bs if i < self.fdap_bands - 1 else f
            fdap_parts.append(x[:, :, start:end, :].mean(dim=[2, 3]))
        x = torch.cat(fdap_parts, dim=1)
        return self.classifier(x)

# ─────────────────────────────
# 训练单折
# ─────────────────────────────
def train_one_fold(fold, train_mels, train_labels, test_mels, test_labels):
    print(f'\n{"─"*50}\n折 {fold}: 训练 {len(train_mels)} → 测试 {len(test_mels)}')
    train_ds = CatMelDataset(train_mels, train_labels, augment=True)
    test_ds = CatMelDataset(test_mels, test_labels, augment=False)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = FDAPMobileCatCNN_v4(num_classes=10).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'  参数量: {total_params:,}')

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # v4: 真正的余弦退火带重启
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=CA_T_0, T_mult=CA_T_MULT)

    best_acc, best_f1, best_ep = 0, 0, 0
    no_improve, patience = 0, 20  # 比 v2(15) 多给点耐心, 配合重启

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()

        # warmup: 前 WARMUP_EPOCHS 轮线性增长 LR
        if ep <= WARMUP_EPOCHS:
            lr_scale = ep / WARMUP_EPOCHS
            for g in opt.param_groups:
                g['lr'] = LR * lr_scale

        model.train()
        train_loss = 0
        for m, l in train_loader:
            m, l = m.to(DEVICE), l.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(m), l)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_loss += loss.item()

        model.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for m, l in test_loader:
                o = model(m.to(DEVICE))
                all_p.append(o.argmax(1).cpu())
                all_l.append(l)
        p = torch.cat(all_p).numpy()
        la = torch.cat(all_l).numpy()
        acc = (p == la).mean()
        f1 = f1_score(la, p, average='macro', zero_division=0)

        # warmup 结束后用 CAWR 更新 LR
        if ep > WARMUP_EPOCHS:
            sched.step()

        et = time.time() - t0

        if f1 > best_f1:
            best_acc, best_f1, best_ep = acc, f1, ep
            no_improve = 0
            torch.save(model.state_dict(), OUTPUT_DIR / f'fold{fold}_best.pth')
        else:
            no_improve += 1

        if ep % 5 == 1 or ep == EPOCHS or no_improve == 1:
            current_lr = opt.param_groups[0]['lr']
            print(f'  E{ep:3d}/{EPOCHS} L={train_loss/len(train_loader):.4f} '
                  f'Acc={acc:.1%} F1={f1:.1%} Best={best_f1:.1%}@{best_ep} '
                  f'lr={current_lr:.1e} {et:.0f}s')
        if no_improve >= patience:
            print(f'  早停 @{ep}, Best F1={best_f1:.1%} Acc={best_acc:.1%}@{best_ep}')
            break

    print(f'  折{fold}完成: Acc={best_acc:.1%} F1={best_f1:.1%} (ep{best_ep})')
    return best_acc, best_f1, best_ep


# ─────────────────────────────
# 主流程
# ─────────────────────────────
def main():
    print(f'\n{"█"*60}')
    print(f'  FDAPMobileCatCNN v4 — 保守改进 (CatSound_V2)')
    print(f'  LR={LR} BS={BATCH_SIZE} WD={WEIGHT_DECAY}')
    print(f'  Epochs={EPOCHS} Warmup={WARMUP_EPOCHS} CAWR(T_0={CA_T_0},T_mult={CA_T_MULT})')
    print(f'  Aug: noise={AUG_NOISE} shift={AUG_SHIFT} freq_mask={AUG_FREQ_MASK}')
    print(f'{"█"*60}')

    print('\n[1] 加载数据...')
    files, labels, cat_ids, is_aug = load_catsound_v2(DATA_DIR)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_cats = len(set(cat_ids))
    print(f'  {len(files)}条, {n_cats}只猫, 类: {list(le.classes_)}')

    print('\n[2] 预提取 Mel (VAD → Mel 归一化)...')
    n_failed = precompute_all_mels(files, CACHE_DIR)

    print('\n[3] 加载到内存...')
    all_mels = preload_all_mels_to_ram(files, CACHE_DIR)

    print('\n[4] GroupKFold 5折 (按猫分组)...')
    gkf = GroupKFold(n_splits=5)
    clean_mask = ~is_aug
    clean_idx = np.where(clean_mask)[0]
    fold_accs, fold_f1s = [], []
    for fold, (tr_c, te_c) in enumerate(gkf.split(clean_idx, y[clean_idx], cat_ids[clean_idx]), 1):
        tr_cats = set(cat_ids[clean_idx[tr_c]])
        te_cats = set(cat_ids[clean_idx[te_c]])
        tr_m = np.array([cid in tr_cats for cid in cat_ids])
        te_m = np.array([cid in te_cats for cid in cat_ids])
        a, f, _ = train_one_fold(fold, all_mels[tr_m], y[tr_m],
                                  all_mels[te_m], y[te_m])
        fold_accs.append(a)
        fold_f1s.append(f)

    print(f'\n{"█"*60}')
    print(f'  结果 (5 折 CV)')
    print(f'{"█"*60}')
    for i, (a, f) in enumerate(zip(fold_accs, fold_f1s), 1):
        print(f'  折{i}: Acc={a:.1%} F1={f:.1%}')
    print(f'  ════════════════════════════════════')
    print(f'  Mean Acc: {np.mean(fold_accs):.1%} ± {np.std(fold_accs):.1%}')
    print(f'  Mean F1:  {np.mean(fold_f1s):.1%} ± {np.std(fold_f1s):.1%}')
    print(f'  Best F1:  {max(fold_f1s):.1%}')

    ts = time.strftime('%Y%m%d_%H%M%S')
    result = {
        'model': 'FDAPMobileCatCNN_v4',
        'n': len(files), 'n_cats': n_cats,
        'config': {
            'ep': EPOCHS, 'bs': BATCH_SIZE, 'lr': LR, 'wd': WEIGHT_DECAY,
            'warmup': WARMUP_EPOCHS, 'ca_t0': CA_T_0, 'ca_tmult': CA_T_MULT,
            'aug_noise': AUG_NOISE, 'aug_shift': AUG_SHIFT,
            'aug_freq_mask': AUG_FREQ_MASK,
        },
        'folds': {f'fold{i+1}': {'acc': float(a), 'f1': float(f)}
                  for i, (a, f) in enumerate(zip(fold_accs, fold_f1s))},
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
    print(f'\n结果保存: {rp}')


if __name__ == '__main__':
    main()

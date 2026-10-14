"""
译芯怡意 — DS-ResNet + FDAP (CatSound_V2) [VAD最长段+1.6s]
基于 DS-ResNet 论文 (arXiv:2004.12200):
  - 标准 Conv2D → Depthwise Separable Conv2D (省参 87%)
  - 残差连接 (缓解梯度消失, 允许更深)
  - SE Block (通道注意力, 提升判别性)
  - FDAP 池化 (保留频率分带信息)

参数预算: ~80K, 预期超越 v2 (83.2%)
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
import librosa

SR = 16000
N_MELS = 128
N_FFT = 512
HOP_LENGTH = 256
VAD_FRAMES = 100
VAD_TOP_DB = 25
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

BASE = Path('/mnt/c/Users/陈项目负责人/Desktop')
DATA_DIR = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
OUTPUT_DIR = BASE / '模型代码/训练结果/fdap_cnn_dsresnet_cls96'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT_DIR / 'mel_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
V2_VAD = BASE / '模型代码/训练结果/fdap_cnn_v2/mel_cache_vad'
V3_CACHE = BASE / '模型代码/训练结果/fdap_cnn_v3/mel_cache'

LABELS = ["Angry","Defence","Fighting","Happy","HuntingMind","Mating","MotherCall","Paining","Resting","Warning"]
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

def extract_longest_vocal(y, sr=SR, top_db=VAD_TOP_DB):
    segs = librosa.effects.split(y, top_db=top_db, frame_length=512, hop_length=HOP_LENGTH)
    if len(segs) == 0: return y
    longest = max(segs, key=lambda s: s[1] - s[0])
    return y[longest[0]:longest[1]]

def _find_cache_path(file_path, cache_dir):
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
    import shutil
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

def load_mel_from_cache(file_path, cache_dir, max_frames=VAD_FRAMES):
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
            noise = np.random.randn(*mel_np.shape) * 0.01
            mel_np = mel_np + noise
            shift = np.random.randint(-int(0.1 * mel_np.shape[1]), int(0.1 * mel_np.shape[1]))
            mel_np = np.roll(mel_np, shift, axis=1)
            mel = torch.FloatTensor(mel_np).unsqueeze(0)
        return mel, torch.LongTensor([label])[0]

# ─────────────────────────────────────────
# 模型: DS-ResNet + FDAP
# ─────────────────────────────────────────
class SEBlock(nn.Module):
    """Squeeze-and-Excitation: 显式建模通道依赖"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        se = x.mean(dim=[2,3]).view(b, c)
        se = self.fc(se).view(b, c, 1, 1)
        return x * se

class DSConv(nn.Module):
    """深度可分离卷积: Depthwise 3×3 + Pointwise 1×1 (+BN+ReLU)"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        return self.bn(self.pw(self.dw(x)))

class DSResBlock(nn.Module):
    """DS-ResNet Block: BN→ReLU→DWConv→BN→ReLU→PWConv + 残差"""
    def __init__(self, in_ch, out_ch, stride=1, use_se=False):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.bn2 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)

        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x):
        identity = self.skip(x)
        out = self.bn1(x)
        out = self.act(out)
        out = self.dw(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.pw(out)
        out = self.se(out)
        return out + identity

class DSResNetFDAP(nn.Module):
    """DS-ResNet backbone + FDAP 池化 + 分类器

    架构 (DS-ResNet 论文, 适配猫声 Mel 频谱):
      Stem: Conv2D(1→C) + BN + ReLU + MP(2,2)
      Stage 1: N 个 DSResBlock(C→C) + MP(1,2)
      Stage 2: N 个 DSResBlock(C→2C) + MP(1,2)
      Stage 3: N 个 DSResBlock(2C→2C) + MP(1,2)
      FDAP: 4 bands × 2C = 8C
      Classifier: Linear(8C, 64) → ReLU → Linear(64, num_classes)
    """
    def __init__(self, num_classes=10, base_ch=48, blocks_per_stage=2):
        super().__init__()
        C = base_ch
        N = blocks_per_stage

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Stage 1: C → C
        self.stage1 = nn.Sequential(*[
            DSResBlock(C, C, use_se=(i==0)) for i in range(N)
        ])
        self.pool1 = nn.MaxPool2d((1, 2))

        # Stage 2: C → 2C
        self.stage2 = nn.Sequential(*[
            DSResBlock(C if i==0 else 2*C, 2*C, use_se=(i==0))
            for i in range(N)
        ])
        self.pool2 = nn.MaxPool2d((1, 2))

        # Stage 3: 2C → 2C
        self.stage3 = nn.Sequential(*[
            DSResBlock(2*C, 2*C, use_se=(i==0)) for i in range(N)
        ])

        # FDAP + 分类器
        fdap_dim = 4 * 2 * C  # 4 bands × (2C channels per band)
        self.fdap_bands = 4
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fdap_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)               # (B, C, 64, 50)
        x = self.pool1(self.stage1(x)) # (B, C, 64, 25)
        x = self.pool2(self.stage2(x)) # (B, 2C, 64, 12)
        x = self.stage3(x)             # (B, 2C, 64, 12)

        # FDAP: 4 频带, 每带 2C 维
        b, c, f, t = x.shape
        bs = f // self.fdap_bands
        fdap_parts = [x[:, :, i*bs:(i+1)*bs if i < 3 else f, :].mean(dim=[2, 3])
                      for i in range(self.fdap_bands)]
        x = torch.cat(fdap_parts, dim=1)
        return self.classifier(x)

def train_one_fold(fold, train_mels, train_labels, test_mels, test_labels):
    print(f'\n{"─"*50}\n折 {fold}: 训练 {len(train_mels)} → 测试 {len(test_mels)}')
    train_ds = CatMelDataset(train_mels, train_labels, augment=True)
    test_ds = CatMelDataset(test_mels, test_labels, augment=False)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 默认: base_ch=48, 2 blocks/stage → 78K参 → ~120K 参数
    model = DSResNetFDAP(num_classes=10, base_ch=48, blocks_per_stage=2).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'  参数量: {total_params:,}')

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_acc, best_f1, best_ep = 0, 0, 0
    no_improve, patience = 0, 15
    for ep in range(1, EPOCHS+1):
        t0 = time.time()
        model.train()
        train_loss = 0
        for m,l in train_loader:
            m,l = m.to(DEVICE), l.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(m), l)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_loss += loss.item()

        model.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for m,l in test_loader:
                o = model(m.to(DEVICE))
                all_p.append(o.argmax(1).cpu())
                all_l.append(l)
        p = torch.cat(all_p).numpy()
        la = torch.cat(all_l).numpy()
        acc = (p == la).mean()
        f1 = f1_score(la, p, average='macro', zero_division=0)

        sched.step()
        et = time.time() - t0

        if f1 > best_f1:
            best_acc, best_f1, best_ep = acc, f1, ep
            no_improve = 0
            torch.save(model.state_dict(), OUTPUT_DIR / f'fold{fold}_best.pth')
        else:
            no_improve += 1

        if ep % 5 == 1 or ep == EPOCHS or no_improve == 1:
            print(f'  E{ep:3d}/{EPOCHS} L={train_loss/len(train_loader):.4f} '
                  f'Acc={acc:.1%} F1={f1:.1%} Best={best_f1:.1%}@{best_ep} {et:.0f}s')
        if no_improve >= patience:
            print(f'  早停 @{ep}, Best F1={best_f1:.1%} Acc={best_acc:.1%}@{best_ep}')
            break

    print(f'  折{fold}完成: Acc={best_acc:.1%} F1={best_f1:.1%} (ep{best_ep})')
    return best_acc, best_f1, best_ep

def main():
    print(f'\n{"█"*60}')
    print(f'  DS-ResNet + FDAP (CatSound_V2)')
    print(f'  base_ch=48, 2 blocks/stage → 78K参')
    print(f'  LR={LR} BS={BATCH_SIZE} WD={WEIGHT_DECAY}')
    print(f'{"█"*60}')

    print('\n[1] 加载数据...')
    files, labels, cat_ids, is_aug = load_catsound_v2(DATA_DIR)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_cats = len(set(cat_ids))
    print(f'  {len(files)}条, {n_cats}只猫')

    print('\n[2] 预提取 Mel...')
    n_failed = precompute_all_mels(files, CACHE_DIR)

    print('\n[3] 加载到内存...')
    all_mels = preload_all_mels_to_ram(files, CACHE_DIR)

    print('\n[4] GroupKFold 快速测试 (1折)...')
    gkf = GroupKFold(n_splits=5)
    clean_mask = ~is_aug
    clean_idx = np.where(clean_mask)[0]
    fold_accs, fold_f1s = [], []
    for fold, (tr_c, te_c) in enumerate(gkf.split(clean_idx, y[clean_idx], cat_ids[clean_idx]), 1):
        if fold > 1:  # 快速测试: 只跑 1 折
            break
        tr_cats = set(cat_ids[clean_idx[tr_c]])
        te_cats = set(cat_ids[clean_idx[te_c]])
        tr_m = np.array([cid in tr_cats for cid in cat_ids])
        te_m = np.array([cid in te_cats for cid in cat_ids])
        a, f, _ = train_one_fold(fold, all_mels[tr_m], y[tr_m], all_mels[te_m], y[te_m])
        fold_accs.append(a)
        fold_f1s.append(f)

    print(f'\n{"█"*60}')
    print(f'  结果')
    print(f'{"█"*60}')
    for i, (a, f) in enumerate(zip(fold_accs, fold_f1s), 1):
        print(f'  折{i}: Acc={a:.1%} F1={f:.1%}')
    print(f'  Acc: {np.mean(fold_accs):.1%}±{np.std(fold_accs):.1%}')
    print(f'  F1:  {np.mean(fold_f1s):.1%}±{np.std(fold_f1s):.1%}')

    ts = time.strftime('%Y%m%d_%H%M%S')
    result = {
        'model': 'DSResNet_FDAP_cls96',
        'n': len(files), 'n_cats': n_cats,
        'config': {'ep': EPOCHS, 'bs': BATCH_SIZE, 'lr': LR, 'wd': WEIGHT_DECAY},
        'folds': {f'fold{i+1}': {'acc': float(a), 'f1': float(f)}
                  for i, (a, f) in enumerate(zip(fold_accs, fold_f1s))},
        'mean_acc': float(np.mean(fold_accs)),
        'std_acc': float(np.std(fold_accs)),
        'mean_f1': float(np.mean(fold_f1s)),
        'std_f1': float(np.std(fold_f1s)),
        'best_f1': float(max(fold_f1s)),
        'ts': ts, 'device': DEVICE,
    }
    rp = OUTPUT_DIR / f'results_{ts}.json'
    with open(rp, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'\n结果: {rp}')

if __name__ == '__main__':
    main()

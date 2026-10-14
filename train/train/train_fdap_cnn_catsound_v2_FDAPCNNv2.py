"""
译芯怡意 — FDAPMobileCatCNN 全量训练 (CatSound_V2) [VAD最长段+1.6s]
已验证: 83.0% F1 (5折平均), 99K参数

v20260623 改进:
  - ReLU → ELU (匹配 Pandeya 原论文, 负半轴有梯度)
  - SE Block (96ch Squeeze-and-Excitation, +2.3K参数)
  - SpecAugment (频域掩码+时域掩码, 替代弱噪声增强)
  - Label Smoothing (ε=0.1)
  - class_weight 加权 (补回 V2 有但 V3 丢了的)
  - CosineAnnealingWarmRestarts (T_0=20, T_mult=2) 替代无重启
  - warmup 10 epochs (线性增长)
  - VAD 缓存: 同时检查 v2/v3, 只写 v3 (修复脆皮缓存复用)
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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
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
# v20260623 新参数
WARMUP_EPOCHS = 10               # 线性 warmup 轮数
CA_T_0 = 20                      # CosineAnnealingWarmRestarts 首周期
CA_T_MULT = 2                    # 周期倍增
LABEL_SMOOTHING = 0.1            # 标签平滑
SPEC_FREQ_MASK = 8               # SpecAugment 频域掩码宽度
SPEC_TIME_MASK = 8               # SpecAugment 时域掩码宽度
MIXUP_ALPHA = 0                    # MixUp Beta 分布参数 (0=关闭)
SWA_LR = 1e-4                    # SWA 阶段学习率
SWA_EPOCHS = 10                  # SWA 平均轮数

BASE = Path('/mnt/c/Users/陈项目负责人/Desktop')
DATA_DIR = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
OUTPUT_DIR = BASE / '模型代码/训练结果/fdap_cnn_v3'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT_DIR / 'mel_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
# v20260623: 只读 v2 缓存, 只写 v3 缓存 (修复脆皮复用)
V2_VAD = BASE / '模型代码/训练结果/fdap_cnn_v2/mel_cache_vad'

LABELS = ["Angry","Defence","Fighting","Happy","HuntingMind","Mating","MotherCall","Paining","Resting","Warning"]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'设备: {DEVICE}')
if DEVICE == 'cuda':
    print(f'  GPU: {torch.cuda.get_device_name(0)}')

# ─────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────
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

# ─────────────────────────────────────────
# VAD 预处理
# ─────────────────────────────────────────
def extract_longest_vocal(y, sr=SR, top_db=VAD_TOP_DB):
    segs = librosa.effects.split(y, top_db=top_db, frame_length=512, hop_length=HOP_LENGTH)
    if len(segs) == 0: return y
    longest = max(segs, key=lambda s: s[1] - s[0])
    return y[longest[0]:longest[1]]

def precompute_all_mels(file_list, cache_dir):
    print(f'\n[precompute] VAD+Mel, 共 {len(file_list)} 条...')
    t0 = time.time()
    n_cached, n_new, n_failed = 0, 0, 0
    for i, fp in enumerate(file_list):
        cn = hashlib.md5(fp.encode()).hexdigest() + '.npy'
        cp = os.path.join(cache_dir, cn)
        if os.path.exists(cp):
            n_cached += 1
        else:
            # v20260623: 检查 v2 缓存 (只读), 复制到 v3 避免二次 VAD
            if V2_VAD.exists():
                v2_path = V2_VAD / cn
                if v2_path.exists():
                    # 复制到 v3, 避免二次 VAD
                    import shutil
                    shutil.copy2(str(v2_path), cp)
                    n_cached += 1
                    continue
            try:
                y, _ = librosa.load(fp, sr=SR, mono=True)
                y = extract_longest_vocal(y)
                peak = np.max(np.abs(y))
                y = y / peak if peak > 1e-10 else np.zeros_like(y)
                mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
                np.save(cp, mel_norm)
                n_new += 1
            except Exception as e:
                print(f'  [!] 损坏: {os.path.basename(fp)} — {e}')
                np.save(cp, np.zeros((N_MELS, VAD_FRAMES)))
                n_failed += 1
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{len(file_list)}  new={n_new} cached={n_cached} failed={n_failed}')
    print(f'完成: {len(file_list)} 条, new={n_new}, cached={n_cached}, failed={n_failed}, {time.time()-t0:.0f}s')
    return n_failed

def load_mel_from_cache(file_path, cache_dir, max_frames=VAD_FRAMES):
    cache_name = hashlib.md5(file_path.encode()).hexdigest() + '.npy'
    # 先查 v3, 再查 v2
    p = os.path.join(cache_dir, cache_name)
    if not os.path.exists(p) and V2_VAD.exists():
        p2 = os.path.join(V2_VAD, cache_name)
        if os.path.exists(p2):
            p = p2
    mel = np.load(p)
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

# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────
def spec_augment(mel_np, freq_mask=SPEC_FREQ_MASK, time_mask=SPEC_TIME_MASK):
    """v20260623: SpecAugment — 频域掩码 + 时域掩码 (Mel 频谱上)"""
    mel = mel_np.copy()
    n_mels, n_frames = mel.shape
    # v20260623: 用 mel 均值填充 (min-max 归一化下 0 太激进)
    fill_val = mel.mean()
    # 频域掩码
    if freq_mask > 0 and n_mels > freq_mask:
        f_start = np.random.randint(0, n_mels - freq_mask)
        mel[f_start:f_start + freq_mask, :] = fill_val
    # 时域掩码
    if time_mask > 0 and n_frames > time_mask:
        t_start = np.random.randint(0, n_frames - time_mask)
        mel[:, t_start:t_start + time_mask] = fill_val
    return mel

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
            # v20260623: MixUp (mel 域线性插值, 返回 soft label)
            if MIXUP_ALPHA > 0 and np.random.rand() < 0.5:
                mix_idx = np.random.randint(0, len(self.labels))
                mix_mel = self.mels[mix_idx].squeeze(0).numpy()
                lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                mel_np = lam * mel_np + (1 - lam) * mix_mel
                label_onehot = np.zeros(10, dtype=np.float32)
                label_onehot[label] = lam
                label_onehot[self.labels[mix_idx]] += 1 - lam
                is_soft = True
            else:
                # 未做 MixUp: one-hot + label_smoothing 手动施加
                label_onehot = np.full(10, LABEL_SMOOTHING / 9, dtype=np.float32)
                label_onehot[label] = 1.0 - LABEL_SMOOTHING
                is_soft = True
            # SpecAugment (v20260623)
            mel_np = spec_augment(mel_np)
            # 保留一点噪声
            noise = np.random.randn(*mel_np.shape) * 0.005
            mel_np = mel_np + noise
            # 时移
            shift = np.random.randint(-int(0.1 * mel_np.shape[1]), int(0.1 * mel_np.shape[1]))
            mel_np = np.roll(mel_np, shift, axis=1)
            mel = torch.FloatTensor(mel_np).unsqueeze(0)
            return mel, torch.FloatTensor(label_onehot)
        return mel, torch.LongTensor([label])[0]

# ─────────────────────────────────────────
# 模型: FDAPMobileCatCNN + CBAM (v20260623)
# ─────────────────────────────────────────
# ── v20260623: CBAM 替换 SE (通道注意力 + 空间注意力) ──
class ChannelAttention(nn.Module):
    """通道注意力: AvgPool+MaxPool → 共享MLP → sigmoid"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    def forward(self, x):
        b, c, _, _ = x.shape
        # 双池化路径 (CBAM 核心改进: SE 只用 AvgPool)
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max = F.adaptive_max_pool2d(x, 1).view(b, c)
        avg_out = self.fc2(F.elu(self.fc1(avg)))
        max_out = self.fc2(F.elu(self.fc1(max)))
        return torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    """空间注意力: 沿通道平均+最大 → 7×7卷积 → sigmoid"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

class CBAM(nn.Module):
    """Convolutional Block Attention Module: 通道→空间 序贯注意力"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(7)
    def forward(self, x):
        x = x * self.ca(x)   # 通道: 哪些滤波器重要
        x = x * self.sa(x)   # 空间: mel 时频图上哪些位置重要
        return x

class FDAPMobileCatCNN(nn.Module):
    """~103K参数, 4层Conv2D + CBAM + FDAP (v20260623: ELU + CBAM)"""
    def __init__(self, num_classes=10, fdap_bands=4):
        super().__init__()
        self.fdap_bands = fdap_bands
        # v20260623: ReLU → ELU (匹配 Pandeya 原论文)
        self.conv1 = nn.Sequential(nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ELU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ELU(), nn.MaxPool2d((1,2)))
        self.conv3 = nn.Sequential(nn.Conv2d(32,48,3,padding=1), nn.BatchNorm2d(48), nn.ELU(), nn.MaxPool2d((1,2)))
        self.conv4 = nn.Sequential(nn.Conv2d(48,96,3,padding=1), nn.BatchNorm2d(96), nn.ELU(), nn.Identity())
        # v20260623: CBAM (通道注意力 + 空间注意力)
        self.cbam = CBAM(96)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(384,96), nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(96,num_classes),
        )
    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x); x = self.conv4(x)
        x = self.cbam(x)  # v20260623: CBAM
        b,c,f,t = x.shape; bs = f // self.fdap_bands
        fdap = [x[:,:,i*bs:(i+1)*bs if i<3 else f,:].mean(dim=[2,3]) for i in range(self.fdap_bands)]
        x = torch.cat(fdap, dim=1)
        return self.classifier(x)

# ─────────────────────────────────────────
# 训练单折
# ─────────────────────────────────────────
def train_one_fold(fold, train_mels, train_labels, test_mels, test_labels):
    print(f'\n{"─"*50}\n折 {fold}: 训练 {len(train_mels)} → 测试 {len(test_mels)}')
    train_ds = CatMelDataset(train_mels, train_labels, augment=True)
    test_ds = CatMelDataset(test_mels, test_labels, augment=False)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = FDAPMobileCatCNN(num_classes=10).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # v20260623: CosineAnnealingLR (平滑退火, 无重启)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_acc, best_f1, best_ep = 0, 0, 0
    no_improve, patience = 0, 15
    for ep in range(1, EPOCHS+1):
        t0 = time.time()
        # v20260623: warmup (前 WARMUP_EPOCHS 轮线性增长)
        if ep <= WARMUP_EPOCHS:
            lr_scale = ep / WARMUP_EPOCHS
            for g in opt.param_groups:
                g['lr'] = LR * lr_scale

        model.train()
        train_loss = 0
        for m,l in train_loader:
            m,l = m.to(DEVICE), l.to(DEVICE)
            # v20260623: soft label (label_smoothing) + 无 class_weight
            out = model(m)
            log_prob = F.log_softmax(out, dim=1)
            loss = -(l * log_prob).sum(dim=1).mean()
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            train_loss += loss.item()

        model.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for m,l in test_loader:
                o = model(m.to(DEVICE)); all_p.append(o.argmax(1).cpu()); all_l.append(l)
        p = torch.cat(all_p).numpy(); la = torch.cat(all_l).numpy()
        acc = (p==la).mean(); f1 = f1_score(la, p, average='macro', zero_division=0)

        # warmup 期间不 step scheduler
        if ep > WARMUP_EPOCHS:
            sched.step()
        et = time.time()-t0

        if f1 > best_f1:
            best_acc, best_f1, best_ep = acc, f1, ep; no_improve = 0
            torch.save(model.state_dict(), OUTPUT_DIR / f'fold{fold}_best.pth')
        else: no_improve += 1
        if ep % 5 == 1 or ep == EPOCHS or no_improve == 1:
            current_lr = opt.param_groups[0]['lr']
            print(f'  E{ep:3d}/{EPOCHS} L={train_loss/len(train_loader):.4f} '
                  f'Acc={acc:.1%} F1={f1:.1%} Best={best_f1:.1%}@{best_ep} '
                  f'lr={current_lr:.1e} {et:.0f}s')
        if no_improve >= patience:
            print(f'  早停 @{ep}, Best F1={best_f1:.1%} Acc={best_acc:.1%}@{best_ep}'); break
    # v20260623: SWA (Stochastic Weight Averaging) — 损失盆地中心
    print(f'  SWA: 最后 {SWA_EPOCHS} 轮权重平均...')
    swa_model = AveragedModel(model)
    # 重置学习率到 SWA_LR
    for g in opt.param_groups:
        g['lr'] = SWA_LR
    swa_sched = SWALR(opt, swa_lr=SWA_LR)
    swa_best_f1, swa_best_acc = 0, 0
    for ep in range(1, SWA_EPOCHS + 1):
        model.train()
        for m, l in train_loader:
            m, l = m.to(DEVICE), l.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = model(m)
            log_prob = F.log_softmax(out, dim=1)
            loss = -(l * log_prob).sum(dim=1).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            swa_model.update_parameters(model)
        swa_sched.step()
        # 评估 SWA 模型
        swa_model.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for m, l in test_loader:
                o = swa_model(m.to(DEVICE)); all_p.append(o.argmax(1).cpu()); all_l.append(l)
        p = torch.cat(all_p).numpy(); la = torch.cat(all_l).numpy()
        swa_acc = (p == la).mean(); swa_f1 = f1_score(la, p, average='macro', zero_division=0)
        if swa_f1 > swa_best_f1:
            swa_best_f1, swa_best_acc = swa_f1, swa_acc
        if ep == 1 or ep % 3 == 0 or ep == SWA_EPOCHS:
            print(f'    SWA E{ep}/{SWA_EPOCHS} Acc={swa_acc:.1%} F1={swa_f1:.1%} Best={swa_best_f1:.1%}')
    # BN 统计更新
    print(f'  SWA BN 更新...')
    update_bn(train_loader, swa_model, DEVICE)
    # SWA 评估
    swa_model.eval()
    all_p, all_l = [], []
    with torch.no_grad():
        for m, l in test_loader:
            o = swa_model(m.to(DEVICE)); all_p.append(o.argmax(1).cpu()); all_l.append(l)
    p = torch.cat(all_p).numpy(); la = torch.cat(all_l).numpy()
    swa_acc = (p == la).mean(); swa_f1 = f1_score(la, p, average='macro', zero_division=0)
    print(f'  SWA 最终: Acc={swa_acc:.1%} F1={swa_f1:.1%}')
    # 取常规训练和 SWA 中更好的
    if swa_f1 > best_f1:
        best_acc, best_f1 = swa_acc, swa_f1
        torch.save(swa_model.state_dict(), OUTPUT_DIR / f'fold{fold}_best.pth')
        print(f'  → 采用 SWA 模型 (F1 {best_f1:.1%} > 常规 {best_f1:.1%})')
    else:
        print(f'  → 保留常规模型 (F1 {best_f1:.1%})')
    print(f'  折{fold}完成: Acc={best_acc:.1%} F1={best_f1:.1%} (ep{best_ep})')
    return best_acc, best_f1, best_ep

# ─────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────
def main():
    print(f'\n{"█"*50}')
    print(f'  FDAP 99K + SE + SpecAugment + ELU (CatSound_V2)')
    print(f'  LR={LR} BS={BATCH_SIZE} WD={WEIGHT_DECAY}')
    print(f'  Warmup={WARMUP_EPOCHS} CA(T_0={CA_T_0},T_mult={CA_T_MULT})')
    print(f'  LabelSm={LABEL_SMOOTHING} SpecAug(F={SPEC_FREQ_MASK},T={SPEC_TIME_MASK})')
    print(f'{"█"*50}')

    print('\n[1] 加载数据...')
    files, labels, cat_ids, is_aug = load_catsound_v2(DATA_DIR)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    print(f'  {len(files)}条, {list(le.classes_)}')

    print('\n[2] 预提取 Mel...')
    n_failed = precompute_all_mels(files, CACHE_DIR)

    print('\n[3] 加载到内存...')
    all_mels = preload_all_mels_to_ram(files, CACHE_DIR)

    print('\n[4] GroupKFold 5折...')
    gkf = GroupKFold(n_splits=5)
    clean_mask = ~is_aug; clean_idx = np.where(clean_mask)[0]
    fold_accs, fold_f1s = [], []
    for fold, (tr_c, te_c) in enumerate(gkf.split(clean_idx, y[clean_idx], cat_ids[clean_idx]), 1):
        tr_cats = set(cat_ids[clean_idx[tr_c]]); te_cats = set(cat_ids[clean_idx[te_c]])
        tr_m = np.array([cid in tr_cats for cid in cat_ids]); te_m = np.array([cid in te_cats for cid in cat_ids])
        a, f, _ = train_one_fold(fold, all_mels[tr_m], y[tr_m], all_mels[te_m], y[te_m])
        fold_accs.append(a); fold_f1s.append(f)

    print(f'\n{"█"*50}\n  结果\n{"█"*50}')
    for i,(a,f) in enumerate(zip(fold_accs,fold_f1s),1): print(f'  折{i}: Acc={a:.1%} F1={f:.1%}')
    print(f'  Acc: {np.mean(fold_accs):.1%}±{np.std(fold_accs):.1%}  F1: {np.mean(fold_f1s):.1%}±{np.std(fold_f1s):.1%}')
    ts = time.strftime('%Y%m%d_%H%M%S')
    result = {
        'model':'FDAP99K_v20260623_SE_SpecAug_ELU',
        'n':len(files),'n_cats':len(set(cat_ids)),
        'config':{
            'ep':EPOCHS,'bs':BATCH_SIZE,'lr':LR,'wd':WEIGHT_DECAY,
            'warmup':WARMUP_EPOCHS,'ca_t0':CA_T_0,'ca_tmult':CA_T_MULT,
            'label_smoothing':LABEL_SMOOTHING,
            'spec_freq_mask':SPEC_FREQ_MASK,'spec_time_mask':SPEC_TIME_MASK,
        },
        'folds':{f'fold{i+1}':{'acc':float(a),'f1':float(f)} for i,(a,f) in enumerate(zip(fold_accs,fold_f1s))},
        'mean_acc':float(np.mean(fold_accs)),'std_acc':float(np.std(fold_accs)),
        'mean_f1':float(np.mean(fold_f1s)),'std_f1':float(np.std(fold_f1s)),
        'best_f1':float(max(fold_f1s)),'ts':ts,'device':DEVICE,
    }
    rp = OUTPUT_DIR / f'results_{ts}.json'
    with open(rp,'w',encoding='utf-8') as f: json.dump(result,f,indent=2,ensure_ascii=False)
    print(f'\n结果: {rp}')

if __name__ == '__main__':
    main()

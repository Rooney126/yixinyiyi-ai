"""
译芯怡意 — FDAPMobileCatCNN 全量训练 (CatSound_V2) [VAD最长段+1.6s]
conv4=64ch 搜索: 验证通道数对精度的影响
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
import librosa

SR = 16000
N_MELS = 128
N_FFT = 512
HOP_LENGTH = 256
VAD_FRAMES = 100
VAD_TOP_DB = 25
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

BASE = Path('/mnt/c/Users/陈项目负责人/Desktop')
DATA_DIR = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
OUTPUT_DIR = BASE / '模型代码/训练结果/fdap_cnn_ch64_aug'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT_DIR / 'mel_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
# 复用 v2 的 VAD 缓存
V2_VAD = BASE / '模型代码/训练结果/fdap_cnn_v2/mel_cache_vad'
if V2_VAD.exists():
    print(f'复用 v2 VAD 缓存: {V2_VAD}')
    CACHE_DIR = V2_VAD

# Speed/Pitch 增强缓存目录
AUG_CACHE_DIR = BASE / '模型代码/训练结果/aug_sp_cache'

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
    mel = np.load(os.path.join(cache_dir, cache_name))
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
        if self.augment:
            mel_np = mel.squeeze(0).numpy()
            noise = np.random.randn(*mel_np.shape) * 0.01
            mel_np = mel_np + noise
            shift = np.random.randint(-int(0.1 * mel_np.shape[1]), int(0.1 * mel_np.shape[1]))
            mel_np = np.roll(mel_np, shift, axis=1)
            mel = torch.FloatTensor(mel_np).unsqueeze(0)
        return mel, torch.LongTensor([self.labels[idx]])[0]

class FDAPMobileCatCNN(nn.Module):
    """conv4=64ch, FDAP=4×64=256维"""
    def __init__(self, num_classes=10, fdap_bands=4):
        super().__init__()
        self.fdap_bands = fdap_bands
        self.conv1 = nn.Sequential(nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((1,2)))
        self.conv3 = nn.Sequential(nn.Conv2d(32,48,3,padding=1), nn.BatchNorm2d(48), nn.ReLU(), nn.MaxPool2d((1,2)))
        self.conv4 = nn.Sequential(nn.Conv2d(48,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Identity())
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256,96), nn.ReLU(), nn.Dropout(0.3), nn.Linear(96,num_classes))
    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x); x = self.conv4(x)
        b,c,f,t = x.shape; bs = f // self.fdap_bands
        fdap = [x[:,:,i*bs:(i+1)*bs if i<3 else f,:].mean(dim=[2,3]) for i in range(self.fdap_bands)]
        x = torch.cat(fdap, dim=1)
        return self.classifier(x)

def train_one_fold(fold, train_mels, train_labels, test_mels, test_labels):
    print(f'\n{"─"*50}\n折 {fold}: 训练 {len(train_mels)} → 测试 {len(test_mels)}')
    train_ds = CatMelDataset(train_mels, train_labels, augment=True)
    test_ds = CatMelDataset(test_mels, test_labels, augment=False)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    model = FDAPMobileCatCNN(num_classes=10).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_acc, best_f1, best_ep = 0, 0, 0
    no_improve, patience = 0, 10
    for ep in range(1, EPOCHS+1):
        t0 = time.time()
        model.train()
        train_loss = 0
        for m,l in train_loader:
            m,l = m.to(DEVICE), l.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(m), l)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            train_loss += loss.item()
        model.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for m,l in test_loader:
                o = model(m.to(DEVICE)); all_p.append(o.argmax(1).cpu()); all_l.append(l)
        p = torch.cat(all_p).numpy(); la = torch.cat(all_l).numpy()
        acc = (p==la).mean(); f1 = f1_score(la, p, average='macro', zero_division=0)
        sched.step(); et = time.time()-t0
        if f1 > best_f1:
            best_acc, best_f1, best_ep = acc, f1, ep; no_improve = 0
            torch.save(model.state_dict(), OUTPUT_DIR / f'fold{fold}_best.pth')
        else: no_improve += 1
        if ep % 5 == 1 or ep == EPOCHS or no_improve == 1:
            print(f'  E{ep:3d}/{EPOCHS} L={train_loss/len(train_loader):.4f} Acc={acc:.1%} F1={f1:.1%} Best={best_f1:.1%}@{best_ep} {et:.0f}s')
        if no_improve >= patience:
            print(f'  早停 @{ep}, Best F1={best_f1:.1%} Acc={best_acc:.1%}@{best_ep}'); break
    print(f'  折{fold}完成: Acc={best_acc:.1%} F1={best_f1:.1%} (ep{best_ep})')
    return best_acc, best_f1, best_ep

def main():
    print(f'\n{"█"*50}\n  FDAP 64ch VAD 1.6s (CatSound_V2)\n  LR={LR} BS={BATCH_SIZE} WD={WEIGHT_DECAY}\n{"█"*50}')
    print('\n[1] 加载数据...')
    files, labels, cat_ids, is_aug = load_catsound_v2(DATA_DIR)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    print(f'  {len(files)}条, {list(le.classes_)}')
    print('\n[2] 预提取 Mel...')
    n_failed = precompute_all_mels(files, CACHE_DIR)
    print('\n[3] 加载到内存...')
    all_mels = preload_all_mels_to_ram(files, CACHE_DIR)
    print('\n[4] 加载增强数据...')
    # 构建 hash → cat_id 映射（从原始文件路径hash推导）
    hash_to_catid = {}
    for i, fp in enumerate(files):
        h = hashlib.md5(fp.encode()).hexdigest()
        hash_to_catid[h] = cat_ids[i]
    
    aug_list_path = AUG_CACHE_DIR / 'aug_file_list.txt'
    aug_mels_list = []
    aug_labels_list = []
    aug_catids_list = []
    with open(aug_list_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                cache_name = parts[0]
                label_idx = int(parts[1])
                # 从缓存文件名提取原始文件hash：hash_augname.npy → hash
                base_hash = cache_name.split('_')[0]
                orig_catid = hash_to_catid.get(base_hash, '')
                if not orig_catid:
                    continue  # 找不到映射，跳过
                mel_path = AUG_CACHE_DIR / cache_name
                if mel_path.exists():
                    mel = np.load(str(mel_path))
                    aug_mels_list.append(mel)
                    aug_labels_list.append(label_idx)
                    aug_catids_list.append(orig_catid)
    aug_mels_all = torch.FloatTensor(np.stack(aug_mels_list, axis=0)).unsqueeze(1)
    aug_labels_all = torch.LongTensor(aug_labels_list)
    aug_catids_all = np.array(aug_catids_list)
    print(f'  增强数据总计: {len(aug_mels_list)} 条')
    
    print('\n[5] GroupKFold 5折 (含增强训练)...')
    gkf = GroupKFold(n_splits=5)
    clean_mask = ~is_aug; clean_idx = np.where(clean_mask)[0]
    fold_accs, fold_f1s = [], []
    for fold, (tr_c, te_c) in enumerate(gkf.split(clean_idx, y[clean_idx], cat_ids[clean_idx]), 1):
        tr_cats = set(cat_ids[clean_idx[tr_c]]); te_cats = set(cat_ids[clean_idx[te_c]])
        tr_m = np.array([cid in tr_cats for cid in cat_ids]); te_m = np.array([cid in te_cats for cid in cat_ids])
        # 只取训练集猫的增强数据（避免数据泄露）
        aug_mask = np.array([cid in tr_cats for cid in aug_catids_all])
        train_combined = torch.cat([all_mels[tr_m], aug_mels_all[aug_mask]], dim=0)
        labels_combined = torch.cat([torch.LongTensor(y[tr_m]), aug_labels_all[aug_mask]], dim=0)
        a, f, _ = train_one_fold(fold, train_combined, labels_combined, all_mels[te_m], y[te_m])
        fold_accs.append(a); fold_f1s.append(f)
    print(f'\n{"█"*50}\n  结果\n{"█"*50}')
    for i,(a,f) in enumerate(zip(fold_accs,fold_f1s),1): print(f'  折{i}: Acc={a:.1%} F1={f:.1%}')
    print(f'  Acc: {np.mean(fold_accs):.1%}±{np.std(fold_accs):.1%}  F1: {np.mean(fold_f1s):.1%}±{np.std(fold_f1s):.1%}')
    ts = time.strftime('%Y%m%d_%H%M%S')
    result = {'model':'FDAP64ch_VAD1.6s','n':len(files),'n_cats':len(set(cat_ids)),
        'config':{'ep':EPOCHS,'bs':BATCH_SIZE,'lr':LR,'wd':WEIGHT_DECAY},
        'folds':{f'fold{i+1}':{'acc':float(a),'f1':float(f)} for i,(a,f) in enumerate(zip(fold_accs,fold_f1s))},
        'mean_acc':float(np.mean(fold_accs)),'std_acc':float(np.std(fold_accs)),
        'mean_f1':float(np.mean(fold_f1s)),'std_f1':float(np.std(fold_f1s)),
        'best_f1':float(max(fold_f1s)),'ts':ts,'device':DEVICE}
    rp = OUTPUT_DIR / f'results_{ts}.json'
    with open(rp,'w',encoding='utf-8') as f: json.dump(result,f,indent=2,ensure_ascii=False)
    print(f'\n结果: {rp}')

if __name__ == '__main__':
    main()

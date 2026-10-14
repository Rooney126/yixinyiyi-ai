"""
重训折2 (patience=20)
读取已有缓存 → GroupKFold取折2 → train_one_fold
"""
import sys, json, time, csv, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/mnt/c/Users/陈项目负责人/Desktop/模型代码')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from train_cnn_pipeline_v2 import (
    load_catsound_v2, preprocess_all, CatMelDataset, FDAPMobileCatCNN,
    train_one_fold, SR, N_MELS, N_FFT, HOP_LENGTH, MEL_FRAMES,
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, NUM_WORKERS, PATIENCE,
    CLIP_GRAD, LABELS, DEVICE, DATA_DIR, CACHE_DIR, OUTPUT_DIR
)

# 重设 patience
PATIENCE = 20

def main():
    print(f'\n{"█"*50}')
    print(f'  重训折2 (patience={PATIENCE})')
    print(f'{"█"*50}')
    
    # 加载数据 (会读取已有缓存)
    files, labels, cat_ids, is_aug = load_catsound_v2(DATA_DIR)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    all_mels, all_labels, _, seg_cat_ids, seg_file_idxs = preprocess_all(
        files, y, cat_ids, CACHE_DIR)
    if all_mels is None:
        print('无数据')
        return
    
    # GroupKFold 取折2
    gkf = GroupKFold(n_splits=5)
    seg_indices = np.arange(len(seg_cat_ids))
    
    for fold, (tr, te) in enumerate(gkf.split(seg_indices, all_labels, seg_cat_ids), 1):
        if fold == 2:
            print(f'\n重训折{fold}...')
            a, f, ep = train_one_fold(
                fold,
                torch.FloatTensor(all_mels[tr]).unsqueeze(1),
                torch.LongTensor(all_labels[tr]),
                torch.LongTensor(seg_file_idxs[tr]),
                torch.FloatTensor(all_mels[te]).unsqueeze(1),
                torch.LongTensor(all_labels[te]),
                torch.LongTensor(seg_file_idxs[te]),
                OUTPUT_DIR)
            print(f'\n折2重训完成: Acc={a:.1%} F1={f:.1%} (ep{ep})')
            break

if __name__ == '__main__':
    main()

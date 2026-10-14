"""
重训折2 (patience=20)
读取已有缓存 → GroupKFold取折2 → train_one_fold
"""
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Desktop/模型代码')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from train_cnn_pipeline_v2 import (
    CACHE_DIR,
    DATA_DIR,
    OUTPUT_DIR,
    PATIENCE,
    load_catsound_v2,
    preprocess_all,
    train_one_fold,
)

# 覆盖 patience：本脚本专用于 fold2 重训，用更宽松的早停
PATIENCE = 20  # noqa: F811 有意覆盖导入值

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

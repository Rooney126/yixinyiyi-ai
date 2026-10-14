"""
FDAP + SVM 分类实验
从缓存加载 FDAP/GAP 特征 → SVM 交叉验证 → 对比准确率
# FDAP/GAP 特征 + SVM 5折交叉验证 → 对比准确率
"""

import json
import sys
import warnings

warnings.filterwarnings('ignore')
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from pathlib import Path

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

# ── 路径 ──
BASE = Path(r'/path/to/dataset')
CACHE_DIR = BASE / '模型代码/fdap_cache'
OUTPUT = BASE / '模型代码/结果_FDAP_SVM对比.json'

# 找最新缓存的 FDAP/GAP 特征
cache_files = sorted(CACHE_DIR.glob('*fdap_*.npy'))
print(f'找到 {len(cache_files)} 个缓存文件')
for f in cache_files:
    print(f'  {f.name}')

# 取最新的
fdap_path = sorted(CACHE_DIR.glob('*_fdap_*.npy'))[-1]
gap_path = sorted(CACHE_DIR.glob('*_gap_*.npy'))[-1]
label_path = sorted(CACHE_DIR.glob('*_labels_*.npy'))[-1]

print(f'\n加载: {fdap_path.name}')
print(f'     : {gap_path.name}')
print(f'     : {label_path.name}')

fdap_feats = np.load(fdap_path)
gap_feats = np.load(gap_path)
labels = np.load(label_path).tolist()

print(f'FDAP: {fdap_feats.shape}')
print(f'GAP:  {gap_feats.shape}')
print(f'标签: {len(labels)} 条')

# ── 标签统计 ──
from collections import Counter

for l, c in Counter(labels).most_common():
    print(f'  {l:<12}: {c}')

# ── 10类 SVM 交叉验证 ──
print('\n' + '='*50)
print('10类 SVM 交叉验证 (5折)')
print('='*50)

le = LabelEncoder()
y = le.fit_transform(labels)

for name, feats in [('FDAP', fdap_feats), ('GAP', gap_feats)]:
    # LinearSVM + 标准化
    pipe = make_pipeline(
        StandardScaler(),
        LinearSVC(C=1.0, max_iter=10000, dual='auto', random_state=42)
    )
    scores = cross_val_score(pipe, feats, y, cv=5, scoring='accuracy')
    print(f'\n  {name} (dim={feats.shape[1]}):')
    print(f'    {scores}')
    print(f'    均值: {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%')

# ── 6类合并 ──
print('\n' + '='*50)
print('6类 SVM 交叉验证 (5折)')
print('='*50)

MERGE_6 = {
    'Angry':'Aggressive', 'Defence':'Aggressive',
    'Fighting':'Aggressive', 'Warning':'Aggressive',
    'Happy':'Happy', 'HuntingMind':'Happy',
    'Mating':'Mating', 'MotherCall':'MotherCall',
    'Paining':'Paining', 'Resting':'Resting',
}

y6 = le.fit_transform([MERGE_6.get(l, l) for l in labels])

for name, feats in [('FDAP', fdap_feats), ('GAP', gap_feats)]:
    pipe = make_pipeline(
        StandardScaler(),
        LinearSVC(C=1.0, max_iter=10000, dual='auto', random_state=42)
    )
    scores = cross_val_score(pipe, feats, y6, cv=5, scoring='accuracy')
    print(f'\n  {name} (dim={feats.shape[1]}):')
    print(f'    {scores}')
    print(f'    均值: {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%')

# ── RBF SVM 对比 ──
print('\n' + '='*50)
print('RBF SVM 对比 (6类, 5折, 仅FDAP)')
print('='*50)
from sklearn.svm import SVC

pipe_rbf = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
)
scores_rbf = cross_val_score(pipe_rbf, fdap_feats, y6, cv=5, scoring='accuracy')
print(f'  RBF SVM: {scores_rbf}')
print(f'  均值: {scores_rbf.mean()*100:.2f}% ± {scores_rbf.std()*100:.2f}%')

# ── 对比 HuBERT 基线 ──
print('\n' + '='*50)
print('对比 HuBERT freeze SVM 结果:')
print('='*50)
print('  HuBERT + GAP(均值) 10类:   82.2% (CatSound_V2, 5折)')
print('  HuBERT + GAP(均值) 6类:    84.3% (CatSound_V2, 5折)')

# 重新跑一次 FDAP/GAP 各 SVC 存结果
results = {}
all_configs = [
    ('fdap_10c', fdap_feats, y),
    ('gap_10c', gap_feats, y),
    ('fdap_6c', fdap_feats, y6),
    ('gap_6c', gap_feats, y6),
]
for cfg_name, feats, y_ in all_configs:
    pipe = make_pipeline(
        StandardScaler(),
        LinearSVC(C=1.0, max_iter=10000, dual='auto', random_state=42)
    )
    scores = cross_val_score(pipe, feats, y_, cv=5, scoring='accuracy')
    results[cfg_name] = {
        'scores': [round(float(s), 4) for s in scores],
        'mean': round(float(scores.mean()), 4),
        'std': round(float(scores.std()), 4),
    }

with open(OUTPUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\n结果 → {OUTPUT}')

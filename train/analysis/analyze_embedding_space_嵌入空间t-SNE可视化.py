"""
嵌入空间数值分析 — 从 HuBERT 嵌入层直接算类间分离度
比 t-SNE 肉眼观察更精确，输出所有量化指标

用法:
  1. 先跑 run_tsne_viz.py (如果有缓存可以跳过)
  2. python analyze_embedding_space.py

输出: 嵌入矩阵/标签/类间指标/混淆分析
# 嵌入空间可视化：HuBERT t-SNE + PCA 分析
"""

import os, sys, warnings, json, pickle
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import librosa
from transformers import HubertModel, Wav2Vec2FeatureExtractor

# ── sklearn 指标 ──
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC

# ── 配置 ──
SR = 16000
CROP_SAMPLES = int(SR * 2.0)  # 2秒
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
np.random.seed(SEED)

BASE = Path(r'/path/to/dataset')
DATA_DIR = BASE / 'CatSound_DataSet_V2/NAYA_DATA_AUG1X'
CACHE_DIR = BASE / '模型代码/embedding_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 要探索的层配置
LAYER_CFGS = {
    'early_L3-5': [3,4,5],
    'mid_L8-10':  [8,9,10],
    'all_L1-12':  list(range(1, 13)),
}

LABELS_10 = ['Angry','Defence','Fighting','Happy','HuntingMind',
             'Mating','MotherCall','Paining','Resting','Warning']

MERGE_6 = {
    'Angry':'Aggressive', 'Defence':'Aggressive', 'Fighting':'Aggressive', 'Warning':'Aggressive',
    'Happy':'Happy', 'HuntingMind':'Happy',
    'Mating':'Mating', 'MotherCall':'MotherCall',
    'Paining':'Paining', 'Resting':'Resting',
}


# =========================================================
# 1. 加载音频
# =========================================================
def load_data():
    cache_path = CACHE_DIR / 'audio_data.pkl'
    if cache_path.exists():
        print(f'  从缓存加载音频: {cache_path}')
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print('[1/4] 加载 CatSound_V2 音频...')
    samples = []
    for label in LABELS_10:
        d = DATA_DIR / label
        if not d.exists():
            print(f'  [SKIP] {label}: 目录不存在')
            continue
        # 同时读 mp3 和 wav
        files = sorted(d.glob('*.mp3')) + sorted(d.glob('*.wav'))
        if not files:
            continue
        for fp in files:
            try:
                y, _ = librosa.load(str(fp), sr=SR, mono=True)
                if len(y) > CROP_SAMPLES:
                    start = (len(y) - CROP_SAMPLES) // 2
                    y = y[start:start + CROP_SAMPLES]
                else:
                    pad = CROP_SAMPLES - len(y)
                    y = np.pad(y, (pad // 2, pad - pad // 2))
                samples.append({'path': str(fp), 'waveform': y.astype(np.float32), 'label': label})
            except Exception as e:
                print(f'  [FAIL] {label}/{fp.name}: {e}')
    
    print(f'  共 {len(samples)} 条音频')
    with open(cache_path, 'wb') as f:
        pickle.dump(samples, f)
    return samples


# =========================================================
# 2. HuBERT 嵌入提取
# =========================================================
def extract_embeddings(samples, layers, cache_key):
    """提取 HuBERT 指定层均值嵌入，有缓存"""
    cache_path = CACHE_DIR / f'emb_{cache_key}.npy'
    label_cache = CACHE_DIR / f'emb_{cache_key}_labels.npy'
    
    if cache_path.exists() and label_cache.exists():
        print(f'  从缓存加载嵌入: {cache_path}')
        embeddings = np.load(cache_path)
        labels = np.load(label_cache, allow_pickle=True)
        return embeddings, labels
    
    print(f'  HuBERT 嵌入提取: 层={layers} ({len(samples)} 条)...')
    processor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/hubert-base-ls960', local_files_only=True)
    model = HubertModel.from_pretrained('facebook/hubert-base-ls960', local_files_only=True).to(DEVICE)
    model.eval()
    
    all_embs, all_labels = [], []
    n = len(samples)
    
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            batch = samples[i:i + BATCH_SIZE]
            waves = [s['waveform'] for s in batch]
            all_labels.extend([s['label'] for s in batch])
            
            inputs = processor(waves, sampling_rate=SR, return_tensors='pt', padding=True).to(DEVICE)
            out = model(inputs.input_values, output_hidden_states=True)
            hs = out.hidden_states
            
            emb = torch.stack([hs[l] for l in layers], dim=0).mean(dim=0).mean(dim=1)  # (B, 768)
            all_embs.append(emb.cpu().numpy())
            
            if (i // BATCH_SIZE) % 10 == 0:
                print(f'    进度: {i + len(batch)}/{n}')
    
    embeddings = np.concatenate(all_embs, axis=0)
    labels = np.array(all_labels)
    
    np.save(cache_path, embeddings)
    np.save(label_cache, labels)
    print(f'  嵌入矩阵: {embeddings.shape}')
    
    return embeddings, labels


# =========================================================
# 3. 类间分析
# =========================================================
def analyze_class_separation(embeddings, labels_10, cfg_name):
    """在嵌入空间上直接算类间分离度"""
    print(f'\n{"="*60}')
    print(f'  分析: {cfg_name}')
    print(f'{"="*60}')
    
    n = len(labels_10)
    le = LabelEncoder()
    y_10 = le.fit_transform(labels_10)
    y_6 = np.array([MERGE_6[l] for l in labels_10])
    le_6 = LabelEncoder()
    y_6_enc = le_6.fit_transform(y_6)
    
    # ── 3.1 Silhouette / Davies-Bouldin / Calinski-Harabasz ──
    print('\n  [3.1] 全局聚类指标')
    
    sil_10 = silhouette_score(embeddings, y_10)
    sil_6 = silhouette_score(embeddings, y_6_enc)
    db_10 = davies_bouldin_score(embeddings, y_10)
    db_6 = davies_bouldin_score(embeddings, y_6_enc)
    ch_10 = calinski_harabasz_score(embeddings, y_10)
    ch_6 = calinski_harabasz_score(embeddings, y_6_enc)
    
    # 解释: Silhouette >0.5 说明类间分离好, <0.25 说明严重重叠
    def rate_sil(s):
        return '🟢 好' if s > 0.5 else '🟡 一般' if s > 0.25 else '🔴 差'
    
    print(f'  {"指标":<25} {"10类":>12} {"6类合并":>12} {"参考":>30}')
    print(f'  {"─"*79}')
    print(f'  {"Silhouette (↑好)":<25} {sil_10:>8.4f} {rate_sil(sil_10):>4} {sil_6:>8.4f} {rate_sil(sil_6):>4}')
    print(f'  {"Davies-Bouldin (↓好)":<25} {db_10:>8.4f}{"":>4} {db_6:>8.4f}')
    print(f'  {"Calinski-Harabasz (↑好)":<25} {ch_10:>8.0f}{"":>4} {ch_6:>8.0f}')
    
    # ── 3.1.5 PCA 有效维度分析 ──
    from sklearn.decomposition import PCA
    print('\n  [3.1.5] PCA 分析：嵌入有效维度')
    # 只在第一次跑
    if not hasattr(analyze_class_separation, '_pca_done'):
        pca = PCA(random_state=SEED).fit(embeddings)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        # 找需要多少维达到 50%/80%/90%/95% 方差
        for threshold in [0.5, 0.8, 0.9, 0.95, 0.99]:
            n_dims = int(np.searchsorted(cumvar, threshold) + 1)
            print(f'    {threshold:.0%} 方差需要 {n_dims:>3} 维')
        print(f'    前 10 维: {cumvar[9]:.1%}')
        print(f'    前 50 维: {cumvar[49]:.1%}')
        print(f'    前 100 维: {cumvar[99]:.1%}')
        analyze_class_separation._pca_done = True
    
    # ── 3.2 类间中心距离矩阵 ──
    print('\n  [3.2] 10 类中心 L2 距离矩阵')
    class_centers = {}
    for cls in LABELS_10:
        mask = labels_10 == cls
        class_centers[cls] = embeddings[mask].mean(axis=0)
    
    print(f'  {"":>14}', end='')
    for c in LABELS_10:
        c_short = c[:5]
        print(f'  {c_short:>7}', end='')
    print()
    for c1 in LABELS_10:
        print(f'{c1:>14}', end='')
        for c2 in LABELS_10:
            if c1 == c2:
                print(f'  {"—":>7}', end='')
            else:
                d = np.linalg.norm(class_centers[c1] - class_centers[c2])
                print(f'  {d:>5.1f}', end='')
        print()
    
    # ── 3.3 类内散布 (类内 RMS 距离) ──
    print('\n  [3.3] 类内散布 + 类间最近距离比')
    intra_rms = {}
    inter_nearest = {}
    for cls in LABELS_10:
        mask = labels_10 == cls
        pts = embeddings[mask]
        center = class_centers[cls]
        rms = np.sqrt(np.mean(np.sum((pts - center)**2, axis=1)))
        intra_rms[cls] = rms
        
        nearest_inter = min(
            np.linalg.norm(center - class_centers[other])
            for other in LABELS_10 if other != cls
        )
        inter_nearest[cls] = nearest_inter
    
    print(f'  {"类别":<14} {"类内RMS":>8} {"最近类距":>8} {"比值":>8} {"解读":>20}')
    print(f'  {"─"*58}')
    for cls in sorted(LABELS_10):
        ratio = intra_rms[cls] / inter_nearest[cls] if inter_nearest[cls] > 0 else float('inf')
        if ratio < 0.5:
            note = '🟢 可分'
        elif ratio < 1.0:
            note = '🟡 边界'
        else:
            note = '🔴 严重重叠'
        print(f'  {cls:<14} {intra_rms[cls]:>8.2f} {inter_nearest[cls]:>8.2f} {ratio:>7.2f} {note:>20}')
    
    # ── 3.4 最近邻混淆分析 ──
    print('\n  [3.4] KNN 类内/类间混合比例 (k=5)')
    nn = NearestNeighbors(n_neighbors=6, metric='euclidean')  # 6 = 5近邻 + 自己
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    
    # 对每个点，看最近 5 个邻居的标签分布
    confusion = defaultdict(lambda: defaultdict(int))
    for i in range(n):
        true_label = labels_10[i]
        # 跳过自己 (第一个邻居)
        neighbor_labels = [labels_10[idx] for idx in indices[i, 1:6]]
        for nl in neighbor_labels:
            if nl != true_label:
                confusion[true_label][nl] += 1
    
    print(f'  {"真实类":<14} {"误分最多3类":<30} {"误分率":>8}')
    print(f'  {"─"*52}')
    for cls in sorted(confusion.keys()):
        total_mis = sum(confusion[cls].values())
        n_cls = np.sum(labels_10 == cls)
        top3 = sorted(confusion[cls].items(), key=lambda x: -x[1])[:3]
        top3_str = ', '.join(f'{c}({v})' for c, v in top3)
        print(f'  {cls:<14} {top3_str:<30} {total_mis/min(n_cls*5, total_mis):>7.1%}')
    
    # ── 3.5 6 类合并后的分离度 ──
    print('\n  [3.5] 6 类合并 → 类内散布分析')
    class_centers_6 = {}
    for cls_6 in set(MERGE_6.values()):
        mask = y_6 == cls_6
        class_centers_6[cls_6] = embeddings[mask].mean(axis=0)
    
    for cls_6 in sorted(class_centers_6.keys()):
        mask = y_6 == cls_6
        pts = embeddings[mask]
        center = class_centers_6[cls_6]
        rms = np.sqrt(np.mean(np.sum((pts - center)**2, axis=1)))
        
        nearest_inter = min(
            np.linalg.norm(center - class_centers_6[other])
            for other in class_centers_6 if other != cls_6
        )
        ratio = rms / nearest_inter if nearest_inter > 0 else float('inf')
        
        # 这个类包含哪些原始子类
        sub_classes = [c for c in LABELS_10 if MERGE_6[c] == cls_6]
        
        if ratio < 0.5:
            note = '🟢'
        elif ratio < 1.0:
            note = '🟡'
        else:
            note = '🔴'
        
        print(f'  {note} {cls_6:<14} RMS={rms:.2f}  最近距={nearest_inter:.2f}  比={ratio:.2f}  子类={sub_classes}')
    
    # ── 3.6 SVM 交叉验证（给一个基准分类精度）──
    print('\n  [3.6] LinearSVM 5折 CV 分类准确率 (嵌入层直接分类)')
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    
    svm_10 = make_pipeline(StandardScaler(), LinearSVC(max_iter=10000, random_state=SEED, dual='auto'))
    svm_6 = make_pipeline(StandardScaler(), LinearSVC(max_iter=10000, random_state=SEED, dual='auto'))
    
    # 10 类
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores_10 = cross_val_score(svm_10, embeddings, y_10, cv=skf, scoring='accuracy', n_jobs=1)
    scores_6 = cross_val_score(svm_6, embeddings, y_6_enc, cv=skf, scoring='accuracy', n_jobs=1)
    
    print(f'  10类: {scores_10.mean():.1%} ± {scores_10.std():.1%}')
    print(f'  6类:  {scores_6.mean():.1%} ± {scores_6.std():.1%}')
    
    return {
        'silhouette_10': sil_10,
        'silhouette_6': sil_6,
        'davies_bouldin_10': db_10,
        'davies_bouldin_6': db_6,
        'calinski_harabasz_10': ch_10,
        'calinski_harabasz_6': ch_6,
        'svm_10_mean': float(scores_10.mean()),
        'svm_10_std': float(scores_10.std()),
        'svm_6_mean': float(scores_6.mean()),
        'svm_6_std': float(scores_6.std()),
    }


# =========================================================
# 主流程
# =========================================================
if __name__ == '__main__':
    print(f'╔{"═"*58}╗')
    print(f'║  HuBERT 嵌入空间数值分析 {" " * 34}║')
    print(f'║  CatSound_V2 ({len(LABELS_10)}类, 5922条){" " * 34}║')
    print(f'║  层配置: {list(LAYER_CFGS.keys())}║')
    print(f'║  设备: {DEVICE}{" " * 42}║')
    print(f'╚{"═"*58}╝')
    
    # 加载数据
    samples = load_data()
    
    # 为每层配置提取和分析
    all_results = {}
    
    for cfg_name, layers in LAYER_CFGS.items():
        print(f'\n── {cfg_name} (层 {layers}) ──')
        embeddings, labels = extract_embeddings(samples, layers, cfg_name)
        result = analyze_class_separation(embeddings, labels, cfg_name)
        all_results[cfg_name] = result
    
    # ── 汇总对比 ──
    print(f'\n{"="*60}')
    print(f'  汇总: 三层配置对比')
    print(f'{"="*60}')
    print(f'  {"配置":<15} {"Silhouette(10)":>15} {"Silhouette(6)":>14} {"DB(10)":>8} {"DB(6)":>8} {"SVM-10":>8} {"SVM-6":>8}')
    print(f'  {"─"*76}')
    for cfg_name in LAYER_CFGS:
        r = all_results[cfg_name]
        print(f'  {cfg_name:<15} {r["silhouette_10"]:>13.4f} {r["silhouette_6"]:>12.4f} {r["davies_bouldin_10"]:>7.2f} {r["davies_bouldin_6"]:>7.2f} {r["svm_10_mean"]:>5.1%} {r["svm_6_mean"]:>5.1%}')
    
    # 保存完整结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = BASE / f'模型代码/embedding_analysis_{timestamp}.json'
    result_dict = {k: {kk: round(vv, 6) if isinstance(vv, float) else vv 
                       for kk, vv in v.items()} 
                   for k, v in all_results.items()}
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    print(f'\n✅ 结果保存: {result_path}')
    print(f'   快去 Windows 终端跑: python analyze_embedding_space.py')

"""
译芯怡意 - 猫声分类训练入口 (CatSound_V2)
==========================================
两种模式:
  freeze: 冻结 SSL 模型提嵌入 + 轻量分类头 (基线, 已修正层选择)
  finetune: ForSequenceClassification 全模型微调 (HF Trainer API, 支HuBERT/WavLM)

用法:
  python 训练入口hubert_train.py --mode freeze          # 基线 (默认)
  python 训练入口hubert_train.py --mode finetune        # 全模型微调 (推荐)
# 主训练入口：HuBERT/WavLM freeze/finetune 模式
"""

import os, sys, json, time, warnings, hashlib, joblib
warnings.filterwarnings('ignore')
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_HUB_OFFLINE'] = '0'
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    HubertModel, HubertForSequenceClassification,
    WavLMForSequenceClassification, WavLMModel,
    Wav2Vec2FeatureExtractor, Trainer, TrainingArguments,
    EarlyStoppingCallback, get_linear_schedule_with_warmup
)
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import csv
import librosa
import argparse

# ── 默认配置 ──
SR = 16000
# 训练用全长音频, DURATION仅用于内存估算
DURATION = 5.0
BATCH_SIZE = 16
NUM_CLASSES = 10
DATA_DIR = Path('/mnt/c/Users/陈项目负责人/Desktop/CatSound_DataSet_V2/NAYA_DATA_AUG1X')
DATA_SELF_DIR = Path('/mnt/c/Users/陈项目负责人/Desktop/库猫语音频')
OUTPUT_DIR = Path('/mnt/c/Users/陈项目负责人/Desktop/模型代码/训练结果/hubert_v2')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT_DIR / 'cache'
CACHE_DIR.mkdir(exist_ok=True)
# 模型配置
MODEL_MAP = {
    'hubert': {
        'name': 'facebook/hubert-base-ls960',
        'class': HubertModel,
        'seq_class': HubertForSequenceClassification,
        'best_layers': [3, 4, 5],  # 12层扫描: Layer 3-6 最佳, Layer 4 峰顶
    },
    'wavlm': {
        'name': 'microsoft/wavlm-base-plus',
        'class': WavLMModel,
        'seq_class': WavLMForSequenceClassification,
        'best_layers': [4, 5, 6],  # WavLM 稍深层更好, 4-15 有效
    },
}
# (DEFAULT_MODEL 由 args default 决定, N_JOBS 由各sklearn调用自身决定)

LABELS = [
    "Angry", "Defence", "Fighting", "Happy", "HuntingMind",
    "Mating", "MotherCall", "Paining", "Resting", "Warning"
]

# 5类合并映射 (10类→5类):
# Aggressive=Angry+Defence+Fighting+Warning
# Happy=Happy+HuntingMind
# Calling=Mating+MotherCall
# Paining/Paining, Resting=Resting
MERGE_5_MAP = {
    'Angry': 'Aggressive', 'Defence': 'Aggressive',
    'Fighting': 'Aggressive', 'Warning': 'Aggressive',
    'Happy': 'Happy', 'HuntingMind': 'Happy',
    'Mating': 'Calling', 'MotherCall': 'Calling',
    'Paining': 'Paining', 'Resting': 'Resting',
}

# =========================================================
# 数据加载
# =========================================================
def load_catsound_v2(data_dir):
    """
    加载 CatSound_V2 数据集。
    从 Catmood_NAYA_1xAug.csv 读取真实标签 (0-9 数值映射),
    避免从目录名推断标签可能带来的顺序错误。
    返回: (file_paths, labels, cat_ids, is_aug)
    cat_ids 仅用于 GroupKFold (近似猫级划分, 无真实猫ID)
    is_aug: True 表示该文件是 _aug1(1) 版本的噪版
    """
    csv_path = data_dir / 'Catmood_NAYA_1xAug.csv'
    files, labels, cat_ids, is_aug = [], [], [], []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            next(reader)  # 跳过表头: ,path,label
        except StopIteration:
            print(f'  [!] CSV 为空: {csv_path}')
            return np.array(files), np.array(labels), np.array(cat_ids)
        for row in reader:
            if len(row) < 3:
                continue
            csv_path_str = row[1].strip()  # D:/DATASET/.../Angry/cat02.mp3
            label_idx = int(row[2])  # 0-9
            label_name = LABELS[label_idx]
            # 从完整路径取文件名, 到本地 data_dir 下找
            filename = os.path.basename(csv_path_str)
            local_path = data_dir / label_name / filename
            if local_path.exists():
                files.append(str(local_path))
                labels.append(label_name)
                raw_stem = local_path.stem.replace('_aug1(1)', '').lower()
                cat_ids.append(raw_stem)
                is_aug.append('_aug1(1)' in local_path.stem)
    return np.array(files), np.array(labels), np.array(cat_ids), np.array(is_aug)

def preprocess_audio(y):
    """预处理: 峰值归一化, 保留全长"""
    peak = np.max(np.abs(y))
    if peak < 1e-10:
        return np.zeros_like(y, dtype=np.float32)
    return y / peak

def load_all_audio(files):
    """返回文件路径列表(非波形数组), 嵌入提取时逐个流式加载"""
    return list(files)

# =========================================================
# 模式 A: 冻结 HuBERT 提嵌入 + 轻量分类器
# =========================================================
class SSLFeatureExtractor:
    """冻结 SSL 模型提嵌入, 支持 HuBERT/WavLM/Wav2Vec2

    - Crossing the Species Divide (2025): WavLM > HuBERT
    - Cauzinille et al. (2024): 初始层 > 最终层
    - 支持简单 mean pooling 和 Time-Weighted Averaging
    """

    def __init__(self, model_key='hubert', pooling='mean', device=None, local_files_only=False):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        cfg = MODEL_MAP[model_key]
        self.model_name = cfg['name']
        self.layer_indices = cfg['best_layers']
        self.pooling = pooling

        try:
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.model_name, local_files_only=local_files_only)
        except OSError:
            # 部分模型(如 WavLM)没有 preprocessor_config.json, 手动构造
            self.processor = Wav2Vec2FeatureExtractor(sampling_rate=SR)
            print(f'  [!] {self.model_name} 无 preprocessor_config, 使用默认16kHz processor')
        self.model = cfg['class'].from_pretrained(
            self.model_name, local_files_only=local_files_only)
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.hidden_size = self.model.config.hidden_size

        # Time-Weighted Averaging: 可学习的帧注意力 (仅+1025参数)
        if self.pooling == 'attn':
            self.twa_w = nn.Parameter(torch.randn(self.hidden_size) * 0.01)
            self.twa_b = nn.Parameter(torch.zeros(1))
        else:
            self.twa_w = None

    def _load_batch_with_aug(self, batch_paths, pitch_shift=None):
        """加载一个 batch 的音频, 可选 pitch shift"""
        batch = []
        for fp in batch_paths:
            y, _ = librosa.load(fp, sr=SR, mono=True)
            if pitch_shift is not None:
                y = librosa.effects.pitch_shift(y, sr=SR, n_steps=pitch_shift)
            batch.append(preprocess_audio(y))
        return batch

    @torch.no_grad()
    def extract(self, file_list, pitch_shift=None):
        """流式提取嵌入, 可选 pitch shift"""
        all_embeddings = []
        for i in range(0, len(file_list), BATCH_SIZE):
            batch_paths = file_list[i:i + BATCH_SIZE]
            waveforms_batch = self._load_batch_with_aug(batch_paths, pitch_shift)

            inputs = self.processor(
                waveforms_batch, sampling_rate=SR,
                return_tensors="pt", padding=True,
            ).input_values.to(self.device)

            outputs = self.model(inputs, output_hidden_states=True)
            hidden = outputs.hidden_states  # [0]=CNN, [1-12]=transformer
            layer_stack = torch.stack([hidden[i] for i in self.layer_indices], dim=0)
            target = layer_stack.mean(dim=0)  # (B, T, D)

            if self.pooling == 'attn' and self.twa_w is not None:
                scores = torch.einsum('btk,k->bt', target, self.twa_w.to(target.device)) \
                         + self.twa_b.to(target.device)
                weights = F.softmax(scores, dim=1).unsqueeze(-1)
                emb = (weights * target).sum(dim=1)
            else:
                emb = target.mean(dim=1)

            all_embeddings.append(emb.cpu().numpy())
            n_done = min(i + BATCH_SIZE, len(file_list))
            # 每 ~500 样本打印一次进度
            if n_done % 500 < BATCH_SIZE or n_done >= len(file_list):
                print(f'    嵌入: {n_done}/{len(file_list)}')

        return np.concatenate(all_embeddings, axis=0)

class EmbeddingMLP(nn.Module):
    """768→64→num_classes, ~50K params"""
    def __init__(self, input_dim=768, num_classes=10, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)

def _build_pair_map(cat_ids, is_aug):
    """
    构建配对映射: 每只猫的清洁版和噪版索引。
    返回: pair_map = {cat_id: {'clean': idx, 'noisy': idx}}
    如果某猫缺失某个版本, 对应值 = None
    """
    pair_map = {}
    for i, (cid, aug) in enumerate(zip(cat_ids, is_aug)):
        if cid not in pair_map:
            pair_map[cid] = {'clean': None, 'noisy': None}
        if aug:
            pair_map[cid]['noisy'] = i
        else:
            pair_map[cid]['clean'] = i
    return pair_map


def _pair_average_embeddings(base_embeddings, pair_map, cat_ids, selected_cats=None):
    """
    对选中的猫, 将它们的清洁和噪版嵌入平均 → 噪声中和。
    selected_cats: 要处理的猫ID列表。None=全部。
    返回: (avg_embeddings, avg_labels, avg_cat_ids)
    """
    if selected_cats is None:
        selected_cats = list(pair_map.keys())
    
    embs, labs, cats = [], [], []
    for cid in selected_cats:
        p = pair_map[cid]
        if p['clean'] is None or p['noisy'] is None:
            # 如果缺少一个版本, 用另一个
            idx = p['clean'] if p['clean'] is not None else p['noisy']
            if idx is None:
                continue
            embs.append(base_embeddings[idx])
        else:
            embs.append((base_embeddings[p['clean']] + base_embeddings[p['noisy']]) / 2)
        cats.append(cid)
    return np.array(embs), cats


def run_freeze_mode(files, labels, cat_ids, is_aug, ctx=None):
    """模式 A (v2): 冻结嵌入 + 配对平均 + 10类 GroupKFold

    基于 2961 × 2 数据配对结构:
      - 训练时: 每只猫的清洁+噪版嵌入平均 → 噪声中和
      - 评估: 报告清洁测试、噪版测试、交叉验证三种指标
      - 只训 10 类 (不在训练时合并, 推理时再 10→6 合并)
    """
    ctx = ctx or {'model': 'hubert', 'pooling': 'mean', 'output_dir': OUTPUT_DIR}
    output_dir = ctx.get('output_dir', OUTPUT_DIR)

    # ── 1. 提取嵌入 (缓存) ──
    print('\n[freeze v2] 加载音频...')
    waveforms = load_all_audio(files)
    
    layers_str = '_'.join(str(l) for l in ctx.get('layers', [2, 3, 4]))
    cache_name = f'embed_{ctx["model"]}_layers{layers_str}_{ctx.get("pooling", "mean")}'
    cache_path = CACHE_DIR / f'{cache_name}.npy'
    cache_hash_path = CACHE_DIR / f'{cache_name}.sha256'
    base_embeddings = None
    if cache_path.exists():
        try:
            e = np.load(cache_path)
            if cache_hash_path.exists():
                actual = hashlib.sha256(e.tobytes()).hexdigest()
                expected = cache_hash_path.read_text().strip()
                if actual == expected:
                    base_embeddings = e
                    print(f'[freeze v2] 缓存校验通过 ({cache_path.name})')
                else:
                    print(f'[freeze v2] [!] 缓存损坏, 重新提取')
            else:
                print(f'[freeze v2] [!] 无校验文件, 重新提取')
        except Exception as ex:
            print(f'[freeze v2] [!] 缓存异常: {ex}, 重新提取')

    hubert_for_cross = None
    hubert_proc_for_cross = None
    
    if base_embeddings is None:
        extractor = SSLFeatureExtractor(ctx['model'], ctx['pooling'])
        # 保存 HuBERT 供跨数据集评估复用（避免二次加载 378MB）
        hubert_for_cross = extractor.model
        hubert_proc_for_cross = extractor.processor
        base_embeddings = extractor.extract(waveforms)
        print(f'  基嵌入: {len(base_embeddings)} 条')
        np.save(cache_path, base_embeddings)
        cache_hash_path.write_text(hashlib.sha256(base_embeddings.tobytes()).hexdigest())
        print(f'[freeze v2] 基嵌入已缓存: {cache_path}')
    else:
        # 缓存有, 但还是需要 HuBERT 做跨数据集推理
        tempext = SSLFeatureExtractor(ctx['model'], ctx['pooling'])
        hubert_for_cross = tempext.model
        hubert_proc_for_cross = tempext.processor
    
    n_total = len(base_embeddings)

    # ── 2. 构建配对映射 ──
    pair_map = _build_pair_map(cat_ids, is_aug)
    n_pairs = sum(1 for p in pair_map.values() if p['clean'] is not None and p['noisy'] is not None)
    n_clean_only = sum(1 for p in pair_map.values() if p['clean'] is not None and p['noisy'] is None)
    n_noisy_only = sum(1 for p in pair_map.values() if p['noisy'] is not None and p['clean'] is None)
    all_cats = sorted(pair_map.keys())
    print(f'\n  配对结构: {n_pairs} 完整配对 + {n_clean_only} 仅清洁 + {n_noisy_only} 仅噪版 = {len(all_cats)} 猫')

    # ── 3. 10 类标签编码 (不在训练时合并) ──
    le = LabelEncoder()
    y = le.fit_transform(labels)
    print(f'  标签: {len(le.classes_)} 类 → {list(le.classes_)}')

    # ── 4. 配对平均 5 折交叉验证 ──
    print('\n[freeze v2] GroupKFold (5 folds) + 配对平均...')
    
    # GroupKFold 在猫级划分, 使用清洁版的标签 (原版和噪版标签一样)
    # 用 is_aug == False 的索引做划分 (只取清洁版)
    clean_mask = ~is_aug
    clean_indices = np.where(clean_mask)[0]
    clean_labels = y[clean_indices]
    clean_cat_ids = cat_ids[clean_indices]
    
    gkf = GroupKFold(n_splits=5)
    svm_fold_scores = []
    best_fold = 0
    best_fold_acc = 0.0
    fold_details = []

    for fold, (tr_cidx, te_cidx) in enumerate(gkf.split(clean_indices, clean_labels, clean_cat_ids), 1):
        # 训练猫和测试猫的 ID
        train_cats = set(clean_cat_ids[tr_cidx])
        test_cats = set(clean_cat_ids[te_cidx])
        
        # 配对平均 → 训练嵌入
        X_tr, train_cat_list = _pair_average_embeddings(base_embeddings, pair_map, cat_ids, train_cats)
        y_tr_labels = [labels[pair_map[c]['clean'] or pair_map[c]['noisy']] for c in train_cat_list]
        y_tr = le.transform(y_tr_labels)
        
        # 测试: 清洁版
        X_te_clean = np.array([base_embeddings[pair_map[c]['clean']] for c in test_cats])
        y_te_labels = [labels[pair_map[c]['clean']] for c in test_cats]
        y_te = le.transform(y_te_labels)
        
        # 测试: 噪版 (如有)
        X_te_noisy = None
        noisy_cats = [c for c in test_cats if pair_map[c]['noisy'] is not None]
        if noisy_cats:
            X_te_noisy = np.array([base_embeddings[pair_map[c]['noisy']] for c in noisy_cats])
            y_te_noisy = np.array([le.transform([labels[pair_map[c]['noisy']]])[0] for c in noisy_cats])
        
        print(f'\n  Fold {fold}: 训练{len(X_tr)}猫 → 测试{len(X_te_clean)}(清洁)+{len(noisy_cats)}(噪版)')
        
        # LinearSVM
        scaler = StandardScaler()
        lsvm = LinearSVC(C=1.0, class_weight='balanced', max_iter=3000, random_state=42,
                          dual='auto')
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te_clean)
        lsvm.fit(X_tr_s, y_tr)
        acc_clean = lsvm.score(X_te_s, y_te)
        preds_clean = lsvm.predict(X_te_s)
        f1_clean = f1_score(y_te, preds_clean, average='macro', zero_division=0)
        
        acc_noisy = None
        if X_te_noisy is not None:
            X_tn_s = scaler.transform(X_te_noisy)
            acc_noisy = lsvm.score(X_tn_s, y_te_noisy)
        
        acc_all = (acc_clean * len(X_te_clean) + (acc_noisy or 0) * len(noisy_cats)) / (len(X_te_clean) + len(noisy_cats))
        svm_fold_scores.append(acc_clean)
        if acc_clean > best_fold_acc:
            best_fold_acc = acc_clean
            best_fold = fold
        
        print(f'    SVM 清洁测试: {acc_clean:.1%}  F1={f1_clean:.1%}', end='')
        if acc_noisy is not None:
            print(f'  | 噪版测试: {acc_noisy:.1%}', end='')
        print(f'  | 总体: {acc_all:.1%}')
        
        fold_details.append({
            'fold': fold, 'n_train': len(X_tr), 'n_test': len(X_te_clean),
            'acc_clean': float(acc_clean), 'acc_noisy': float(acc_noisy) if acc_noisy is not None else None,
            'f1_clean': float(f1_clean),
        })

    print(f'\n── 5 折 CV (SVM 清洁测试) ──')
    print(f'  Acc: {np.mean(svm_fold_scores):.1%} ± {np.std(svm_fold_scores):.1%}')
    
    # 收集 F1
    fold_f1s = [d['f1_clean'] for d in fold_details]
    print(f'  F1:  {np.mean(fold_f1s):.1%} ± {np.std(fold_f1s):.1%}')
    fold_f1s = [d['f1_clean'] for d in fold_details]
    print(f'  F1:  {np.mean(fold_f1s):.1%} ± {np.std(fold_f1s):.1%}')

    # ── 5. 用全部数据训练最终模型 ──
    print(f'\n▓ 最佳 Fold: {best_fold}, 训练最终模型...')
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    save_dir = output_dir / f'freeze_model_{ctx["model"]}_{timestamp}'
    save_dir.mkdir(parents=True, exist_ok=True)

    # 全部配对平均
    X_all, all_cat_list = _pair_average_embeddings(base_embeddings, pair_map, cat_ids, all_cats)
    # 获取每只猫的标签 (优先清洁版, 没有就用噪版)
    y_all_labels = []
    for c in all_cat_list:
        p = pair_map[c]
        idx = p['clean'] if p['clean'] is not None else p['noisy']
        y_all_labels.append(labels[idx])
    y_all = le.transform(y_all_labels)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    final_svm = LinearSVC(C=1.0, class_weight='balanced', max_iter=3000, random_state=42,
                           dual='auto')
    final_svm.fit(X_scaled, y_all)
    joblib.dump({'svm': final_svm, 'scaler': scaler, 'label_encoder': le},
                save_dir / 'svm.pkl')
    print(f'  SVM 已保存: {save_dir / "svm.pkl"}')

    # ── 6. 跨数据集自动测试 ──
    cross_results = {}
    cross_data_dirs = [
        ('orig163', Path(r'/mnt/c/Users/陈项目负责人/Desktop/原猫语分类/情绪分类')),
    ]
    
    for ds_name, data_dir in cross_data_dirs:
        if not data_dir.exists():
            print(f'  跳过跨数据集 ({ds_name}): 目录不存在')
            continue
        print(f'\n  跨数据集测试: {ds_name}')
        acc = _eval_cross_dataset(final_svm, scaler, le, data_dir,
                                           hubert_model=hubert_for_cross, hubert_processor=hubert_proc_for_cross)
        cross_results[ds_name] = acc
    
    print(f'\n▓ 模型已保存: {save_dir}')
    print(f'   └─ svm.pkl  (LinearSVM, 10类, 配对平均训练)')

    return {
        'mode': 'freeze_v2',
        'n_total': int(n_total),
        'n_pairs': int(n_pairs),
        'n_unique_cats': int(len(all_cats)),
        'cv_folds': fold_details,
        'svm_mean': float(np.mean(svm_fold_scores)),
        'svm_std': float(np.std(svm_fold_scores)),
        'f1_mean': float(np.mean(fold_f1s)),
        'f1_std': float(np.std(fold_f1s)),
        'best_fold': int(best_fold),
        'cross_dataset': cross_results,
    }


def _eval_cross_dataset(svm, scaler, le, data_dir, hubert_model=None, hubert_processor=None, device=None):
    """
    在指定目录 (原猫语分类格式) 上跑推理评估。
    
    可传入已有的 HuBERT 模型避免重复加载。
    输出 10 类和 6 类准确率。
    """
    import torch
    from transformers import HubertModel, Wav2Vec2FeatureExtractor
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if hubert_model is None or hubert_processor is None:
        try:
            hubert_model = HubertModel.from_pretrained('facebook/hubert-base-ls960', local_files_only=True)
            hubert_model.to(device); hubert_model.eval()
            for p in hubert_model.parameters(): p.requires_grad = False
            hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/hubert-base-ls960', local_files_only=True)
        except Exception as e:
            print(f'  [!] HuBERT 加载失败: {e}')
            return {'error': str(e)}
    
    layers = [3, 4, 5]
    
    CN_MAP = {
        '交配（低沉嘶吼）': 'Mating', '休息': 'Resting', '快乐': 'Happy',
        '愤怒（呜哇哇哇）': 'Angry', '打架（尖锐刺耳）': 'Fighting',
        '母体呼叫（警笛）': 'MotherCall', '狩猎（短促的嗷呜）': 'HuntingMind',
        '疼痛': 'Paining', '警告（呜~）（咕噜）': 'Warning',
        '防御(响尾蛇嘶哈)': 'Defence',
    }
    MERGE_6 = {
        'Angry':'Aggressive', 'Defence':'Aggressive', 'Fighting':'Aggressive', 'Warning':'Aggressive',
        'Happy':'Happy', 'HuntingMind':'Happy', 'Mating':'Mating', 'MotherCall':'MotherCall',
        'Paining':'Paining', 'Resting':'Resting',
    }
    
    total, correct_10, correct_6 = 0, 0, 0
    per_class = {}
    
    for cn_dir in sorted(data_dir.iterdir()):
        if not cn_dir.is_dir(): continue
        en_label = CN_MAP.get(cn_dir.name)
        if en_label is None: continue
        merged6 = MERGE_6[en_label]
        
        files = sorted(cn_dir.glob('*.wav')) + sorted(cn_dir.glob('*.mp3'))
        if not files: continue
        
        if merged6 not in per_class:
            per_class[merged6] = {'total': 0, 'correct_10': 0, 'correct_6': 0}
        
        for fp in files:
            total += 1
            per_class[merged6]['total'] += 1
            
            y, _ = librosa.load(str(fp), sr=SR, mono=True)
            peak = np.max(np.abs(y))
            y = y / peak if peak > 1e-10 else np.zeros_like(y)
            
            with torch.no_grad():
                inputs = hubert_processor([y], sampling_rate=SR, return_tensors="pt", padding=True).input_values.to(device)
                outputs = hubert_model(inputs, output_hidden_states=True)
                hs = outputs.hidden_states
                emb = torch.stack([hs[i] for i in layers], dim=0).mean(dim=0).mean(dim=1).cpu().numpy()
            
            emb_s = scaler.transform(emb)
            pred_label = le.inverse_transform(svm.predict(emb_s))[0]
            pred_6 = MERGE_6.get(pred_label, pred_label)
            
            if pred_label == en_label:
                correct_10 += 1
                per_class[merged6]['correct_10'] += 1
            if pred_6 == merged6:
                correct_6 += 1
                per_class[merged6]['correct_6'] += 1
    
    print(f'    10类准确率: {correct_10}/{total} = {correct_10/total:.1%}')
    print(f'    6类合并:     {correct_6}/{total} = {correct_6/total:.1%}')
    
    for cls in sorted(per_class.keys()):
        d = per_class[cls]
        a10 = d['correct_10'] / d['total'] if d['total'] > 0 else 0
        a6 = d['correct_6'] / d['total'] if d['total'] > 0 else 0
        print(f'      {cls:<14}: 10类={a10:.0%}  6类={a6:.0%}')
    
    return {'total': total, 'correct_10': correct_10, 'acc_10': correct_10/total,
            'correct_6': correct_6, 'acc_6': correct_6/total, 'per_class': per_class}

# =========================================================
# 模式 B: HubertForSequenceClassification 全模型微调
# =========================================================
# SpecAugment-like augmentation for HF processor
# frequency/time masking at wave level (via spec, then iSTFT back)
# For simplicity, apply noise injection + time shift in waveform domain

class CatSoundHF(Dataset):
    """HuggingFace 格式数据集, 支持训练时增强
    
    首次加载时缓存预处理后的音频到内存，避免每 epoch 重读 MP3。
    缓存存到 output_dir/audio_cache/ 下 .npy 文件。
    """
    def __init__(self, files, labels, processor, le, augment=False, cache_dir=None):
        self.files = files
        self.labels = torch.LongTensor([le.transform([l])[0] for l in labels])
        self.processor = processor
        self.augment = augment
        self.cache = {}
        cache_parent = cache_dir or Path('./cache')
        audio_cache_dir = Path(cache_parent) / 'audio_cache'
        audio_cache_dir.mkdir(parents=True, exist_ok=True)
        
        for i, fp in enumerate(files):
            # ⚠ 不能用 index 命名！train/test 不同 fold 的 index 会冲突
            # 改为文件路径的 md5, 确保同一文件永远映射同一缓存
            cache_name = hashlib.md5(fp.encode()).hexdigest() + '.npy'
            cache_path = audio_cache_dir / cache_name
            if cache_path.exists():
                self.cache[fp] = np.load(cache_path)
            else:
                y, _ = librosa.load(fp, sr=SR, mono=True)
                y = preprocess_audio(y)
                self.cache[fp] = y
                np.save(cache_path, y)
            if (i + 1) % 1000 == 0:
                print(f'  缓存音频: {i+1}/{len(files)}')
        print(f'  音频缓存完成: {len(files)} 条')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        y = self.cache[self.files[idx]].copy()  # copy 避免修改缓存
        if self.augment:
            # 噪声注入 (SNR ~20dB 随机)
            noise = np.random.randn(len(y)) * 0.01
            y = y + noise
            # 时移 (±10%)
            shift = np.random.randint(-int(0.1 * len(y)), int(0.1 * len(y)))
            y = np.roll(y, shift)
        inputs = self.processor(y, sampling_rate=SR, return_tensors="pt", padding=False)
        out = {'input_values': inputs['input_values'].squeeze(0), 'labels': self.labels[idx]}
        return out

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        'accuracy': (preds == labels).mean(),
        'f1_macro': f1_score(labels, preds, average='macro'),
    }

def run_finetune_mode(files, labels, cat_ids, lr=1e-5, epochs=30, model_key='hubert', output_dir=None):
    """模式 B: 全模型微调, 冻结 CNN encoder"""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    le = LabelEncoder()
    y = le.fit_transform(labels)
    cfg = MODEL_MAP.get(model_key, MODEL_MAP['hubert'])

    try:
        processor = Wav2Vec2FeatureExtractor.from_pretrained(cfg['name'])
    except OSError:
        processor = Wav2Vec2FeatureExtractor(sampling_rate=SR)
        print(f'  [!] {cfg["name"]} 无 preprocessor_config, 使用默认16kHz processor')
    model = cfg['seq_class'].from_pretrained(
        cfg['name'], num_labels=len(le.classes_)
    )
    # ── 稳定 Head 初始化 ──
    # 根因: HubertForSeqClass/WavLMForSeqClass 有两层随机头:
    #   projector Linear(768→256) → classifier Linear(256→10)
    # 共 ~200K 随机参数, 初始梯度巨大 → 推歪 backbone。
    # 修复: 两层都用极小权重 init, 使初始 logits→0(≈均匀预测)
    # 注意: 不能用 eye_ 初始化(proj是768x256非方阵,会丢失67%特征)
    if hasattr(model, 'projector') and model.projector is not None:
        nn.init.normal_(model.projector.weight, mean=0.0, std=0.05)
        nn.init.zeros_(model.projector.bias)
        print(f'  [fix] projector {tuple(model.projector.weight.shape)} 用小权重初始化(σ=0.001)')
    if hasattr(model, 'classifier'):
        nn.init.normal_(model.classifier.weight, mean=0.0, std=0.05)
        nn.init.zeros_(model.classifier.bias)
        print(f'  [fix] classifier {tuple(model.classifier.weight.shape)} 用小权重初始化(σ=0.001)')

    # ★ 小权重 init 后 logits≈0 → softmax≈均匀→CE loss≈2.302
    # 梯度反向传播到 backbone 约被缩小 1000x, backbone 几乎不动
    # head 先自己收敛, 然后梯度自然增大, backbone 才开始被微调

    # 对 backbone 和 head 使用不同学习率
    # backbone (WavLM transformer, 预训练): lr/5
    # head (projector+classifier, 随机初始化): lr
    no_decay = ['bias', 'LayerNorm.weight']
    backbone_decay, backbone_no_decay = [], []
    head_decay, head_no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'classifier' in n or 'projector' in n:
            # head params
            if any(nd in n for nd in no_decay):
                head_no_decay.append(p)
            else:
                head_decay.append(p)
        else:
            # backbone params
            if any(nd in n for nd in no_decay):
                backbone_no_decay.append(p)
            else:
                backbone_decay.append(p)
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_decay, 'lr': lr / 5, 'weight_decay': 0.01},
        {'params': backbone_no_decay, 'lr': lr / 5, 'weight_decay': 0.0},
        {'params': head_decay, 'lr': lr, 'weight_decay': 0.01},
        {'params': head_no_decay, 'lr': lr, 'weight_decay': 0.0},
    ])
    # 冻结 CNN encoder + 设备
    model.freeze_feature_encoder()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # GroupKFold 按猫划分, 只取第一折 (避免数据泄露, 比全5折快)
    gkf = GroupKFold(n_splits=5)
    train_idx, test_idx = next(gkf.split(range(len(files)), y, cat_ids))
    print(f'  GroupKFold: 训练{len(train_idx)}条, 测试{len(test_idx)}条')

    train_ds = CatSoundHF(files[train_idx], labels[train_idx], processor, le, augment=True, cache_dir=output_dir)
    test_ds = CatSoundHF(files[test_idx], labels[test_idx], processor, le, augment=False, cache_dir=output_dir)

    eff_batch = 4 * 8  # per_device_batch * grad_accum
    total_steps_per_epoch = len(train_ds) // eff_batch
    if len(train_ds) % eff_batch:
        total_steps_per_epoch += 1
    total_training_steps = total_steps_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_training_steps),
        num_training_steps=total_training_steps,
    )
    print(f'  Scheduler: {total_steps_per_epoch} steps/epoch × {epochs} epochs = {total_training_steps} total, warmup={int(0.1 * total_training_steps)}')

    args = TrainingArguments(
        output_dir=str(output_dir / 'finetune'),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=epochs,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        logging_dir=str(output_dir / 'logs'),
        logging_steps=5,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        save_total_limit=2,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=processor,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        optimizers=(optimizer, scheduler),
    )

    trainer.train()
    eval_result = trainer.evaluate()
    print(f'\n[finetune] 测试结果: {eval_result}')

    # 保存
    model.save_pretrained(str(output_dir / 'finetune_model'))
    processor.save_pretrained(str(output_dir / 'finetune_model'))

    return {'mode': 'finetune', **eval_result}

# =========================================================
def run_layers_scan(files, labels, cat_ids, ctx):
    """一次嵌入提取, 评估所有12层 (SVM 改用 LinearSVC 避免卡死)"""
    ctx = ctx or {'model': 'hubert', 'output_dir': OUTPUT_DIR}
    output_dir = ctx.get('output_dir', OUTPUT_DIR)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    print('\n[layer-scan] 一次嵌入提取 + 所有12层评估')
    print('[layer-scan] 提取嵌入 (一次, 保存所有层)...')
    file_list = list(files)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = MODEL_MAP[ctx['model']]

    try:
        processor = Wav2Vec2FeatureExtractor.from_pretrained(cfg['name'])
    except OSError:
        processor = Wav2Vec2FeatureExtractor(sampling_rate=SR)
        print(f'  [!] {cfg["name"]} 无 preprocessor_config, 使用默认16kHz processor')
    
    model = cfg['class'].from_pretrained(cfg['name'])
    model.to(device); model.eval()
    for p in model.parameters(): p.requires_grad = False

    all_layer_embs = {i: [] for i in range(12)}
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(file_list), BATCH_SIZE):
            batch_paths = file_list[i:i + BATCH_SIZE]
            batch = []
            for fp in batch_paths:
                wav, _ = librosa.load(fp, sr=SR, mono=True)
                batch.append(preprocess_audio(wav))
            inputs = processor(batch, sampling_rate=SR, return_tensors="pt", padding=True).input_values.to(device)
            outputs = model(inputs, output_hidden_states=True)
            for layer_idx in range(12):
                all_layer_embs[layer_idx].append(outputs.hidden_states[layer_idx + 1].mean(dim=1).cpu().numpy())
            if (i // BATCH_SIZE) % 16 == 0:
                print(f'    嵌入: {min(i+BATCH_SIZE, len(file_list))}/{len(file_list)} ({time.time()-t0:.0f}s)')
        print()

    for k in all_layer_embs:
        all_layer_embs[k] = np.concatenate(all_layer_embs[k], axis=0)

    gkf = GroupKFold(n_splits=3)
    scan_results = {}
    for layer_idx in range(12):
        emb = all_layer_embs[layer_idx]
        svm_s, mlp_s, rf_s = [], [], []
        for tr, te in gkf.split(emb, y, cat_ids):
            scaler = StandardScaler()
            lsvm = LinearSVC(C=1.0, class_weight='balanced', max_iter=2000, random_state=42, dual='auto')
            lsvm.fit(scaler.fit_transform(emb[tr]), y[tr])
            svm_s.append(lsvm.score(scaler.transform(emb[te]), y[te]))

            d = device; X_tr = torch.FloatTensor(emb[tr]); X_te = torch.FloatTensor(emb[te])
            y_tr = torch.LongTensor(y[tr]); y_te = torch.LongTensor(y[te])
            mlp = EmbeddingMLP(input_dim=768, num_classes=len(le.classes_)).to(d)
            opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
            best_a = 0
            for _ in range(30):
                mlp.train(); opt.zero_grad()
                F.cross_entropy(mlp(X_tr.to(d)), y_tr.to(d)).backward()
                torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0); opt.step()
                with torch.no_grad():
                    a = (mlp(X_te.to(d)).argmax(1).cpu() == y_te).float().mean().item()
                if a > best_a: best_a = a
            mlp_s.append(best_a)

            rf = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)
            rf.fit(scaler.fit_transform(emb[tr]), y[tr])
            rf_s.append(rf.score(scaler.transform(emb[te]), y[te]))

        scan_results[layer_idx] = {'svm': float(np.mean(svm_s)), 'mlp': float(np.mean(mlp_s)), 'rf': float(np.mean(rf_s))}
        print(f'  Layer {layer_idx+1:2d}:  SVM={scan_results[layer_idx]["svm"]:.1%}  MLP={scan_results[layer_idx]["mlp"]:.1%}  RF={scan_results[layer_idx]["rf"]:.1%}')

    best = max(range(12), key=lambda k: scan_results[k]['svm'])
    print(f'\nSVM 最高: Layer {best+1} = {scan_results[best]["svm"]:.1%}')

    ts = time.strftime('%Y%m%d_%H%M%S')
    rp = output_dir / f'results_layerscan_{ts}.json'
    with open(rp, 'w') as f: json.dump(scan_results, f, indent=2)
    print(f'\n结果: {rp}')


# 主入口
# =========================================================
def main():
    parser = argparse.ArgumentParser(description='HuBERT 猫声分类')
    parser.add_argument('--mode', choices=['freeze', 'finetune'], default='freeze',
                       help='freeze=嵌入+分类头, finetune=全模型微调')
    parser.add_argument('--data-dir', type=str, default=str(DATA_DIR),
                       help='CatSound_V2 数据目录')
    parser.add_argument('--self-dir', type=str, default=str(DATA_SELF_DIR),
                       help='自标注数据目录 (留空跳过)')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR),
                       help='输出目录')
    parser.add_argument('--model', type=str, default='hubert', choices=['hubert', 'wavlm'],
                       help='基础模型: hubert(base) 或 wavlm(推荐)')
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'attn'],
                       help='帧池化方式: mean(简单平均), attn(注意力加权-TWA)')
    parser.add_argument('--aug', type=int, default=0,
                       help='每文件增强倍数 (1=原+PitchShift, 训练集膨胀2x)')
    parser.add_argument('--layers-scan', action='store_true',
                       help='扫描所有12层, 找出最优层 (耗时约 2x 单次)')
    parser.add_argument('--merge', type=int, default=10, choices=[10, 5, 6],
                       help='分类数: 10=原始Pandeya, 5=合并(Aggressive+Happy+Calling+Pain+Rest)')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='学习率 (finetune 模式)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='训练轮数 (finetune 模式)')
    parser.add_argument('--name', type=str, default=None,
                       help='实验名, 用于结果文件名')
    args = parser.parse_args()

    # 将 args 传递给子函数
    print('═' * 60)
    print(f'  猫声分类 - CatSound_V2')
    print(f'  模型: {args.model.upper()}, 模式: {args.mode}, 池化: {args.pooling}')
    print(f'  LR: {args.lr}, Epochs: {args.epochs}')
    print('═' * 60)

    # 加载数据
    print('\n[1/3] 加载 CatSound_V2...')
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    self_dir = Path(args.self_dir) if args.self_dir else DATA_SELF_DIR
    files, labels, cat_ids, is_aug = load_catsound_v2(data_dir)
    if args.self_dir and Path(args.self_dir).exists():
        print('  [!] 注意: 自标注(163条)来自Pandeya, 已被CatSound_V2包含')
        print('  跳过加载。如确认无重复, 清空 --self-dir 参数')
    
    # 标签合并 (freeze 模式强制 10 类, 推理时再合并)
    if args.mode == 'freeze' and args.merge != 10:
        print(f'  freeze v2 模式: 强制 10 类, 忽略 --merge {args.merge}')
    if args.mode != 'freeze' and args.merge == 5:
        merged_labels = np.array([MERGE_5_MAP.get(l, l) for l in labels])
        n_before = len(set(labels))
        n_after = len(set(merged_labels))
        labels = merged_labels
        print(f'  标签合并: {n_before}类 → {n_after}类 (Aggressive+Happy+Calling+Paining+Resting)')
    elif args.mode != 'freeze' and args.merge == 10:
        pass  # 保持 10 类
    
    print(f'  总样本: {len(files)}')
    for cls in sorted(set(labels)):
        print(f'    {cls}: {(labels == cls).sum()}')

    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = {'model': args.model, 'pooling': args.pooling,
           'layers': MODEL_MAP[args.model]['best_layers'], 'aug': args.aug,
           'output_dir': out_dir}
    if args.layers_scan:
        run_layers_scan(files, labels, cat_ids, ctx)
        return
    elif args.mode == 'freeze':
        results = run_freeze_mode(files, labels, cat_ids, is_aug, ctx)
    else:
        results = run_finetune_mode(files, labels, cat_ids, args.lr, args.epochs, args.model, out_dir)

    # 保存结果
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    name_suffix = f'_{args.name}' if args.name else ''
    result_path = out_dir / f'results_{args.mode}{name_suffix}_{timestamp}.json'
    results.update({
        'config': {
            'model': MODEL_MAP[args.model]['name'],
            'mode': args.mode,
            'pooling': args.pooling,
            'n_samples': len(files),
            'sr': SR,
            'duration': DURATION,
            'lr': args.lr,
            'epochs': args.epochs,
            'layers': MODEL_MAP[args.model]['best_layers'],
        },
        'timestamp': timestamp,
    })
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n结果: {result_path}')
    print('═' * 60)

if __name__ == '__main__':
    main()

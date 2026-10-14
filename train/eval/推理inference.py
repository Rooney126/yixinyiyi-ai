"""
译芯怡意 — 猫声分类推理脚本脚本 v1
========================
加载 Wav2Vec2 嵌入 + 训练好的 MLP/SVM 分类器
对一条音频文件做实时推理

用法:
  python 推理inference.py --audio /path/to/cat.wav
  python 推理inference.py --audio /path/to/cat.wav --model mlp
  python 推理inference.py --audio /path/to/cat.wav --model svm --topk 3

输出:
  - 预测标签 + 置信度
  - 可选 Top-K 概率分布
# 推理管线（与 inference.py 重复，旧版）
"""

import os
import pickle
import warnings

os.environ['JOBLIB_TEMP_FOLDER'] = 'C:/temp'
warnings.filterwarnings('ignore')

import argparse
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor

try:
    from transformers import HubertModel, WavLMModel
    HAS_NEW_MODELS = True
except ImportError:
    HAS_NEW_MODELS = False

# ── 默认路径 (与训练脚本一致) ──
USER = os.environ.get('USERNAME', os.environ.get('USER', 'chen'))
BASE = Path(f'C:/Users/{USER}/Desktop/模型代码/训练结果')

# ── 参数 (必须与训练脚本一致) ──
SR = 16000
# DURATION/TARGET_LEN 不再用于裁剪, 保留全长
DURATION = 2.0
WAV2VEC2_MODEL = "facebook/wav2vec2-base"
LABELS = [
    "交配（低沉嘶吼）", "休息", "快乐", "愤怒（呜哇哇哇）",
    "打架（尖锐刺耳）", "母体呼叫（警笛）", "狩猎（短促的嗷呜）",
    "疼痛", "警告（呜~）（咕噜）", "防御(响尾蛇嘶哈)"
]


# =========================================================
# MLP 定义 (与训练脚本一致)
# =========================================================
class EmbeddingMLP(nn.Module):
    """768 → 64 → num_classes (~50K params)"""
    def __init__(self, input_dim=768, num_classes=10, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# 管理器 — 持有所有模型
# =========================================================
class CatSoundInference:
    """
    一次性加载所有模型，可对多段音频做推理。
    """

    # 支持的嵌入模型
    EMBED_MODELS = {
        'wav2vec2': ('facebook/wav2vec2-base', Wav2Vec2Model),
        'hubert': ('facebook/hubert-base-ls960', HubertModel if HAS_NEW_MODELS else None),
        'wavlm': ('microsoft/wavlm-base-plus', WavLMModel if HAS_NEW_MODELS else None),
    }

    def __init__(self, results_dir: Path = BASE, device=None, embed_model='wav2vec2'):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dir = Path(results_dir)
        self.embed_model = embed_model
        self._loaded = {}

        if embed_model not in self.EMBED_MODELS:
            raise ValueError(f"未知嵌入模型: {embed_model}, 可选: {list(self.EMBED_MODELS.keys())}")
        if embed_model != 'wav2vec2' and not HAS_NEW_MODELS:
            raise ImportError(f"{embed_model} 需要安装 transformers >= 4.20")

        print('🔊 猫声分类推理引擎')
        print(f'   设备: {self.device}')
        print(f'   嵌入模型: {embed_model}')
        print(f'   模型目录: {self.results_dir}')
        print()

    # ── 加载嵌入模型 (Wav2Vec2/HuBERT/WavLM) ──
    def _load_embedder(self):
        if self.embed_model in self._loaded:
            return self._loaded[self.embed_model]
        model_name, model_class = self.EMBED_MODELS[self.embed_model]
        print(f'  [1/3] 加载嵌入模型: {model_name}')
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = model_class.from_pretrained(model_name)
        model.to(self.device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self._loaded[self.embed_model] = (processor, model)
        print(f'        隐藏维度: {model.config.hidden_size}')
        return processor, model

    # ── 加载 MLP ──
    def _load_mlp(self):
        if 'mlp' in self._loaded:
            return self._loaded['mlp']
        ckpt_path = self.results_dir / 'mlp.pth'
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f'MLP 模型文件不存在: {ckpt_path}\n'
                f'请先运行 Wav2Vec2_嵌入训练.py 进行训练'
            )
        print(f'  [2/3] 加载 MLP 分类器: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        mlp = EmbeddingMLP(
            input_dim=ckpt['input_dim'],
            num_classes=ckpt['num_classes'],
        ).to(self.device)
        mlp.load_state_dict(ckpt['mlp_state_dict'])
        mlp.eval()
        self._loaded['mlp'] = (mlp, ckpt['scaler'], ckpt['label_encoder'])
        print(f'        类别数: {ckpt["num_classes"]}')
        return mlp, ckpt['scaler'], ckpt['label_encoder']

    # ── 加载 SVM/RF ──
    def _load_sklearn(self):
        if 'svm_rf' in self._loaded:
            return self._loaded['svm_rf']
        pkl_path = self.results_dir / 'svm_rf.pkl'
        if not pkl_path.exists():
            raise FileNotFoundError(
                f'SVM/RF 模型文件不存在: {pkl_path}\n'
                f'请先运行 Wav2Vec2_嵌入训练.py 进行训练'
            )
        print(f'  [2/3] 加载 SVM/RF 分类器: {pkl_path}')
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        self._loaded['svm_rf'] = data
        print('        含: SVM + RF + PCA')
        return data

    # ── 预处理 (与训练脚本完全一致) ──
    @staticmethod
    def preprocess(y: np.ndarray) -> np.ndarray:
        """峰值归一化, 保留全长 (HuBERT/WavLM 支持变长输入)"""
        peak = np.max(np.abs(y))
        if peak < 1e-10:
            raise ValueError(f'音频疑似静音或损坏 (peak={peak:.2e})')
        return y / peak

    # ── 嵌入提取 ──
    def extract_embedding(self, y: np.ndarray) -> np.ndarray:
        """音频 → 嵌入模型 768维嵌入 (用早期层融合)"""
        processor, model = self._load_embedder()
        y_proc = self.preprocess(y)
        inputs = processor(
            y_proc, sampling_rate=SR,
            return_tensors="pt", padding=True
        ).input_values.to(self.device)
        with torch.no_grad():
            outputs = model(inputs, output_hidden_states=True)
            # 取第2、3、4层均值 (Cauzinille et al. 验证初始层最好)
            hidden = outputs.hidden_states
            target = torch.stack([hidden[i] for i in [2, 3, 4]], dim=0).mean(dim=0)
            embedding = target.mean(dim=1).squeeze().cpu().numpy()
        return embedding

    # ── 从文件路径加载音频 ──
    @staticmethod
    def load_audio(path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f'音频文件不存在: {path}')
        y, sr = librosa.load(path, sr=SR, mono=True)
        if len(y) == 0:
            raise ValueError(f'音频文件为空: {path}')
        return y

    # ── MLP 推理 ──
    def predict_mlp(self, y: np.ndarray, topk: int = 1):
        """用 MLP 做推理，返回 (label, confidence) list"""
        mlp, scaler, le = self._load_mlp()
        emb = self.extract_embedding(y)
        emb_scaled = scaler.transform(emb.reshape(1, -1))
        emb_tensor = torch.FloatTensor(emb_scaled).to(self.device)
        with torch.no_grad():
            logits = mlp(emb_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        top_indices = np.argsort(probs)[::-1][:topk]
        results = []
        for idx in top_indices:
            label = le.inverse_transform([idx])[0] if hasattr(le, 'inverse_transform') else LABELS[idx]
            results.append((label, float(probs[idx])))
        return results

    # ── SVM 推理 ──
    def predict_svm(self, y: np.ndarray, topk: int = 1):
        """用 SVM 做推理"""
        data = self._load_sklearn()
        svm_model = data['svm_model']
        svm_scaler = data['svm_scaler']
        le = data['label_encoder']
        pca = data.get('pca')
        emb = self.extract_embedding(y)
        emb_scaled = svm_scaler.transform(emb.reshape(1, -1))
        if pca:
            emb_scaled = pca.transform(emb_scaled)
        # SVM decision_function 给出每个类的距离
        if hasattr(svm_model, 'decision_function'):
            scores = svm_model.decision_function(emb_scaled)[0]
            # 转为概率 (Platt scaling)
            probs = 1.0 / (1.0 + np.exp(-scores))
            probs /= probs.sum()  # 归一化
        else:
            probs = np.zeros(len(LABELS))
        top_indices = np.argsort(probs)[::-1][:topk]
        results = []
        for idx in top_indices:
            label = le.inverse_transform([idx])[0] if hasattr(le, 'inverse_transform') else LABELS[idx]
            results.append((label, float(probs[idx])))
        return results

    # ── RF 推理 ──
    def predict_rf(self, y: np.ndarray, topk: int = 1):
        """用 RF 做推理"""
        data = self._load_sklearn()
        rf_model = data['rf_model']
        le = data['label_encoder']
        pca = data.get('pca')
        emb = self.extract_embedding(y)
        emb_2d = emb.reshape(1, -1)
        if pca:
            emb_2d = pca.transform(emb_2d)
        probs = rf_model.predict_proba(emb_2d)[0]
        # RF predict_proba 是 list of arrays (每个类一个 array)
        # 需要还原成单一概率向量
        if isinstance(probs, list):
            # 多输出情况 → 取第一个输出
            probs = probs[0] if len(probs) > 0 else probs
        top_indices = np.argsort(probs)[::-1][:topk]
        results = []
        for idx in top_indices:
            label = le.inverse_transform([idx])[0] if hasattr(le, 'inverse_transform') else LABELS[idx]
            results.append((label, float(probs[idx])))
        return results

    # ── 统一推理接口 ──
    def predict(self, audio_input, model_type='mlp', topk=1):
        """
        统一推理接口.

        参数:
            audio_input: 音频文件路径 或 numpy 数组(16kHz mono)
            model_type: 'mlp' | 'svm' | 'rf'
            topk: 返回 top-K 预测结果

        返回:
            [(label, confidence), ...]
        """
        if isinstance(audio_input, (str, Path)):
            y = self.load_audio(str(audio_input))
        else:
            y = audio_input

        if model_type == 'mlp':
            return self.predict_mlp(y, topk)
        elif model_type == 'svm':
            return self.predict_svm(y, topk)
        elif model_type == 'rf':
            return self.predict_rf(y, topk)
        else:
            raise ValueError(f"未知模型类型: {model_type}，可选: mlp, svm, rf")


# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser(description='译芯怡意 — 猫声分类推理脚本')
    parser.add_argument('--audio', type=str, required=True,
                       help='音频文件路径 (.wav / .mp3)')
    parser.add_argument('--model', type=str, default='mlp',
                       choices=['mlp', 'svm', 'rf'],
                       help='使用的分类器 (默认: mlp)')
    parser.add_argument('--embed', type=str, default='wav2vec2',
                       choices=['wav2vec2', 'hubert', 'wavlm'],
                       help='嵌入模型类型 (默认: wav2vec2)')
    parser.add_argument('--topk', type=int, default=3,
                       help='返回 Top-K 预测 (默认: 3)')
    parser.add_argument('--results-dir', type=str, default=str(BASE),
                       help='训练结果目录')
    args = parser.parse_args()

    # 初始化
    engine = CatSoundInference(results_dir=args.results_dir, embed_model=args.embed)

    # 推理
    print(f'\n▶ 推理: {args.audio}')
    print(f'   模型: {args.model.upper()}, Top-{args.topk}')
    print()

    results = engine.predict(args.audio, model_type=args.model, topk=args.topk)

    print('═' * 50)
    print(f'  预测结果 (Top-{args.topk})')
    print('═' * 50)
    for i, (label, conf) in enumerate(results):
        bar = '█' * int(conf * 30) + '░' * (30 - int(conf * 30))
        print(f'  #{i + 1}: {bar} {conf:>5.1%}  {label}')
    print('═' * 50)


if __name__ == '__main__':
    main()

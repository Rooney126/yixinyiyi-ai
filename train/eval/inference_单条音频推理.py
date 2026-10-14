"""
译芯怡意 — 猫声情绪推理 (Freeze 模型)
=====================================
用法:
  # 单文件推理
  python inference.py --audio cat.wav

  # 批量推理 (目录下所有 mp3/wav)
  python inference.py --dir 大祭司猫语音频/maomaojisi888/ --report
# 统一推理接口：支持 HuBERT+MLP/SVM/RF 单条音频预测
"""

import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import argparse
from pathlib import Path

import joblib
import librosa
import numpy as np
import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor

# ── 配置 ──
SR = 16000
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 模型路径
BASE_DIR = Path(__file__).resolve().parent
FREEZE_DIR = BASE_DIR / '训练结果/hubert_v2/freeze_model_hubert_20260618_111717'
FINETUNE_DIR = BASE_DIR / '训练结果/hubert_v2/finetune_model'

# 10类 Pandeya → 中文标签
CN_LABELS = [
    'Angry',    # 0
    'Defence',  # 1
    'Fighting', # 2
    'Happy',    # 3
    'Hunting',  # 4 HuntingMind
    'Mating',   # 5
    'Mother',   # 6 MotherCall
    'Paining',  # 7
    'Resting',  # 8
    'Warning',  # 9
]

# 推理时合并映射 (训练10类SVM再合并, 避免类不平衡)
MERGE_6 = {
    'Angry': 'Aggressive', 'Defence': 'Aggressive',
    'Fighting': 'Aggressive', 'Warning': 'Aggressive',
    'Happy': 'Happy', 'Hunting': 'Happy',
    'Mating': 'Mating', 'Mother': 'MotherCall',
    'Paining': 'Paining', 'Resting': 'Resting',
}
MERGE_5 = {
    'Angry': 'Aggressive', 'Defence': 'Aggressive',
    'Fighting': 'Aggressive', 'Warning': 'Aggressive',
    'Happy': 'Happy', 'Hunting': 'Happy',
    'Mating': 'Calling', 'Mother': 'Calling',
    'Paining': 'Paining', 'Resting': 'Resting',
}


# =========================================================
# 模型加载
# =========================================================
class FreezeInference:
    """Freeze 模型推理: HuBERT 嵌入 + SVM 分类"""
    
    def __init__(self, model_dir=FREEZE_DIR):
        print(f'[加载] Freeze 模型: {model_dir}')
        t0 = time.time()
        
        # 加载 SVM 分类器
        data = joblib.load(str(model_dir / 'svm.pkl'))
        self.svm = data['svm']
        self.scaler = data['scaler']
        self.le = data['label_encoder']
        print(f'  SVM 已加载 ({len(self.le.classes_)} 类)')
        
        # 加载 HuBERT
        self.model = HubertModel.from_pretrained(
            'facebook/hubert-base-ls960', local_files_only=True
        )
        self.model.to(DEVICE)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            'facebook/hubert-base-ls960', local_files_only=True
        )
        self.layer_indices = [3, 4, 5]  # 12层扫描最优层
        print(f'  加载完成 ({time.time()-t0:.1f}s)')
    
    def _preprocess(self, y):
        """峰值归一化"""
        peak = np.max(np.abs(y))
        if peak < 1e-10:
            return np.zeros_like(y, dtype=np.float32)
        return y / peak
    
    @torch.no_grad()
    def extract_embedding(self, audio_path):
        """提取单条音频的 HuBERT 嵌入"""
        y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
        y = self._preprocess(y)
        
        inputs = self.processor(
            [y], sampling_rate=SR, return_tensors="pt", padding=True
        ).input_values.to(DEVICE)
        
        outputs = self.model(inputs, output_hidden_states=True)
        hidden = outputs.hidden_states
        layer_stack = torch.stack([hidden[i] for i in self.layer_indices], dim=0)
        emb = layer_stack.mean(dim=0).mean(dim=1)  # (1, 768)
        return emb.cpu().numpy()
    
    def predict(self, audio_path, topk=3, merge_map=None):
        """预测单条音频, 返回 topk 结果"""
        emb = self.extract_embedding(audio_path)
        emb_scaled = self.scaler.transform(emb)
        
        scores = self.svm.decision_function(emb_scaled)[0]
        scores = scores - scores.max()
        probs = np.exp(scores) / np.exp(scores).sum()
        top_indices = np.argsort(probs)[::-1][:topk]
        
        results = []
        for idx in top_indices:
            en_label = self.le.classes_[idx]
            prob = probs[idx]
            merged = merge_map.get(en_label, en_label) if merge_map else en_label
            results.append({
                'label': en_label,
                'merged': merged,
                'probability': round(float(prob), 4),
            })
        return results
    
    def predict_batch(self, audio_dir, topk=3, merge_map=None):
        """批量推理目录下所有音频, 返回统计"""
        audio_dir = Path(audio_dir)
        files = sorted(audio_dir.rglob('*.mp3')) + sorted(audio_dir.rglob('*.wav'))
        print(f'  扫描到 {len(files)} 条音频')
        
        all_preds = []
        for i, fp in enumerate(files):
            try:
                result = self.predict(fp, topk=1, merge_map=merge_map)
                all_preds.append(result[0]['merged'] if merge_map else result[0]['label'])
            except Exception as e:
                print(f'  [!] {fp.name}: {e}')
                all_preds.append(None)
            if (i + 1) % 20 == 0:
                print(f'  推理: {i+1}/{len(files)}')
        
        # 统计
        valid = [p for p in all_preds if p is not None]
        from collections import Counter
        counter = Counter(valid)
        total = len(valid)
        
        print('\n── 批量推理统计 ──')
        print(f'  成功: {total}/{len(files)}')
        for label, count in counter.most_common():
            pct = count / total * 100
            print(f'  {label}: {count} 条 ({pct:.1f}%)')
        
        return counter


def main():
    parser = argparse.ArgumentParser(description='猫声情绪推理 (Freeze 模型)')
    parser.add_argument('--audio', type=str, default=None, help='单条音频路径')
    parser.add_argument('--dir', type=str, default=None, help='批量推理目录')
    parser.add_argument('--model', type=str, default='freeze', choices=['freeze'],
                       help='模型类型 (当前仅 freeze)')
    parser.add_argument('--topk', type=int, default=3, help='返回 topK 结果')
    parser.add_argument('--merge', type=int, default=10, choices=[10, 6, 5],
                       help='输出: 10=原始Pandeya类, 6=Aggressive+Happy+..., 5=Aggressive+Happy+Calling+...')
    parser.add_argument('--report', action='store_true', help='显示统计报告')
    args = parser.parse_args()
    
    if not args.audio and not args.dir:
        parser.print_help()
        print('\n❌ 请指定 --audio 或 --dir')
        return
    
    # 初始化推理引擎
    infer = FreezeInference()
    merge_map = {10: None, 6: MERGE_6, 5: MERGE_5}.get(args.merge, None)
    
    if args.audio:
        # 单条推理
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f'❌ 文件不存在: {audio_path}')
            return
        
        print(f'\n── 推理结果: {audio_path.name} ──')
        results = infer.predict(audio_path, topk=args.topk, merge_map=merge_map)
        for r in results:
            bar = '█' * int(r['probability'] * 30)
            display = f'{r["label"]}' if args.merge == 10 else f'{r["merged"]} ({r["label"]})'
            print(f'  {display}')
            print(f'  置信度: {r["probability"]*100:.1f}% {bar}')
            print()
        
        top = results[0]
        display = f'{top["label"]}' if args.merge == 10 else f'{top["merged"]} ({top["label"]})'
        print(f'结果: {display} (置信度 {top["probability"]*100:.1f}%)')
    
    if args.dir:
        # 批量推理
        counter = infer.predict_batch(args.dir, topk=args.topk, merge_map=merge_map)


if __name__ == '__main__':
    main()

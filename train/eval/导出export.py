"""
译芯怡意 — 模型导出 (ONNX/TFLite)
=================================
将训练好的模型导出为 ONNX 格式, 可选转 TFLite.

用法:
  python 导出export.py --model-dir freeze_model_hubert_20240616_120000
  python 导出export.py --model-dir finetune_model
  python 导出export.py --model-dir ./model --format tflite
# 导出模型到 ONNX/TFLite（模型部署）
"""

import os, sys, warnings, argparse
warnings.filterwarnings('ignore')
from pathlib import Path
import torch
import numpy as np

# PKG_DIR = Path(__file__).parent  # 暂时不调用, 直接从命令行指定

def export_embedding_mlp(model_dir: Path, output_path: Path, input_dim=768, num_classes=10):
    """导出 freeze 模式的 MLP 分类头 (embeddingMLP: 768→64→num_classes)"""
    import torch.nn as nn

    class EmbeddingMLP(nn.Module):
        def __init__(self, input_dim, num_classes, dropout=0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes),
            )
        def forward(self, x):
            return self.net(x)

    ckpt = torch.load(model_dir / 'mlp.pth', map_location='cpu', weights_only=False)
    mlp = EmbeddingMLP(input_dim=ckpt.get('input_dim', input_dim),
                       num_classes=ckpt.get('num_classes', num_classes))
    mlp.load_state_dict(ckpt['model_state'])
    mlp.eval()

    dummy = torch.randn(1, ckpt['input_dim'])
    torch.onnx.export(
        mlp, dummy, str(output_path),
        input_names=['embedding'], output_names=['logits'],
        dynamic_axes={'embedding': {0: 'batch'}},
        opset_version=17,
    )
    print(f'✅ 已导出: {output_path}')
    return mlp, ckpt


def export_finetune_model(model_dir: Path, output_path: Path):
    """导出 HuBERTForSequenceClassification 全模型"""
    from transformers import HubertForSequenceClassification, Wav2Vec2Processor
    import torch

    processor = Wav2Vec2Processor.from_pretrained(str(model_dir))
    model = HubertForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    # 用真实音频维度导出: 1s @ 16kHz
    dummy = torch.randn(1, 16000)
    torch.onnx.export(
        model, dummy, str(output_path),
        input_names=['input_values'], output_names=['logits'],
        dynamic_axes={
            'input_values': {0: 'batch', 1: 'length'},
            'logits': {0: 'batch'},
        },
        opset_version=17,
        export_params=True,
    )
    print(f'✅ 已导出: {output_path}')
    return model


def main():
    parser = argparse.ArgumentParser(description='模型导出 ONNX/TFLite')
    parser.add_argument('--model-dir', type=str, required=True, help='模型目录')
    parser.add_argument('--output', type=str, default=None, help='输出路径 (默认: model_dir/model.onnx)')
    parser.add_argument('--format', choices=['onnx', 'tflite'], default='onnx', help='导出格式')
    parser.add_argument('--type', choices=['freeze', 'finetune'], default='freeze',
                       help='freeze=MLP分类头, finetune=全模型')
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output = Path(args.output) if args.output else model_dir / f'model.{args.format}'

    if not model_dir.exists():
        raise FileNotFoundError(f'模型目录不存在: {model_dir}')

    if args.type == 'freeze':
        mlp_path = model_dir / 'mlp.pth'
        if not mlp_path.exists():
            raise FileNotFoundError(f'未找到 MLP: {mlp_path}')
        export_embedding_mlp(model_dir, output)
    else:
        config_path = model_dir / 'config.json'
        if not config_path.exists():
            raise FileNotFoundError(f'未找到 config.json: {model_dir}')
        export_finetune_model(model_dir, output)

    # TFLite 转换 (需要 onnx2tf)
    if args.format == 'tflite':
        print('\n⚠ TFLite 转换需要 onnx-tf 和 tensorflow')
        print('  手动运行:')
        print(f'    onnx-tf convert -i {output} -o {model_dir}/model_tf/')
        print(f'    再用 TFLiteConverter 转 .tflite')


if __name__ == '__main__':
    main()

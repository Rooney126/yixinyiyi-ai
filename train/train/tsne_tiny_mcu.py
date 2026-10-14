# -*- coding: utf-8 -*-
r"""
t-SNE 可视化 — TinyMCUCatCNN 嵌入空间 (硬件端侧模型)

样式与 tsne_vadexp_per_class.png 完全一致 (2×5 逐类散点 + 总图),
复用 tsne_vadexp.py 的预处理管线 (VAD切分 → Mel 128×80), 
Student 输入为 64×40 (由 128×80 按 2×2 平均降采样, 与 kd_tiny_mcu_distill.py 一致),
嵌入 = GAP 后 32 维 (fc 之前).

用法:
  cd PROJECT_ROOT\train\train
  /path/to/user\AppData\Local\Programs\Python\Python310\python.exe tsne_tiny_mcu.py

输出:
  PROJECT_ROOT\train\output\tsne_tiny_mcu_fold3.png
  PROJECT_ROOT\train\output\tsne_tiny_mcu_per_class.png
"""

import os, sys, time, warnings
# 强制 UTF-8 输出 (Windows GBK 终端无法打印 emoji)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 中文字体修复
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

# ── 复用云端 t-SNE 脚本的预处理管线 ──
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsne_vadexp import (process_audio, load_file_list, LABELS,
                         DATA_DIR, OUTPUT_DIR, DEVICE)

# 硬件模型 (kd 蒸馏 fold3, 与云端 fold3 对应)
MODEL_PATH = Path(r'PROJECT_ROOT\train\output\kd_tiny_mcu\fold3_best.pth')


def downsample_mel(mel_128x80):
    """128×80 → 64×40 (Mel 轴 2→1 平均, 时间轴 2→1 平均), 与 kd 脚本一致"""
    h, w = mel_128x80.shape
    mel_64 = mel_128x80.reshape(h // 2, 2, w).mean(axis=1)
    mel_64_40 = mel_64.reshape(64, w // 2, 2).mean(axis=2)
    return mel_64_40.astype(np.float32)


class TinyMCUCatCNN_Embed(nn.Module):
    """与 kd_tiny_mcu_distill.py 的 TinyMCUCatCNN 一致, 返回 GAP 后 32 维嵌入"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        embed = self.gap(x).flatten(1)  # (B, 32)
        return embed


def main():
    print(f'\n{"█" * 55}')
    print('  t-SNE: TinyMCUCatCNN 嵌入空间 (kd fold3)')
    print(f'  {DEVICE=}')
    print(f'{"█" * 55}\n')

    # ── 加载模型 ──
    print('[0] 加载硬件模型...')
    model = TinyMCUCatCNN_Embed().to(DEVICE)
    model.eval()
    if not MODEL_PATH.exists():
        print(f'  ❌ 模型不存在: {MODEL_PATH}')
        return
    state = torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=False)
    p = sum(v.numel() for v in model.parameters())
    print(f'  ✅ {MODEL_PATH.name} 加载完成, 参数量: {p:,}')

    # ── 加载数据 (每类 ≤200 条, 与云端一致) ──
    print('[1] 加载数据取样 (每类 ≤200 条)...')
    files, labels = load_file_list(DATA_DIR, max_files_per_class=200)

    # ── 管线处理 + 嵌入提取 ──
    print('[2] 管线处理 + 嵌入提取 (128×80 → 降采样 64×40)...')
    t0 = time.time()
    all_embeds, all_labels_list = [], []
    n_empty = 0
    for i, fp in enumerate(files):
        mels = process_audio(fp)
        if not mels:
            n_empty += 1
            continue
        for mel in mels:
            mel_ds = downsample_mel(mel)
            mel_t = torch.FloatTensor(mel_ds).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                embed = model(mel_t).cpu().numpy().squeeze()  # (32,)
            all_embeds.append(embed)
            all_labels_list.append(labels[i])
        if (i + 1) % 200 == 0:
            print(f'  {i+1}/{len(files)} segs={len(all_embeds)} empty={n_empty} {time.time()-t0:.0f}s')

    if not all_embeds:
        print('  ❌ 无有效嵌入!')
        return

    embeds = np.stack(all_embeds, axis=0)
    label_idx = LabelEncoder().fit_transform(all_labels_list)
    unique_labels = np.unique(all_labels_list)
    print(f'  完成: {len(embeds)} 段嵌入, {time.time()-t0:.0f}s, 形状 {embeds.shape}')

    # ── t-SNE ──
    print(f'[3] t-SNE 降维 (perplexity={min(30, len(embeds)//3)})...')
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=min(30, len(embeds) // 3),
                random_state=42, max_iter=1000, verbose=0)
    embed_2d = tsne.fit_transform(embeds)

    # ── 绘图 (与云端同款) ──
    print('[4] 绘图...')
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    # 逐类图 2×5
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for i, label in enumerate(unique_labels):
        ax = axes[i]
        mask = label_idx == i
        ax.scatter(embed_2d[mask, 0], embed_2d[mask, 1],
                   c=[colors[i]], label=label, alpha=0.6, s=8)
        other = ~mask
        ax.scatter(embed_2d[other, 0], embed_2d[other, 1],
                   c='gray', alpha=0.1, s=4)
        ax.set_title(f'{label}\n({mask.sum()} 段)', fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    out2 = OUTPUT_DIR / 'tsne_tiny_mcu_per_class.png'
    fig.savefig(str(out2), dpi=150)
    plt.close()
    print(f'  ✅ 逐类图: {out2}')

    # 总图
    fig, ax = plt.subplots(figsize=(14, 10))
    for i, label in enumerate(unique_labels):
        mask = label_idx == i
        ax.scatter(embed_2d[mask, 0], embed_2d[mask, 1],
                   c=[colors[i]], label=label, alpha=0.6, s=30, edgecolors='none')
    ax.set_title(f't-SNE: TinyMCUCatCNN 嵌入空间 (kd fold3)\n'
                 f'{len(embeds)} 段, 10类', fontsize=14)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out1 = OUTPUT_DIR / 'tsne_tiny_mcu_fold3.png'
    fig.savefig(str(out1), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✅ 总图: {out1}')

    print(f'\n{"=" * 50}')
    print(f'  t-SNE 完成! 输出:')
    print(f'    {out1}')
    print(f'    {out2}')
    print(f'{"=" * 50}')


if __name__ == '__main__':
    main()

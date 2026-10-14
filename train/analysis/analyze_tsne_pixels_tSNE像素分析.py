"""
像素级分析 t-SNE 图片 — numpy 向量化版本，速度百倍提升
用法: python analyze_tsne_pixels.py <图片路径> [--crop x,y,w,h]
# t-SNE 图像像素级分析（辅助工具）
"""

import argparse
import warnings

warnings.filterwarnings('ignore')
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# ── 10 类 Pandeya 配色 ──
COLORS_10 = {
    'Angry':      (231,  76,  60),
    'Defence':    (192,  57,  43),
    'Fighting':   (255, 107, 107),
    'Warning':    (255, 159, 243),
    'Happy':      (46, 204, 113),
    'HuntingMind':(39, 174,  96),
    'Mating':     (155,  89, 182),
    'MotherCall': (52, 152, 219),
    'Paining':    (230, 126,  34),
    'Resting':    (149, 165, 166),
}

# ── 6 类合并配色 ──
COLORS_6 = {
    'Aggressive': (224,  49,  49),
    'Happy':      (240, 140,   0),
    'Mating':     (156,  54, 181),
    'MotherCall': (25, 113, 194),
    'Paining':    (230,  73, 128),
    'Resting':    (47, 158,  68),
}

MERGE_10TO6 = {
    'Angry':'Aggressive', 'Defence':'Aggressive',
    'Fighting':'Aggressive', 'Warning':'Aggressive',
    'Happy':'Happy', 'HuntingMind':'Happy',
    'Mating':'Mating', 'MotherCall':'MotherCall',
    'Paining':'Paining', 'Resting':'Resting',
}


def analyze_image(arr, palette, name, tolerance=40):
    """向量化像素匹配 + 空间分析"""
    h, w = arr.shape[:2]
    total_pixels = h * w
    print(f'  区域: {w}x{h} = {total_pixels:,} px')
    
    # ── 向量化匹配：对调色板中每个颜色算距离掩码 ──
    print('  [1] 向量化像素分类...')
    palette_arr = np.array(list(palette.values()), dtype=np.int16)  # (K, 3)
    palette_names = list(palette.keys())
    
    # arr_pixels: (h*w, 3), int16
    pixels = arr.reshape(-1, 3).astype(np.int16)
    
    class_pixels = {}
    class_masks = []
    matched = np.zeros(len(pixels), dtype=bool)
    
    for i, (name_cls, rgb) in enumerate(palette.items()):
        rgb_arr = np.array(rgb, dtype=np.int16)
        dists = np.sqrt(np.sum((pixels.astype(np.int16) - rgb_arr)**2, axis=1))
        mask = dists <= tolerance
        class_masks.append(mask)
        matched |= mask
        coords = np.column_stack(np.where(mask.reshape(h, w)))[:, ::-1]  # (x,y) = (col,row)
        class_pixels[name_cls] = coords
    
    unmatched = total_pixels - matched.sum()
    print(f'  匹配: {matched.sum():,} px ({100*matched.sum()/total_pixels:.1f}%)')
    print(f'  未匹配: {unmatched:,} px ({100*unmatched/total_pixels:.1f}%)')
    
    # 统计各色像素数
    for name_cls in palette:
        n = len(class_pixels[name_cls])
        print(f'    {name_cls:<12}: {n:>8,} px ({100*n/total_pixels:.1f}%)')
    
    # ── 中心 + 标准差 ──
    print('\n  [2] 空间分布 (中心 + 散布半径)')
    centers = {}
    for name_cls, pts in class_pixels.items():
        if len(pts) == 0:
            centers[name_cls] = np.array([np.nan, np.nan])
            continue
        xs, ys = pts[:,0].astype(float), pts[:,1].astype(float)
        cx, cy = xs.mean(), ys.mean()
        rx, ry = xs.std(), ys.std()
        centers[name_cls] = np.array([cx, cy])
        print(f'    {name_cls:<12}: 中心({cx:7.0f},{cy:7.0f})  σ=({rx:5.0f},{ry:5.0f})')
    
    # ── 类间中心距离 ──
    print('\n  [3] 类间中心距离矩阵 (≤100 表示严重重叠)')
    valid_classes = [c for c in palette if centers[c][0] is not np.nan]
    print(f'  {"":>14}', end='')
    for c in valid_classes:
        print(f'  {c:>10}', end='')
    print()
    for c1 in valid_classes:
        print(f'{c1:>14}', end='')
        for c2 in valid_classes:
            if c1 == c2:
                print(f'  {"—":>10}', end='')
            else:
                d = np.linalg.norm(centers[c1] - centers[c2])
                print(f'  {d:>7.0f}', end='')
        print()
    
    # ── 边界框重叠 ──
    print('\n  [4] 边界框重叠分析 (按重叠区域降序)')
    bboxes = {}
    for name_cls, pts in class_pixels.items():
        if len(pts) < 5:
            continue
        bboxes[name_cls] = {
            'xmin': pts[:,0].min(), 'xmax': pts[:,0].max(),
            'ymin': pts[:,1].min(), 'ymax': pts[:,1].max(),
        }
    
    overlaps = []
    classes = list(bboxes.keys())
    for i in range(len(classes)):
        for j in range(i+1, len(classes)):
            c1, c2 = classes[i], classes[j]
            b1, b2 = bboxes[c1], bboxes[c2]
            
            ox_min = max(b1['xmin'], b2['xmin'])
            ox_max = min(b1['xmax'], b2['xmax'])
            oy_min = max(b1['ymin'], b2['ymin'])
            oy_max = min(b1['ymax'], b2['ymax'])
            
            if ox_min < ox_max and oy_min < oy_max:
                overlap = (ox_max - ox_min) * (oy_max - oy_min)
                a1 = (b1['xmax'] - b1['xmin']) * (b1['ymax'] - b1['ymin'])
                a2 = (b2['xmax'] - b2['xmin']) * (b2['ymax'] - b2['ymin'])
                r1 = overlap / a1 if a1 > 0 else 0
                r2 = overlap / a2 if a2 > 0 else 0
                overlaps.append((c1, c2, overlap, r1, r2, (r1+r2)/2))
    
    overlaps.sort(key=lambda x: -x[5])
    
    print(f'  {"类1":<14} {"类2":<14} {"重叠px":>8} {"→1比例":>8} {"→2比例":>8} {"均值":>8}')
    print(f'  {"─"*56}')
    for c1, c2, oa, r1, r2, avg in overlaps:
        print(f'  {c1:<14} {c2:<14} {oa:>8.0f} {r1:>7.1%} {r2:>7.1%} {avg:>7.1%}')
    
    # ── if 6类, 自动做 10→6 合并 ──
    if name == '6类合并':
        print('\n  [5] 6类重叠热图 (低重叠←→高重叠)')
        conflict_matrix = {}
        for c1, c2, oa, r1, r2, avg in overlaps:
            conflict_matrix[(c1,c2)] = avg
        for c1 in sorted(set(c for pair in conflict_matrix for c in pair)):
            row = []
            for c2 in sorted(set(c for pair in conflict_matrix for c in pair)):
                if c1 == c2:
                    row.append('  —  ')
                else:
                    v = conflict_matrix.get((c1,c2), conflict_matrix.get((c2,c1), 0))
                    if v > 0.5:
                        row.append(' 🔴 ')
                    elif v > 0.2:
                        row.append(' 🟡 ')
                    else:
                        row.append(' 🟢 ')
            print(f'    {c1:<12}: {" ".join(row)}')
    
    # ── 如果是 10 类, 按 6 类合并计算 ──
    if name == '10类原始':
        print('\n  [5] 按 6 类合并后的重叠分析')
        # 合并点
        merged_centers = defaultdict(list)
        for c1, c2, oa, r1, r2, avg in overlaps:
            m1 = MERGE_10TO6[c1]
            m2 = MERGE_10TO6[c2]
            if m1 != m2:
                merged_centers[(m1,m2)].append(avg)
        
        # 统计合并后每对的重叠
        merge_conflicts = defaultdict(list)
        for c1, c2, oa, r1, r2, avg in overlaps:
            m1 = MERGE_10TO6[c1]
            m2 = MERGE_10TO6[c2]
            if m1 != m2:
                merge_conflicts[(m1,m2)].append((c1,c2,avg))
        
        for (m1,m2), items in sorted(merge_conflicts.items(), key=lambda x: -max(i[2] for i in x[1])):
            max_ov = max(i[2] for i in items)
            print(f'    {m1:<12} ↔ {m2:<12}: 最大重叠 {max_ov:.0%}  (来自 {[(i[0],i[1]) for i in items]})')
    
    return class_pixels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='像素级 t-SNE 分析')
    parser.add_argument('images', nargs='+', help='图片路径')
    parser.add_argument('--tolerance', type=int, default=40, help='颜色容差 (默认40)')
    parser.add_argument('--crop', nargs=4, type=int, metavar=('X','Y','W','H'),
                        help='裁剪区域 x,y,w,h (用于网格图子图)')
    
    args = parser.parse_args()
    
    for p_str in args.images:
        p = Path(p_str)
        if not p.exists():
            print(f'[!] 文件不存在: {p}')
            continue
        
        # 自动选调色板
        if '6class' in p.stem.lower():
            palette = COLORS_6
            name = '6类合并'
        else:
            palette = COLORS_10
            name = '10类原始'
        
        print(f'\n{"="*60}')
        print(f'  分析: {p.name}')
        print(f'  调色板: {name} ({len(palette)} 类)')
        print(f'  容差: {args.tolerance}')
        
        img = Image.open(p).convert('RGB')
        arr = np.array(img)
        
        if args.crop:
            x, y, w, h = args.crop
            arr = arr[y:y+h, x:x+w]
            print(f'  裁剪区: ({x},{y}) {w}x{h}')
        
        analyze_image(arr, palette, name, tolerance=args.tolerance)
    
    print('\n✅ 分析完成')

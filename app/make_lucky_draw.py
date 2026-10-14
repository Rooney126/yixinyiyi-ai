# -*- coding: utf-8 -*-
"""猫语翻译局 抽奖片头生成器 v1 原型
输入3条视频 -> 三格并排 + 标题 + 白边框 + 轮流高亮 + 名字轮播 -> 8s片头
用法: python make_lucky_draw.py
"""
import os, subprocess
from PIL import Image, ImageDraw, ImageFont

BASE = r"PROJECT_ROOT\app"
SRC_DIR = r"/path/to/dataset\视频素材"
OUT = os.path.join(SRC_DIR, "抽奖片头_原型.mp4")
WORK = os.path.join(SRC_DIR, "_work")
FFMPEG = os.path.join(BASE, "ffmpeg.exe")

W, H = 720, 1280
CELL_W, CELL_H = 220, 293   # 3:4 格子
GAP = 12                    # 格子间距
TOTAL_W = CELL_W * 3 + GAP * 2  # 720
Y0 = 460                    # 格子起始 y
FONT = r"C:\Windows\Fonts\msyhbd.ttc"

VIDEOS = [
    os.path.join(SRC_DIR, "抖音20261012-727561.mp4"),
    os.path.join(SRC_DIR, "抖音20261012-002687.mp4"),
    os.path.join(SRC_DIR, "发现更多精彩视频_-_抖音搜索.mp4"),
]
NAMES = ["@铲屎官A", "@铲屎官B", "@铲屎官C"]  # 原型占位，真实=投稿ID

os.makedirs(WORK, exist_ok=True)

def load_font(size):
    return ImageFont.truetype(FONT, size)

# ---------- Pillow 生成背景/边框/名字 ----------
print("[1/4] 生成视觉元素...")

# 背景：深色渐变 + 标题 + 副标题
bg = Image.new("RGB", (W, H), (18, 20, 28))
d = ImageDraw.Draw(bg)
for y in range(H):
    t = y / H
    c = (int(18 + 30 * t), int(20 + 32 * t), int(28 + 45 * t))
    d.line([(0, y), (W, y)], fill=c)
# 标题
title = "🎉 恭喜中奖 🎉"
tb = d.textbbox((0, 0), title, font=load_font(64))
d.text(((W - (tb[2] - tb[0])) // 2, 220), title, font=load_font(64), fill=(255, 215, 90))
sub = "本期猫语翻译局 · 投稿中奖公示"
sb = d.textbbox((0, 0), sub, font=load_font(34))
d.text(((W - (sb[2] - sb[0])) // 2, 330), sub, font=load_font(34), fill=(170, 175, 195))
bg.save(os.path.join(WORK, "bg.png"))

# 白色圆角边框（高亮用）：与格子同尺寸
frame = Image.new("RGBA", (CELL_W, CELL_H), (0, 0, 0, 0))
fd = ImageDraw.Draw(frame)
fd.rounded_rectangle([0, 0, CELL_W - 1, CELL_H - 1], radius=18, outline=(255, 255, 255), width=6)
frame.save(os.path.join(WORK, "frame.png"))

# 普通灰边框（低亮态）
frame_dim = Image.new("RGBA", (CELL_W, CELL_H), (0, 0, 0, 0))
fd2 = ImageDraw.Draw(frame_dim)
fd2.rounded_rectangle([0, 0, CELL_W - 1, CELL_H - 1], radius=18, outline=(120, 125, 140), width=4)
frame_dim.save(os.path.join(WORK, "frame_dim.png"))

# 名字卡（底部轮播）
name_img = Image.new("RGBA", (W, 120), (0, 0, 0, 0))
nd = ImageDraw.Draw(name_img)
nd.rounded_rectangle([140, 10, W - 140, 110], radius=30, fill=(30, 34, 48))
name_img.save(os.path.join(WORK, "name_bg.png"))

# ---------- 裁剪三条视频为格子尺寸 ----------
print("[2/4] 裁剪三条视频...")
cells = []
for i, v in enumerate(VIDEOS):
    c = os.path.join(WORK, f"cell_{i}.mp4")
    subprocess.run([FFMPEG, "-y", "-i", v, "-t", "8",
                    "-vf", f"scale={CELL_W}:{CELL_H}:force_original_aspect_ratio=increase,crop={CELL_W}:{CELL_H}",
                    "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", c],
                   capture_output=True)
    cells.append(c)

# ---------- ffmpeg 合成 ----------
print("[3/4] 合成片头...")
# 格子位置
xs = [0, CELL_W + GAP, (CELL_W + GAP) * 2]
cmd = [FFMPEG, "-y", "-i", os.path.join(WORK, "bg.png")]
for c in cells:
    cmd += ["-i", c]
cmd += ["-i", os.path.join(WORK, "frame.png"), "-i", os.path.join(WORK, "name_bg.png")]

# 名字文字图（三张，Pillow 直接画上名字）
for i, n in enumerate(NAMES):
    ni = Image.open(os.path.join(WORK, "name_bg.png")).convert("RGBA")
    ndd = ImageDraw.Draw(ni)
    txt = f"🎁 {n}"
    tb2 = ndd.textbbox((0, 0), txt, font=load_font(44))
    ndd.text(((W - (tb2[2] - tb2[0])) // 2, 32), txt, font=load_font(44), fill=(255, 255, 255))
    p = os.path.join(WORK, f"name_{i}.png")
    ni.save(p)
    cmd += ["-i", p]

# 滤镜图
# 输入顺序: 0=bg, 1-3=cells, 4=frame, 5=name_bg, 6,7,8=name_0..2
# 注意: cmd 里不再加 frame_dim
fps = ["[0:v]scale=720:1280,format=rgba[bgv]"]
for i in range(3):
    fps.append(f"[{i + 1}:v]format=rgba[cell{i}]")
fps.append("[4:v]format=rgba,split=3[fr0][fr1][fr2]")
fps.append("[5:v]format=rgba[nmbg]")
for i in range(3):
    fps.append(f"[{6 + i}:v]format=rgba[nm{i}]")

ov = f"[bgv][cell0]overlay={xs[0]}:{Y0}[ov1];"
ov += f"[ov1][cell1]overlay={xs[1]}:{Y0}[ov2];"
ov += f"[ov2][cell2]overlay={xs[2]}:{Y0}[ov3];"
ov += f"[ov3][nmbg]overlay=0:{H - 150}[ov4];"
ov += f"[ov4][fr0]overlay={xs[0]}:{Y0}:enable='lt(mod(t,8),2.7)'[ov5];"
ov += f"[ov5][fr1]overlay={xs[1]}:{Y0}:enable='between(mod(t,8),2.7,5.4)'[ov6];"
ov += f"[ov6][fr2]overlay={xs[2]}:{Y0}:enable='gte(mod(t,8),5.4)'[ov7];"
ov += f"[ov7][nm0]overlay=0:{H - 150}:enable='lt(t,2.7)'[ov8];"
ov += f"[ov8][nm1]overlay=0:{H - 150}:enable='between(t,2.7,5.4)'[ov9];"
ov += f"[ov9][nm2]overlay=0:{H - 150}:enable='gte(t,5.4)'[vout]"
fps.append(ov)

cmd += ["-filter_complex", ";".join(fps), "-map", "[vout]", "-t", "8",
        "-r", "30", "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p", OUT]
r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
if r.returncode != 0:
    print("FFMPEG ERROR:", r.stderr[-2500:])
else:
    print(f"[4/4] 完成: {OUT} ({os.path.getsize(OUT)//1024} KB)")

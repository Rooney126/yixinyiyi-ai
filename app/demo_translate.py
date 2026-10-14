# -*- coding: utf-8 -*-
"""
译芯怡意 — 猫叫翻译终端演示
用法（在 app 目录下执行）:
  python demo_translate.py                 # 打印 10 类情绪完整翻译文案映射表
  python demo_translate.py xxx.wav         # 翻译单个猫叫音频
  python demo_translate.py 目录路径/       # 批量翻译目录下所有 wav
"""
import glob
import os
import subprocess
import sys
import tempfile

# Windows 控制台统一 UTF-8 输出（配合 bat 里 chcp 65001，emoji/中文不乱码）
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import cloud_server as cs
import librosa
import noisereduce as nr
import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv", ".wmv"}


def extract_audio(video_path):
    """视频 → 临时 wav（16k 单声道），需要 ffmpeg"""
    tmp = os.path.join(tempfile.gettempdir(), "cat_extract.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", tmp],
            check=True, capture_output=True, timeout=120)
        return tmp
    except FileNotFoundError:
        print("  ✗ 视频需要 ffmpeg 提取音频，但系统没装 ffmpeg")
        print("    方案：a) 安装 ffmpeg 后重试  b) 先把视频转成 mp3/wav 再拖")
        return None
    except subprocess.CalledProcessError:
        print("  ✗ ffmpeg 提取音频失败，文件可能损坏或格式不支持")
        return None


def show_map():
    print("=" * 72)
    print("10 类情绪 → 翻译文案 完整映射")
    print("=" * 72)
    for i, label in enumerate(cs.LABELS, 1):
        emoji, state_desc, trans, label_cn = cs.DOME_MAP.get(label, ("❓", "", "未知", ""))
        advice = cs.ADVICE_MAP.get(label, "")
        print(f"{i:2d}. {label:12s} {emoji} {state_desc}（{label_cn}）")
        print(f"    翻译：{trans}")
        if advice:
            print(f"    建议：{advice}")
    print("=" * 72)


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


_LABEL_CN_OVERRIDE = {
    "MotherCall": "寻母",
    "HuntingMind": "狩猎",
}


def cn_label(lbl):
    """英文标签 → 中文类名"""
    if lbl in _LABEL_CN_OVERRIDE:
        return _LABEL_CN_OVERRIDE[lbl]
    return cs.DOME_MAP.get(lbl, ("", "", "", lbl))[3] or lbl


def seg_pipeline(wav):
    """切分音频，返回 [(t0秒, t1秒, mel或None), ...]（与 cs.process_audio 同一套管线，额外带时间戳）"""
    y, _ = librosa.load(wav, sr=cs.SR, mono=True, duration=cs.MAX_DUR)
    if len(y) < cs.SR * 0.3:
        return []
    rms = cs.compute_frame_rms(y)
    y_noise, _ = cs.collect_noise_samples(rms, y)
    if len(y_noise) >= cs.FRAME_LEN:
        stationary = float(np.min(rms)) < cs.MIN_RMS_THRESH
        try:
            y_d = nr.reduce_noise(y=y, sr=cs.SR, y_noise=y_noise, stationary=stationary,
                                  prop_decrease=cs.PROP_DECREASE, n_fft=cs.FRAME_LEN)
            y_d = np.nan_to_num(y_d)
        except Exception:
            y_d = y
    else:
        y_d = y
    rms_w = cs.compute_frame_rms(y_d)
    raw = cs.detect_segments(rms_w, quiet_pct=0.40)
    clean = cs.split_overlapping(raw, rms_w) if raw else []
    out = []
    for seg in clean:
        t0 = seg['onset_sample'] / cs.SR
        t1 = min(seg['offset_sample'], len(y_d)) / cs.SR
        seg_y = y_d[seg['onset_sample']:min(seg['offset_sample'], len(y_d))]
        out.append((t0, t1, cs.extract_mel(seg_y)))
    return out


def translate(wav):
    name = os.path.basename(wav)

    ext = os.path.splitext(wav)[1].lower()
    tmp_audio = None
    if ext in VIDEO_EXTS:
        tmp_audio = extract_audio(wav)
        if tmp_audio is None:
            return
        wav = tmp_audio

    try:
        segs = seg_pipeline(wav)
    finally:
        if tmp_audio and os.path.exists(tmp_audio):
            os.unlink(tmp_audio)
    if not segs:
        print(f"\n──── 翻译: {name} ────")
        print("  ✗ 没有听到猫叫声段")
        return

    # 先逐段算完（门卫+10类），再按版式输出
    seg_results = []
    cat_probs_sum = None
    n_cat = 0
    for (t0, t1, mel) in segs:
        if mel is None:
            seg_results.append(None)
            continue
        inp = mel[np.newaxis, np.newaxis, :, :].astype(np.float32)
        cat_p = float(_softmax(cs.binary_session.run(None, {"mel": inp})[0])[0, 1])
        if cat_p > 0.5:
            # 级联：只有门卫判定为猫叫的段才跑 10 类推理
            probs = _softmax(cs.session.run(None, {"mel": inp})[0])[0]
            idxs = np.argsort(probs)[::-1]
            seg_results.append((t0, t1, cat_p,
                                cs.LABELS[idxs[0]], float(probs[idxs[0]]),
                                cs.LABELS[idxs[1]], float(probs[idxs[1]])))
            cat_probs_sum = probs.copy() if cat_probs_sum is None else cat_probs_sum + probs
            n_cat += 1
        else:
            seg_results.append((t0, t1, cat_p, None, 0.0, None, 0.0))

    print(f"     门卫: 猫声段 {n_cat}/{len(segs)} 通过")
    if n_cat == 0:
        print("  ✗ 门卫拦截：听到声音了，但不像是猫叫")
        return

    print()
    print(f"──── 翻译: {name} ────")
    print('  分段 (共 %d 段):%s情绪相似度' % (len(segs), ' ' * 17))
    for i, r in enumerate(seg_results, 1):
        if r is None:
            print(f"   #{i}  （段提取失败，跳过）")
            continue
        t0, t1, cat_p, l1, p1, l2, p2 = r
        if l1 is None:
            print(f"   #{i}  {t0:4.1f}s–{t1:4.1f}s  噪声 ✗")
        else:
            print(f"   #{i}  {t0:4.1f}s–{t1:4.1f}s  猫叫 ✓   {cn_label(l1)} {p1:.0%} / {cn_label(l2)} {p2:.0%}")
    print("\n\n")

    vote = cat_probs_sum / n_cat
    idxs = np.argsort(vote)[::-1]
    top1, top2 = cs.LABELS[idxs[0]], cs.LABELS[idxs[1]]
    p1 = vote[idxs[0]]
    gap = p1 - vote[idxs[1]]
    fuzzy = gap < 0.20 and {top1, top2} == {"Paining", "Happy"}
    if fuzzy:
        label = "疼痛/开心之间"
    else:
        label = cn_label(top1)
    print("  ── 10类情绪置信度（投票后） ──")
    for rank, j in enumerate(idxs, 1):
        print(f"   {rank:2d}. {cn_label(cs.LABELS[j])}  {vote[j]:.1%}")


if __name__ == "__main__":
    cs.load_model()
    args = sys.argv[1:]
    if not args:
        show_map()
    elif os.path.isdir(args[0]):
        for w in sorted(glob.glob(os.path.join(args[0], "*.wav"))):
            translate(w)
    else:
        for w in args:
            translate(w)

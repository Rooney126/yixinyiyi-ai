# -*- coding: utf-8 -*-
"""分段分析：输出每段猫叫的时间戳+独立翻译结果"""
import sys, os
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
import numpy as np
import librosa
import cloud_server as cs

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def analyze(wav_path):
    y, _ = librosa.load(wav_path, sr=cs.SR, mono=True, duration=cs.MAX_DUR)
    rms = cs.compute_frame_rms(y)
    y_noise, _ = cs.collect_noise_samples(rms, y)
    if len(y_noise) >= cs.FRAME_LEN:
        stationary = float(np.min(rms)) < cs.MIN_RMS_THRESH
        try:
            y_d = cs.nr.reduce_noise(y=y, sr=cs.SR, y_noise=y_noise, stationary=stationary, prop_decrease=cs.PROP_DECREASE, n_fft=cs.FRAME_LEN)
            y_d = np.nan_to_num(y_d)
        except Exception:
            y_d = y
    else:
        y_d = y
    rms_w = cs.compute_frame_rms(y_d)
    raw = cs.detect_segments(rms_w, quiet_pct=0.40)
    clean = cs.split_overlapping(raw, rms_w)
    print(f"音频总长 {len(y)/cs.SR:.1f}s，切成 {len(clean)} 段\n")
    rows = []
    for i, seg in enumerate(clean, 1):
        t0 = seg['onset_sample'] / cs.SR
        t1 = min(seg['offset_sample'], len(y_d)) / cs.SR
        seg_y = y_d[seg['onset_sample']:min(seg['offset_sample'], len(y_d))]
        mel = cs.extract_mel(seg_y)
        if mel is None:
            rows.append((t0, t1, None)); continue
        inp = mel[np.newaxis, np.newaxis, :, :].astype(np.float32)
        cat_p = float(softmax(cs.binary_session.run(None, {"mel": inp})[0])[0, 1])
        probs = softmax(cs.session.run(None, {"mel": inp})[0])[0]
        idxs = np.argsort(probs)[::-1]
        top1, p1 = cs.LABELS[idxs[0]], probs[idxs[0]]
        top3 = ", ".join(f"{cs.LABELS[j]} {probs[j]:.0%}" for j in idxs[:3])
        trans = cs.DOME_MAP.get(top1, ("","","",""))[2]
        rows.append((t0, t1, dict(cat=cat_p, top1=top1, p1=float(p1), top3=top3, trans=trans)))
    for i, (t0, t1, r) in enumerate(rows, 1):
        if r is None:
            print(f"第{i}段  {t0:5.1f}s–{t1:5.1f}s  （mel提取失败，跳过）"); continue
        flag = "✓猫" if r['cat'] > 0.5 else "✗非猫"
        print(f"第{i}段  {t0:5.1f}s–{t1:5.1f}s（{t1-t0:4.1f}s）  [{flag} {r['cat']:.0%}]  {r['top1']} {r['p1']:.0%}  「{r['trans']}」")
        print(f"        Top3: {r['top3']}")
    # 汇总（门卫过滤后投票）
    valid = [r for _,_,r in rows if r and r['cat'] > 0.5]
    if valid:
        print(f"\n→ 进入最终投票的猫声段: {len(valid)}/{len(rows)}")
    return rows

if __name__ == "__main__":
    cs.load_model()
    analyze(sys.argv[1])

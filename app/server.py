"""
译芯怡意 App — Windows 原生版 v2
新管线: 11步预处理 + 多段加权投票
双击 run.bat 启动，手机浏览器打开即用
"""
import os, sys, json, warnings, tempfile, time, sqlite3, hashlib, uuid, shutil
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import librosa
import noisereduce as nr
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, send_from_directory, make_response
from functools import wraps
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# ═══════════════════════════════════════
#  配置 — 与训练管线参数完全一致
# ═══════════════════════════════════════

SR = 16000
MAX_DUR = 30.0

FRAME_LEN = 2048          # 帧长
HOP_LENGTH = 256           # 帧间步进

N_MELS = 128
FMIN = 50
FMAX = 8000
N_FFT = 512
MEL_FRAMES = 80            # 固定80帧 (1.28s)

PROP_DECREASE = 0.8        # 降噪强度
MIN_RMS_THRESH = 0.03      # 平稳/非平稳切换

MIN_NOISE_SAMPLES = 2048   # 噪声样本最短采样点数

ONSET_MULT = 3.5           # onset 阈值 = 安静值 × 3.5 (对齐训练脚本)
OFFSET_MULT = 3.0          # offset 阈值 = 安静值 × 3.0 (对齐训练脚本)
BUF_SEC = 0.075            # 缓冲 0.15s (0.075s/侧)
MIN_SEG_DUR = 0.25         # 最短段长

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PORT = 5173
BASE = Path(__file__).parent

FEEDBACK_FILE = BASE / 'data' / 'feedback_log.json'
DB_PATH = BASE / 'data' / 'app.db'
RECORDINGS_DIR = BASE / 'data' / 'recordings'
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=None)

# ─── IP 日志 (Cloudflare 真实来源) ───
@app.before_request
def log_real_ip():
    ip = request.headers.get("CF-Connecting-IP", request.remote_addr)
    if request.path.startswith("/api"):
        print(f">> {ip} {request.method} {request.path}")

# ─── 模型 ───
model = None
binary_model = None
LABELS = [
    "Angry","Defence","Fighting","Happy","HuntingMind",
    "Mating","MotherCall","Paining","Resting","Warning"
]

# ─── 10类翻译文案 v4 ───
DOME_MAP = {
    "Angry":       ("😡", "现在很火大", "我讨厌这样！不要这样！", "生气"),
    "Defence":     ("😰", "现在很紧张", "退后！别靠近！", "防御"),
    "Fighting":    ("💥", "充满敌意", "要打是吗？来啊！", "战斗"),
    "Warning":     ("⚠️", "正在划边界", "再走一步试试？", "警告"),
    "Happy":       ("😊", "它很开心", "嘿嘿～好开心！陪我玩嘛～", "开心"),
    "Resting":     ("😌", "非常满足", "呼噜呼噜～好舒服～", "满足"),
    "HuntingMind": ("🎯", "发现了猎物", "别吵！我看到猎物了！", "狩猎之心"),
    "Mating":      ("📞", "想恋爱了", "好想谈恋爱～噢～越想越难耐～", "发情"),
    "MotherCall":  ("🥺", "在找妈妈", "妈妈！你在哪呀？", "寻找妈妈"),
    "Paining":     ("😭", "可能不舒服", "呜…我难受…", "疼痛"),
}

PAIN_HAPPY_FUZZY = ("😶", "不太确定", "听起来又像不舒服又像开心", "不确定")

ADVICE_MAP = {
    "Angry":"猫咪很生气，如果您在做一些事情（比如剪指甲）请注意猫咪后续情绪走向，避免猫咪情绪爆发。可以的话，给它空间让它自己缓缓",
    "Defence":"猫咪现在很紧张，给它一点时间和空间，它可以自己调整过来",
    "Fighting":"猫咪正处于高度激动状态，注意让它和刺激源隔开，避免冲突升级",
    "Warning":"猫咪正在划清界限，先不要靠近，等它自己放松下来",
    "Happy":"猫咪现在很开心，趁这个状态多陪它玩吧",
    "Resting":"猫咪很放松很享受。如果它在睡觉，别打扰；如果是你的抚摸让它发出这个声音，继续保持～",
    "HuntingMind":"猫咪发现了感兴趣的东西，让它尽情探索就好",
    "Mating":"猫咪似乎在发情，注意关好门窗防止它跑出去，如果不想繁殖可以考虑绝育",
    "MotherCall":"猫咪在找妈妈或同伴，叫它一声它会安心",
    "Paining":"猫咪可能不太舒服，留意它的精神状态和身体情况，持续异常建议联系兽医",
}


# ═══════════════════════════════════════
#  模型架构
# ═══════════════════════════════════════

class MiniCatBinary(nn.Module):
    """猫/人二分类门卫 (Mel→16→32→64→2)"""
    def __init__(self, in_channels=1, n_mels=128, n_frames=80, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ═══════════════════════════════════════
#  模型架构 (FDAPMobileCatCNN 64ch)
# ═════════════════════════════════════==

class FDAPMobileCatCNN(nn.Module):
    def __init__(self, num_classes=10, fdap_bands=4):
        super().__init__()
        self.fdap_bands = fdap_bands
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((1, 2)))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 48, 3, padding=1),
            nn.BatchNorm2d(48), nn.ReLU(), nn.MaxPool2d((1, 2)))
        self.conv4 = nn.Sequential(
            nn.Conv2d(48, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.Identity())
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 96), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(96, num_classes))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        b, c, f, t = x.shape
        bs = f // self.fdap_bands
        fdap = [
            x[:, :, i * bs:(i + 1) * bs if i < 3 else f, :].mean(dim=[2, 3])
            for i in range(self.fdap_bands)
        ]
        x = torch.cat(fdap, dim=1)
        return self.classifier(x)


def load_model():
    global model, binary_model
    if model is not None:
        return model, binary_model

    # 10类情绪模型
    model = FDAPMobileCatCNN(num_classes=10).to(DEVICE)
    pth = BASE / "model" / "fold3_best.pth"
    if pth.exists():
        state = torch.load(str(pth), map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        print(f"  ✅ 已加载10类模型: {pth.name}")
    else:
        print("  ⚠️ 未找到10类模型权重，使用随机初始化")
    model.eval()

    # 猫/人二分类门卫
    binary_model = MiniCatBinary().to(DEVICE)
    binary_pth = BASE.parent / "models" / "minicat_binary.pt"
    if not binary_pth.exists():
        print(f"  ❌ 未找到门卫模型: {binary_pth}")
        print(f"     无门卫则无法区分猫声/人声，服务终止。")
        sys.exit(1)
    state = torch.load(str(binary_pth), map_location=DEVICE, weights_only=True)
    binary_model.load_state_dict(state)
    print(f"  ✅ 已加载门卫模型: {binary_pth.name}")
    binary_model.eval()

    return model, binary_model


# ═══════════════════════════════════════
#  11步预处理管线 (与训练完全一致)
# ═══════════════════════════════════════

def compute_frame_rms(y):
    """步骤1/4: 逐帧RMS能量"""
    n = len(y)
    frames = []
    for start in range(0, n - FRAME_LEN + 1, HOP_LENGTH):
        frame = y[start:start + FRAME_LEN]
        rms = np.sqrt(np.mean(frame ** 2))
        frames.append(rms)
    return np.array(frames, dtype=np.float32)


def collect_noise_samples(rms, y, bottom_pct=0.15, min_samples=MIN_NOISE_SAMPLES, hop=HOP_LENGTH):
    """
    步骤2: 收集底部 N% 帧的音频拼成噪声样本
    (所有底部帧拼接, 非最长连续段 — 对齐训练脚本)
    返回: (y_noise_array, used_pct)
    """
    n_frames = len(rms)
    n_bottom = max(1, int(n_frames * bottom_pct))
    sorted_indices = np.argsort(rms)[:n_bottom]
    noise_pieces = []
    total_samples = 0
    for idx in sorted_indices:
        start_s = idx * hop
        end_s = min((idx + 1) * hop, len(y))
        if end_s - start_s > 0:
            noise_pieces.append(y[start_s:end_s])
            total_samples += (end_s - start_s)
    if total_samples < min_samples and bottom_pct < 0.40:
        new_pct = min(bottom_pct + 0.05, 0.40)
        return collect_noise_samples(rms, y, new_pct, min_samples, hop)
    y_noise = np.concatenate(noise_pieces) if len(noise_pieces) > 0 else np.array([])
    return y_noise, bottom_pct


def detect_segments(rms, quiet_pct=0.40):
    """
    步骤5/6/7: Onset/Offset检测 + 过滤 + 缓冲
    安静值 = 底部 quiet_pct 帧的中位数
    onset = quiet × ONSET_MULT (3.5), offset = quiet × OFFSET_MULT (3.0)
    对齐训练脚本
    """
    n_frames = len(rms)
    if n_frames < 2:
        return []
    n_bottom = max(1, int(n_frames * quiet_pct))
    sorted_rms = np.sort(rms)
    quiet_value = float(np.median(sorted_rms[:n_bottom]))
    if quiet_value < 1e-10:
        quiet_value = np.percentile(rms, quiet_pct * 100)
    onset_thresh = quiet_value * ONSET_MULT
    offset_thresh = quiet_value * OFFSET_MULT
    if onset_thresh < 1e-8 or onset_thresh <= offset_thresh:
        if quiet_pct < 0.50:
            return detect_segments(rms, quiet_pct + 0.05)
        return []
    # 逐帧扫描
    segments = []
    in_call = False
    cur_onset = None
    for i in range(n_frames):
        if not in_call:
            if rms[i] > onset_thresh:
                cur_onset = i
                in_call = True
        else:
            if rms[i] < offset_thresh:
                segments.append((cur_onset, i))
                in_call = False
                cur_onset = None
    if in_call and cur_onset is not None:
        segments.append((cur_onset, n_frames - 1))
    # 过滤 + 缓冲
    buf_frames = int(BUF_SEC * SR / HOP_LENGTH)
    results = []
    for onset_frame, offset_frame in segments:
        orig_dur = (offset_frame - onset_frame) * HOP_LENGTH / SR
        if orig_dur < MIN_SEG_DUR:
            continue
        seg_max_rms = float(np.max(rms[onset_frame:offset_frame + 1]))
        if seg_max_rms < 0.01:
            continue
        buf_onset = max(0, onset_frame - buf_frames)
        buf_offset = min(n_frames - 1, offset_frame + buf_frames)
        results.append({
            'onset_sample': int(buf_onset * HOP_LENGTH),
            'offset_sample': int((buf_offset + 1) * HOP_LENGTH),
            'orig_dur': orig_dur,
        })
    return results


def split_overlapping(segments, rms):
    """步骤8: 重叠段切分"""
    if len(segments) <= 1:
        return segments
    result = [segments[0]]
    for seg in segments[1:]:
        prev = result[-1]
        prev_end = prev['offset_sample']
        cur_start = seg['onset_sample']
        if cur_start < prev_end:
            overlap_start_frame = prev_end // HOP_LENGTH
            overlap_end_frame = cur_start // HOP_LENGTH
            overlap_frames = rms[overlap_start_frame:overlap_end_frame + 1]
            if len(overlap_frames) > 0:
                min_frame_offset = int(np.argmin(overlap_frames))
                split_frame = overlap_start_frame + min_frame_offset
            else:
                split_frame = (overlap_start_frame + overlap_end_frame) // 2
            split_sample = split_frame * HOP_LENGTH
            prev['offset_sample'] = split_sample
            if split_sample > seg['onset_sample']:
                seg['onset_sample'] = split_sample
        result.append(seg)
    return result


def mel_preprocess(y_seg):
    """步骤9: Mel预处理 — 与训练一致"""
    # 峰值归一化
    peak = np.max(np.abs(y_seg))
    if peak < 1e-10:
        return None
    y_seg = y_seg / peak
    # Mel频谱
    mel = librosa.feature.melspectrogram(
        y=y_seg, sr=SR,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        n_fft=N_FFT, hop_length=HOP_LENGTH)
    # dB → [0,1]
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_min, mel_max = mel_db.min(), mel_db.max()
    if mel_max - mel_min < 1e-10:
        return None
    mel_norm = (mel_db - mel_min) / (mel_max - mel_min)
    # 固定80帧
    if mel_norm.shape[1] < MEL_FRAMES:
        mel_norm = np.pad(mel_norm, ((0, 0), (0, MEL_FRAMES - mel_norm.shape[1])))
    else:
        start = (mel_norm.shape[1] - MEL_FRAMES) // 2
        mel_norm = mel_norm[:, start:start + MEL_FRAMES]
    return mel_norm.astype(np.float32)


def process_audio(file_path):
    """
    完整11步管线处理单条音频

    返回: (list of mels (128,80) as np.ndarray, list of weights (orig_dur))
    """
    # 步骤0: 加载
    try:
        y, _ = librosa.load(file_path, sr=SR, mono=True)
    except Exception:
        import subprocess
        ffmpeg = BASE / 'ffmpeg.exe'
        wav_path = file_path.rsplit('.', 1)[0] + '_conv.wav'
        subprocess.run([
            str(ffmpeg), '-y', '-i', file_path,
            '-ar', str(SR), '-ac', '1', wav_path
        ], capture_output=True, timeout=30)
        y, _ = librosa.load(wav_path, sr=SR, mono=True)
        os.unlink(wav_path)

    # 约束最长30s
    max_samples = int(MAX_DUR * SR)
    if len(y) > max_samples:
        y = y[:max_samples]

    # 步骤1: 帧RMS
    rms = compute_frame_rms(y)
    if len(rms) < 2:
        return [], []

    # 步骤2: 底部15%所有帧拼接 → 噪声样本 (对齐训练脚本)
    y_noise, _ = collect_noise_samples(rms, y)

    # 步骤3: noisereduce 降噪
    if len(y_noise) >= FRAME_LEN:
        min_rms_val = float(np.min(rms)) if len(rms) > 0 else 0.0
        stationary = min_rms_val < MIN_RMS_THRESH
        try:
            y_work = nr.reduce_noise(
                y=y, sr=SR, y_noise=y_noise, stationary=stationary,
                prop_decrease=PROP_DECREASE, n_fft=FRAME_LEN)
            y_work = np.nan_to_num(y_work)
        except Exception:
            y_work = y
    else:
        y_work = y

    # 步骤4: 降噪后帧RMS
    rms_work = compute_frame_rms(y_work)
    if len(rms_work) < 2:
        return [], []

    # 步骤5-7: Onset/Offset检测 + 过滤 + 缓冲
    raw_segments = detect_segments(rms_work)
    # 对齐训练端(vadexp.py)/云端: VAD没切出段 → 丢弃，不硬塞整段
    if not raw_segments:
        return [], []

    # 步骤8: 重叠段切分
    clean_segments = split_overlapping(raw_segments, rms_work)

    # 步骤9: 每段Mel预处理
    mels, weights = [], []
    for seg in clean_segments:
        onset_s = seg['onset_sample']
        offset_s = min(seg['offset_sample'], len(y_work))
        seg_y = y_work[onset_s:offset_s]
        if len(seg_y) < N_FFT:
            continue
        mel = mel_preprocess(seg_y)
        if mel is not None:
            mels.append(mel)
            weights.append(seg['orig_dur'])

    return mels, weights


# ─── Flask 路由 ───
FRONTEND_DIR = BASE.parent / "frontend" / "dist"

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(FRONTEND_DIR / "assets", filename)

@app.route("/images/<path:filename>")
def images(filename):
    return send_from_directory(FRONTEND_DIR / "images", filename)

@app.route("/favicon.svg")
def favicon():
    return send_from_directory(FRONTEND_DIR, "favicon.svg")

@app.route("/icons.svg")
def icons():
    return send_from_directory(FRONTEND_DIR, "icons.svg")


# 简易频率限制: 每 IP 每分钟 10 次
_rate_limit_store = {}
def rate_limit(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        ip = request.remote_addr or request.headers.get("X-Forwarded-For", "unknown")
        now = time.time()
        # 清理 60s 前的记录
        if ip in _rate_limit_store:
            _rate_limit_store[ip] = [t for t in _rate_limit_store[ip] if now - t < 60]
            if len(_rate_limit_store[ip]) >= 10:
                print(f"[RATE_LIMIT] {ip} 被限流")
                return jsonify({"error": "请求太频繁，请稍后再试"}), 429
            _rate_limit_store[ip].append(now)
        else:
            _rate_limit_store[ip] = [now]
        return f(*args, **kwargs)
    return wrapper

@app.route("/api/translate/predict", methods=["POST"])
@rate_limit
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "没有音频文件"}), 400
        f = request.files["file"]
        orig_name = f.filename or "recording.webm"
        ext = os.path.splitext(orig_name)[1] or ".webm"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
            f.save(tf.name)
            tmp_path = tf.name

        m, bm = load_model()

        # 11步管线预处理 → 多段
        try:
            mels, weights = process_audio(tmp_path)
        except Exception as audio_err:
            os.unlink(tmp_path)
            return jsonify({"error": f"无法解析音频: {str(audio_err)}"}), 400

        if not mels:
            os.unlink(tmp_path)
            return jsonify({
                "error": "没有听到猫叫声，请靠近猫咪等它叫",
                "id": None,
            }), 400

        # 步骤 10.5: 猫/人二分类门卫
        mel_tensor = torch.FloatTensor(np.array(mels)).unsqueeze(1).to(DEVICE)  # (N, 1, 128, 80)
        with torch.no_grad():
            binary_logits = bm(mel_tensor)
            binary_scores = torch.softmax(binary_logits, dim=1)[:, 1].cpu().numpy()  # 猫得分
        # 过滤：保留猫得分 > 0.5 的段
        cat_mask = binary_scores > 0.5
        if not cat_mask.any():
            os.unlink(tmp_path)
            return jsonify({"error": "听到声音了，但不像是猫叫", "id": None}), 400
        mels = [mels[i] for i in range(len(mels)) if cat_mask[i]]
        weights = [weights[i] for i in range(len(weights)) if cat_mask[i]]
        filtered_tensor = mel_tensor[cat_mask]
        _gate_min = float(binary_scores[cat_mask].min()) if cat_mask.any() else 0.0
        _gate_total = len(binary_scores)
        _gate_passed = int(cat_mask.sum())

        # 每段推理 (10类)
        with torch.no_grad():
            logits = m(filtered_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # (N, 10)

        # 兜底：检查 NaN — 10 类模型在该输入上数值不稳定
        if np.isnan(probs).any():
            os.unlink(tmp_path)
            return jsonify({"error": "音频无法识别", "id": None}), 400

        # 步骤11: 等权投票
        vote = probs.mean(axis=0)

        # 排序
        idxs = np.argsort(vote)[::-1]
        top1_idx, top2_idx = int(idxs[0]), int(idxs[1])
        top1_label, top2_label = LABELS[top1_idx], LABELS[top2_idx]
        top1_prob, top2_prob = float(vote[top1_idx]), float(vote[top2_idx])

        gap = top1_prob - top2_prob

        # 检查 Paining ↔ Happy 混淆
        fuzzy_pain_happy = False
        if gap < 0.20:
            pair = {top1_label, top2_label}
            if pair == {"Paining", "Happy"}:
                fuzzy_pain_happy = True

        # 主文案
        if fuzzy_pain_happy:
            emoji_c, _, translation, state = PAIN_HAPPY_FUZZY
            advice = "它发出的声音在舒服和不太舒服之间，结合状态判断更准"
            label = "Paining"
        else:
            emoji_c, _, translation, state = DOME_MAP.get(top1_label, ("❓", "", "无法识别", "未知"))
            advice = ADVICE_MAP.get(top1_label, "多观察猫咪的状态。")
            label = top1_label

        confidence_level = "high" if top1_prob > 0.50 else "low"

        # 保存录音
        result_id = f"result-{int(time.time()*1000)}"
        audio_file = RECORDINGS_DIR / f"{result_id}.wav"
        try:
            shutil.copy2(tmp_path, str(audio_file))
        except Exception:
            pass
        os.unlink(tmp_path)

        # alternative
        show_alternative = gap < 0.20
        alternative = None
        if show_alternative:
            e2, _, t2, s2 = DOME_MAP.get(top2_label, ("❓", "", "未知", ""))
            alternative = {
                "label": top2_label,
                "translation": t2,
                "emoji": e2,
                "state": s2,
                "confidence": round(top2_prob, 3),
            }

        # distribution
        distribution = []
        for i in idxs[:3]:
            l = LABELS[i]
            e, _, t, s = DOME_MAP.get(l, ("❓", "", "未知", ""))
            distribution.append({
                "label": l, "prob": float(vote[i]),
                "emoji": e, "translation": t, "state": s,
            })

        # catIntent
        if fuzzy_pain_happy:
            _, _, cat_display, _ = PAIN_HAPPY_FUZZY
        elif confidence_level == "high":
            if show_alternative:
                cat_display = f'{translation}（也有可能是：{alternative["translation"]}）'
            else:
                cat_display = translation
        else:
            if show_alternative:
                cat_display = f'不太确定，可能是 {translation} 或 {alternative["translation"]}'
            else:
                cat_display = '不太确定'

        return jsonify({
            "id": result_id,
            "timestamp": int(time.time()*1000),
            "emotionState": state,
            "emotionEmoji": emoji_c,
            "catIntent": cat_display,
            "behaviorAdvice": advice,
            "confidence": round(top1_prob, 3),
            "label": label,
            "top3": [{"label": LABELS[i], "prob": f"{vote[i]:.1%}"} for i in idxs[:3]],
            "translation": translation,
            "confidence_level": confidence_level,
            "alternative": alternative,
            "distribution": distribution,
            "n_segments": len(mels),
            "segment_weights": [round(w, 3) for w in weights],
            "gate_score": round(_gate_min, 3),
            "gate_filtered": _gate_total - _gate_passed,
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "detail": traceback.format_exc()}), 500


# ─── 数据库与认证 (不变) ───
def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            cat_name TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id),
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT REFERENCES users(id),
            result_id TEXT,
            label TEXT,
            confidence REAL,
            feedback TEXT,
            timestamp INTEGER
        );
    ''')
    conn.commit()
    conn.close()

def get_user_by_token(token):
    if not token:
        return None
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute('SELECT u.id, u.username, u.cat_name FROM sessions s JOIN users u ON s.user_id=u.id WHERE s.token=?', (token,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "username": row[1], "cat_name": row[2]}
    return None

@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json()
    if not data or not data.get("username") or not data.get("password"):
        return jsonify({"error": "用户名和密码不能为空"}), 400
    username, password = data["username"].strip(), data["password"]
    if len(username) < 2:
        return jsonify({"error": "用户名至少2个字符"}), 400
    if len(password) < 4:
        return jsonify({"error": "密码至少4个字符"}), 400
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    try:
        uid = str(uuid.uuid4())[:8]
        pw_hash = hashlib.sha256(password.encode()).hexdigest()
        c.execute('INSERT INTO users (id, username, password_hash) VALUES (?,?,?)', (uid, username, pw_hash))
        conn.commit()
        token = str(uuid.uuid4())
        c.execute('INSERT INTO sessions (token, user_id) VALUES (?,?)', (token, uid))
        conn.commit()
        conn.close()
        return jsonify({"ok": True, "token": token, "user": {"id": uid, "username": username, "cat_name": ""}})
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error": "用户名已被注册"}), 409

@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data or not data.get("username") or not data.get("password"):
        return jsonify({"error": "用户名和密码不能为空"}), 400
    pw_hash = hashlib.sha256(data["password"].encode()).hexdigest()
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute('SELECT id, username, cat_name FROM users WHERE username=? AND password_hash=?', (data["username"].strip(), pw_hash))
    row = c.fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "用户名或密码错误"}), 401
    token = str(uuid.uuid4())
    c.execute('INSERT INTO sessions (token, user_id) VALUES (?,?)', (token, row[0]))
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "token": token, "user": {"id": row[0], "username": row[1], "cat_name": row[2]}})

@app.route("/api/auth/me", methods=["GET"])
def get_me():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = get_user_by_token(token)
    if not user:
        return jsonify({"error": "未登录"}), 401
    return jsonify({"ok": True, "user": user})

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "service": "译芯怡意", "version": "0.2.0"})

@app.route("/api/translate/feedback", methods=["POST"])
def submit_feedback():
    data = request.get_json()
    if not data:
        return jsonify({"error": "no data"}), 400
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = get_user_by_token(token)
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute('INSERT INTO feedback (user_id, result_id, label, confidence, feedback, timestamp) VALUES (?,?,?,?,?,?)',
        (user["id"] if user else None, data.get("result_id", ""), data.get("label", ""),
         data.get("confidence", 0.0), data.get("feedback", "unsure"), data.get("timestamp", int(time.time() * 1000))))
    conn.commit()
    conn.close()
    rid = data.get("result_id", "")
    if rid:
        meta_path = RECORDINGS_DIR / f'{rid}.json'
        if meta_path.exists():
            try:
                with open(meta_path) as mf:
                    meta = json.load(mf)
                meta['feedback'] = data.get('feedback', 'unsure')
                with open(meta_path, 'w') as mf:
                    json.dump(meta, mf)
            except Exception:
                pass
        fb_map = {'good': '反馈准', 'unsure': '反馈一般', 'bad': '反馈不准'}
        fb = data.get('feedback', '')
        if fb in fb_map:
            fb_dir = RECORDINGS_DIR / fb_map[fb]
            fb_dir.mkdir(exist_ok=True)
            src = RECORDINGS_DIR / f'{rid}.wav'
            if src.exists():
                try:
                    shutil.move(str(src), str(fb_dir / f'{rid}.wav'))
                except Exception:
                    pass
    return jsonify({"ok": True})

@app.route("/api/translate/pending/<result_id>", methods=["DELETE"])
def discard_pending(result_id):
    audio = RECORDINGS_DIR / f'{result_id}.wav'
    if audio.exists():
        try:
            audio.unlink()
            return jsonify({"ok": True})
        except Exception:
            pass
    return jsonify({"ok": True}), 200

@app.route("/api/translate/feedback", methods=["GET"])
def get_feedback():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    user = get_user_by_token(token)
    if not user:
        return jsonify({"error": "需要登录"}), 401
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute('SELECT result_id, label, confidence, feedback, timestamp FROM feedback WHERE user_id=? ORDER BY timestamp DESC', (user["id"],))
    rows = c.fetchall()
    conn.close()
    records = [{"result_id": r[0], "label": r[1], "confidence": r[2], "feedback": r[3], "timestamp": r[4]} for r in rows]
    return jsonify({"records": records, "total": len(records)})

@app.route("/<path:path>")
def spa_fallback(path):
    if path.startswith("api/"):
        return jsonify({"error": "not found"}), 404
    return send_from_directory(FRONTEND_DIR, "index.html")


if __name__ == "__main__":
    cert = BASE / "cert.pem"
    key = BASE / "key.pem"
    use_ssl = cert.exists() and key.exists()

    print("  🐱 译芯怡意 App v2 — 新管线 11步 + 多段投票")
    print(f"  模型: A_aug 折3 (87.9% F1, σ=0.7%)")
    print(f"  原文案 v3 已更新: 10类情绪新文案")
    print(f"  实时访问: https://your-server.example")
    print(f"  停止: 关闭此窗口即停止服务")

    print("  初始化数据库...")
    init_db()
    print("  ✅ 数据库就绪")

    print("  加载模型...")
    load_model()
    print(f"  ✅ 模型加载完成（{DEVICE.upper()}）")

    limit_ts = int(time.time()) - 21600
    orphaned = 0
    for f in RECORDINGS_DIR.glob('*.wav'):
        try:
            ts = int(f.stem.split('-')[-1]) // 1000
            if ts < limit_ts:
                f.unlink()
                orphaned += 1
        except Exception:
            pass
    if orphaned:
        print(f'  🧹 清理了 {orphaned} 条无人处理的遗留录音')

    scheme = "https" if use_ssl else "http"
    print(f"""
╔═══════════════════════════════════════╗
║    🐱 译芯怡意 App v2 (Windows 版)  ║
║    A_aug 折3 87.9% F1 / σ=0.7%      ║
║                                       ║
║    本机浏览器:                        ║
║    {scheme}://localhost:{PORT}        ║
║                                       ║
║    手机浏览器 (同WiFi热点):           ║
║    {scheme}://192.168.137.79:{PORT}      ║
║                                       ║
║    {'🔒 HTTPS 已启用' if use_ssl else '⚠️ HTTP 模式'}
╚═══════════════════════════════════════╝
""")
    ssl_ctx = (str(cert), str(key)) if use_ssl else None
    app.run(host="0.0.0.0", port=PORT, debug=False, ssl_context=ssl_ctx)

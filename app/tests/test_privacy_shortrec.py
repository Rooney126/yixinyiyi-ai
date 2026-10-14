"""隐私声明版本字段 + 短录音服务端拒绝路径（任务配套测试）"""
import io
import re
import wave

import numpy as np


def _current_privacy_version(cloud):
    """从 cloud_server 源码取当前声明版本（2026-09-25 外部已从 v1 升到 v2，
    测试改为跟随源码而不是写死，声明再升级测试不烂）"""
    src = open(cloud.__file__, encoding="utf-8").read()
    m = re.search(r'"privacy_version":\s*"(v[^"]+)"', src)
    assert m, "cloud_server.py 里找不到 privacy_version 常量"
    return m.group(1)


def _gate_pass_model(cloud, monkeypatch):
    """假模型：门卫二分类 class-1 得分 ≈0.98（过门卫），主模型 top1=Defence。
    让 predict 走到成功响应，验证 privacy_version 字段。"""
    def run(_output_names, d):
        x = list(d.values())[0]
        out = np.zeros((x.shape[0], 10), dtype=np.float32)
        out[:, 1] = 6.0
        return (out,)
    S = type("S", (), {"run": staticmethod(run)})
    monkeypatch.setattr(cloud, "load_model", lambda: (S(), S()))


def _wav_bytes(seconds=0.25, sr=16000):
    """内存里生成 wav：默认 0.25 秒静音（< 服务端 0.3s 最短门限）"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.zeros(int(sr * seconds), dtype=np.int16).tobytes())
    return buf.getvalue()


def _meowish_wav(sr=16000):
    """1.6s 有叫声段的 wav：0.2s 静音 + 1.0s 2kHz 正弦 + 0.4s 静音，
    VAD 能切出有效段（峰幅 0.2、段长 > 0.25s），配合假模型走到成功响应"""
    t = np.arange(sr, dtype=np.float64) / sr
    tone = (0.2 * np.sin(2 * np.pi * 2000 * t)).astype(np.float32)
    y = np.concatenate([np.zeros(int(sr * 0.2), np.float32), tone,
                        np.zeros(int(sr * 0.4), np.float32)])
    pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def test_predict_response_has_privacy_version(client, cloud, monkeypatch):
    """成功响应必须带 privacy_version（前端据此强制重弹隐私声明），值与源码常量一致"""
    _gate_pass_model(cloud, monkeypatch)
    r = client.post("/api/translate/predict",
                    data={"file": (io.BytesIO(_meowish_wav()), "ok.wav")})
    assert r.status_code == 200, r.get_data(as_text=True)
    j = r.get_json()
    assert j["privacy_version"] == _current_privacy_version(cloud)
    assert j["label"] == "Defence"              # 假模型的预期输出，确认走到了成功分支


def test_short_wav_rejected_with_vad_message(client, cloud, monkeypatch):
    """≤0.3s 的短录音被服务端拒绝：400 + "没有听到猫叫声"（前端引导文案的映射目标）"""
    _gate_pass_model(cloud, monkeypatch)
    r = client.post("/api/translate/predict",
                    data={"file": (io.BytesIO(_wav_bytes(0.25)), "short.wav")})
    assert r.status_code == 400
    assert "没有听到猫叫声" in r.get_json()["error"]
    assert r.get_json()["id"] is None


def test_silent_long_wav_also_rejected(client, cloud, monkeypatch):
    """1.2s 纯静音：长度够但 VAD 无叫声段，同样 400（文案相同，但前端只对短录音加引导）"""
    _gate_pass_model(cloud, monkeypatch)
    r = client.post("/api/translate/predict",
                    data={"file": (io.BytesIO(_wav_bytes(1.2)), "silent.wav")})
    assert r.status_code == 400
    assert "没有听到猫叫声" in r.get_json()["error"]
"""驻置模式：X-Source 白名单（L3）+ stations 表与 3 接口（L4）

红线：走 conftest 的 /tmp 测试库，真实 data/ 零接触；
predict 识别管线不被本文件改动（X-Source 只改落库的 source 字段）。
"""
import io
import os
import sqlite3
import time
import wave

import numpy as np

NEG_LABELS = ("Angry", "Defence", "Fighting", "Warning", "Paining")


def _conn():
    conn = sqlite3.connect(os.environ["YXY_DB_PATH"])
    conn.row_factory = sqlite3.Row
    return conn


def _meowish_wav(sr=16000):
    """1.6s 有叫声段的 wav（配合假模型走通 predict 成功分支）"""
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


def _fake_model(cloud, monkeypatch, top_label="Defence"):
    """假模型：门卫全过、主模型 top1=top_label（不碰真实 ONNX）"""
    idx = cloud.LABELS.index(top_label)

    def run(_output_names, d):
        x = list(d.values())[0]
        out = np.zeros((x.shape[0], 10), dtype=np.float32)
        out[:, idx] = 6.0
        return (out,)

    S = type("S", (), {"run": staticmethod(run)})
    monkeypatch.setattr(cloud, "load_model", lambda: (S(), S()))


def _hb(client, **fields):
    body = {"device_uid": "station-test1", "name": "驻置设备"}
    body.update(fields)
    return client.post("/api/station/heartbeat", json=body)


# ─────────────── L3：X-Source 白名单 ───────────────

def test_predict_xsource_station_lands_in_records(client, cloud, monkeypatch):
    """X-Source: station + ?uid= → records 行 source='station'、device_uid 落对"""
    _fake_model(cloud, monkeypatch)
    r = client.post("/api/translate/predict?uid=station-e2e",
                    data={"file": (io.BytesIO(_meowish_wav()), "evt.wav")},
                    headers={"X-Source": "station"})
    assert r.status_code == 200, r.get_data(as_text=True)
    rid = r.get_json()["id"]
    conn = _conn()
    try:
        row = conn.execute("SELECT * FROM records WHERE result_id=?", (rid,)).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row["source"] == "station" and row["device_uid"] == "station-e2e"


def test_predict_xsource_invalid_falls_back_to_web(client, cloud, monkeypatch):
    """X-Source 非白名单值（伪造/拼错）→ 回落 'web'，不进驻置口径"""
    _fake_model(cloud, monkeypatch)
    r = client.post("/api/translate/predict",
                    data={"file": (io.BytesIO(_meowish_wav()), "evt.wav")},
                    headers={"X-Source": "hacker"})
    assert r.status_code == 200
    rid = r.get_json()["id"]
    conn = _conn()
    try:
        src = conn.execute("SELECT source FROM records WHERE result_id=?",
                           (rid,)).fetchone()[0]
    finally:
        conn.close()
    assert src == "web"


def test_predict_xsource_default_web_unchanged(client, cloud, monkeypatch):
    """不带 X-Source 的老请求 → source='web'（主站行为零变化）"""
    _fake_model(cloud, monkeypatch)
    r = client.post("/api/translate/predict",
                    data={"file": (io.BytesIO(_meowish_wav()), "web.wav")})
    assert r.status_code == 200
    rid = r.get_json()["id"]
    conn = _conn()
    try:
        src = conn.execute("SELECT source FROM records WHERE result_id=?",
                           (rid,)).fetchone()[0]
    finally:
        conn.close()
    assert src == "web"


# ─────────────── L4：stations 表 + heartbeat ───────────────

def test_l4_stations_table_exists(cloud):
    conn = _conn()
    try:
        row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table'"
                           " AND name='stations'").fetchone()
        assert row is not None and "device_uid" in row["sql"]
        cols = {r[1] for r in conn.execute("PRAGMA table_info(stations)")}
    finally:
        conn.close()
    assert {"device_uid", "name", "first_seen", "last_seen", "battery", "charging",
            "uptime_s", "triggered_total", "uploaded_total"} <= cols


def test_l4_heartbeat_upsert_keeps_first_seen(client):
    """连续两次心跳：单行 upsert，last_seen 更新、first_seen 不变"""
    assert _hb(client, triggered_total=1, uploaded_total=0).status_code == 200
    conn = _conn()
    try:
        row1 = dict(conn.execute("SELECT * FROM stations").fetchone())
    finally:
        conn.close()
    time.sleep(0.02)   # 保证毫秒时间戳前移
    assert _hb(client, triggered_total=3, uploaded_total=1).status_code == 200
    conn = _conn()
    try:
        rows = [dict(r) for r in conn.execute("SELECT * FROM stations")]
    finally:
        conn.close()
    assert len(rows) == 1
    row2 = rows[0]
    assert row2["first_seen"] == row1["first_seen"]
    assert row2["last_seen"] > row1["last_seen"]
    assert row2["triggered_total"] == 3 and row2["uploaded_total"] == 1


def test_l4_heartbeat_validation(client):
    """字段校验：违规一律 400 中文提示，库里零行"""
    cases = [
        {"device_uid": ""},                                  # 缺设备号
        {"device_uid": "x" * 65},                            # 超 64
        {"device_uid": "station-a", "battery": 1.5},         # battery 超界
        {"device_uid": "station-a", "battery": "高"},        # battery 非数字
        {"device_uid": "station-a", "triggered_total": -1},  # 负计数
        {"device_uid": "station-a", "uptime_s": "abc"},      # 非整数
    ]
    for body in cases:
        r = client.post("/api/station/heartbeat", json=body)
        assert r.status_code == 400, body
        assert "error" in r.get_json()
    conn = _conn()
    try:
        assert conn.execute("SELECT COUNT(*) FROM stations").fetchone()[0] == 0
    finally:
        conn.close()


def test_l4_heartbeat_optional_battery_and_name_truncated(client):
    """battery/charging 缺省存 NULL；name 截 20 字"""
    assert _hb(client, name="很" * 30).status_code == 200
    conn = _conn()
    try:
        row = dict(conn.execute("SELECT * FROM stations").fetchone())
    finally:
        conn.close()
    assert row["battery"] is None and row["charging"] is None
    assert len(row["name"]) == 20


# ─────────────── L4：status / events ───────────────

def test_l4_status_requires_key(client):
    assert client.get("/api/station/status").status_code == 403
    assert client.get("/api/station/status?key=wrong").status_code == 403
    assert "口令" in client.get("/api/station/status").get_json()["error"]


def test_l4_status_devices_online_and_today_count(client, cloud, add_record):
    """对 key：设备卡字段齐全；online=2 分钟内有心跳；today_count 只数本设备今日驻置记录"""
    assert _hb(client, triggered_total=2, uploaded_total=2,
               battery=0.66, charging=True).status_code == 200
    now = cloud._now_ms()
    cloud.save_record("result-st-today", "Happy", "开心", "", "", "😊", 0.8,
                      source="station", device_uid="station-test1", created_at=now)
    cloud.save_record("result-st-other", "Angry", "生气", "", "", "😡", 0.8,
                      source="station", device_uid="station-other", created_at=now)
    cloud.save_record("result-st-web", "Happy", "开心", "", "", "😊", 0.9,
                      source="web", device_uid="station-test1", created_at=now)

    r = client.get(f"/api/station/status?key={cloud.STATS_KEY}")
    assert r.status_code == 200
    j = r.get_json()
    assert len(j["devices"]) == 1
    d = j["devices"][0]
    assert d["device_uid"] == "station-test1" and d["online"] is True
    assert d["battery"] == 0.66 and d["charging"] == 1
    assert d["today_count"] == 1 and d["accepted_total"] == 1
    assert d["triggered_total"] == 2 and d["uploaded_total"] == 2
    assert "metrics" in j and "daily" in j["metrics"]

    # 心跳变旧 → online False（直接把 last_seen 拨回 3 分钟前）
    conn = _conn()
    try:
        conn.execute("UPDATE stations SET last_seen=? WHERE device_uid=?",
                     (now - 3 * 60 * 1000, "station-test1"))
        conn.commit()
    finally:
        conn.close()
    j2 = client.get(f"/api/station/status?key={cloud.STATS_KEY}").get_json()
    assert j2["devices"][0]["online"] is False


def test_l4_status_bp_metrics(client, cloud, add_record):
    """BP 指标：健康相关今日/近7日计数 + 近7日日均置信度折线（只算 source='station'）"""
    now = cloud._now_ms()
    cloud.save_record("result-bp1", "Paining", "疼痛", "", "", "😭", 0.42,
                      source="station", device_uid="station-test1", created_at=now)
    cloud.save_record("result-bp2", "Happy", "开心", "", "", "😊", 0.91,
                      source="station", device_uid="station-test1",
                      created_at=now - 86400_000)
    cloud.save_record("result-bp3", "Angry", "生气", "", "", "😡", 0.5,
                      source="web", device_uid="station-test1", created_at=now)
    j = client.get(f"/api/station/status?key={cloud.STATS_KEY}").get_json()
    m = j["metrics"]
    assert m["health_today"] == 1 and m["health_week"] == 1   # 健康口径只有 Paining；web 的 Angry 不计
    daily = m["daily"]
    assert len(daily) == 7
    today = daily[-1]
    assert today["total"] == 1 and today["health"] == 1
    assert today["avg_confidence"] == 0.42
    yesterday = daily[-2]
    assert yesterday["total"] == 1 and yesterday["avg_confidence"] == 0.91


def test_l4_events_requires_key(client):
    assert client.get("/api/station/events").status_code == 403


def test_l4_events_filters_order_and_limit_cap(client, cloud, add_record):
    """events：时间倒序、hours/uid 过滤、limit 封顶 200"""
    now = cloud._now_ms()
    # 3 条 station（本设备 2 条：1 条 2h 内 1 条 30h 前；他设备 1 条）+ 1 条 web
    cloud.save_record("result-ev1", "Happy", "开心", "", "", "😊", 0.8,
                      source="station", device_uid="station-a",
                      created_at=now - 3600_000)
    cloud.save_record("result-ev2", "Angry", "生气", "", "", "😡", 0.6,
                      source="station", device_uid="station-a",
                      created_at=now - 30 * 3600_000)
    cloud.save_record("result-ev3", "Resting", "满足", "", "", "😌", 0.7,
                      source="station", device_uid="station-b",
                      created_at=now - 7200_000)
    cloud.save_record("result-ev4", "Happy", "开心", "", "", "😊", 0.9,
                      source="web", device_uid="station-a", created_at=now)

    key = cloud.STATS_KEY
    j = client.get(f"/api/station/events?key={key}").get_json()
    ids = [e["result_id"] for e in j["events"]]
    assert ids == ["result-ev1", "result-ev3"]   # 倒序、无 web、默认 24h 窗口（30h 前的 ev2 不在）

    j2 = client.get(f"/api/station/events?key={key}&hours=48").get_json()
    assert [e["result_id"] for e in j2["events"]] == \
        ["result-ev1", "result-ev3", "result-ev2"]
    j3 = client.get(f"/api/station/events?key={key}&uid=station-a&hours=48").get_json()
    assert [e["result_id"] for e in j3["events"]] == ["result-ev1", "result-ev2"]
    j4 = client.get(f"/api/station/events?key={key}&hours=abc").status_code
    assert j4 == 400

    # limit 封顶 200：造 205 条 → 默认最多回 200 行
    for i in range(205):
        cloud.save_record(f"result-cap{i:03d}", "Happy", "开心", "", "", "😊", 0.8,
                          source="station", device_uid="station-cap",
                          created_at=now + i)
    j5 = client.get(f"/api/station/events?key={key}&hours=48&uid=station-cap").get_json()
    assert len(j5["events"]) == 200 and j5["total"] == 205


def test_l4_stations_cleared_between_tests(client, cloud):
    """conftest _isolate 清表清单含 stations（本测试首行应为空）"""
    conn = _conn()
    try:
        assert conn.execute("SELECT COUNT(*) FROM stations").fetchone()[0] == 0
    finally:
        conn.close()
    assert client.get(f"/api/station/status?key={cloud.STATS_KEY}").get_json()["devices"] == []

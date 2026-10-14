"""安全深审专项：路径穿越 / SQL注入 / XSS / 信息泄露 / 畸形输入 / 体积上限"""
import io
import json

import numpy as np

# ─── 路径穿越 ───

def test_static_route_traversal_blocked(client):
    """真正逃逸出 frontend 目录的 URL 一律 404，且不落 SPA 兜底"""
    for path in ("/..%2Fcloud_server.py",
                 "/%2e%2e/%2e%2e/data/app.db",
                 "/frontend/../../cloud_server.py",
                 "/frontend/../%2e%2e/%2e%2e/data/app.db"):
        r = client.get(path)
        assert r.status_code == 404, path
        assert r.get_json() == {"error": "not found"}    # 不是兜底的 200 页面


def test_static_route_normal_paths_unaffected(client):
    """合法静态文件照常服务，未知路径照常 SPA 兜底"""
    assert client.get("/cat-archive.html").status_code == 200
    r = client.get("/no-such-page")
    assert r.status_code == 200 and "译芯怡意" in r.get_data(as_text=True)


def test_pending_delete_rejects_traversal_ids(client, cloud):
    """pending 删除接口：穿越类 result_id 全被拦，不碰文件系统。
    解码后含 / 的（../、a/b）在路由层就匹配不到单段规则 → 405；
    反斜杠、盘符冒号、裸 ".." 进到 handler → 400。两种都是拦截，文件都不被动。"""
    target = cloud.RECORDINGS_DIR.parent / "outside.wav"   # 录音目录外的诱饵文件
    target.write_bytes(b"secret")
    for rid in ("..%2F..%2Foutside", "a%2Fb"):
        assert client.delete(f"/api/translate/pending/{rid}").status_code == 405, rid
    for rid in ("..%5C..%5Coutside", "C%3A%5Cevil", "..", ".%2E"):
        r = client.delete(f"/api/translate/pending/{rid}")
        assert r.status_code == 400, rid
        assert "不合法" in r.get_json()["error"]
    assert target.exists()                                  # 诱饵文件原封不动
    target.unlink()


def test_pending_delete_normal_id_still_works(client, cloud):
    rid = "result-normal-123"
    f = cloud.RECORDINGS_DIR / f"{rid}.wav"
    f.write_bytes(b"x")
    assert client.delete(f"/api/translate/pending/{rid}").status_code == 200
    assert not f.exists()


# ─── SQL 注入（参数化查询应天然免疫，这里锁死行为）───

def test_sql_injection_in_cat_name_harmless(client, make_cat):
    cat = make_cat(name="'; DROP TABLE cats;")   # 18 字，过长度校验
    assert cat["name"] == "'; DROP TABLE cats;"  # 原样存储，没被执行
    r = client.get("/api/cats")                    # 表还在，能正常查
    assert r.status_code == 200
    assert [c["name"] for c in r.get_json()["cats"]] == ["'; DROP TABLE cats;"]


def test_sql_injection_in_query_params_harmless(client, add_record):
    add_record(device_uid="u_real")
    # uid 注入字符串匹配不到任何行
    for uid in ("x' OR '1'='1", "u_real' --", "%"):
        j = client.get(f"/api/records/unassigned?uid={uid}").get_json()
        assert j["records"] == [], uid


def test_sql_injection_in_assign_cat_id(client, add_record):
    rid = add_record()
    r = client.post(f"/api/records/{rid}/assign",
                    json={"cat_id": "cat-' OR 1=1 --"})
    assert r.status_code == 404                    # 只是"没找到这只猫"，无注入效果


# ─── XSS：周报 HTML（猫名/表情进 <title>/<h1>/头像）───

def test_weekly_report_escapes_user_input(client, make_cat, add_record):
    cat = make_cat(name="<svg onload=x>", emoji="<b>猫")
    add_record(cat_id=cat["id"], label="Paining",
               translation="<script>alert(1)</script>呜")
    r = client.get(f"/api/cats/{cat['id']}/report/weekly")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    for raw in ("<svg onload", "<b>猫", "<script>alert"):
        assert raw not in html, raw                 # 没有可执行的原始标签
    assert "&lt;svg onload=x&gt;" in html           # 以转义形式展示
    assert "&lt;b&gt;猫" in html


def test_weekly_report_normal_cat_unaffected(client, make_cat, add_record):
    cat = make_cat(name="普通猫", emoji="🐱")
    add_record(cat_id=cat["id"], label="Paining", translation="呜…我难受…")
    html = client.get(f"/api/cats/{cat['id']}/report/weekly").get_data(as_text=True)
    assert "普通猫 的情绪周报" in html
    assert "呜…我难受…" in html                     # 正常文案不被转义破坏


def test_stats_page_escapes_log_input(client, cloud):
    """calls.jsonl 里伪造的 UA / XFF IP（均可客户端伪造）进管理页前被转义"""
    log = cloud.BASE / "data" / "calls.jsonl"
    evil = {"t": 1, "ip": "1.2.3.4<img src=x>", "uid": "",
            "ts": "2026-09-07 09:00:00", "result": "<script>bad()</script>",
            "label": "", "ua": "<img src=x onerror=alert(1)>"}
    try:
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(evil, ensure_ascii=False) + "\n")
        r = client.get(f"/stats?key={cloud.STATS_KEY}")
        assert r.status_code == 200
        html = r.get_data(as_text=True)
        assert "<script>bad()" not in html
        assert "<img src=x" not in html
        assert "&lt;img src=x" in html
    finally:
        log.unlink(missing_ok=True)                 # 不污染同会话其他测试


# ─── 信息泄露：predict 500 不再回传 traceback ───

def test_predict_500_hides_traceback(client, cloud, monkeypatch):
    def boom():
        raise RuntimeError("模型加载失败 secret-path-xyz")
    monkeypatch.setattr(cloud, "load_model", boom)
    r = client.post("/api/translate/predict",
                    data={"file": (io.BytesIO(b"RIFFfake"), "a.wav")})
    assert r.status_code == 500
    j = r.get_json()
    assert "detail" not in j                        # traceback 字段已移除
    body = r.get_data(as_text=True)
    assert "Traceback" not in body and "File \"" not in body


def test_predict_no_file_400(client):
    r = client.post("/api/translate/predict", data={})
    assert r.status_code == 400
    assert "没有音频文件" in r.get_json()["error"]


def test_predict_oversize_413(client):
    """超过 32MB 上限的 body 被 413 挡下（无需模型参与）"""
    big = b"\0" * (33 * 1024 * 1024)
    r = client.post("/api/translate/predict",
                    data={"file": (io.BytesIO(big), "huge.wav")})
    assert r.status_code == 413


def test_predict_weird_extension_sanitized(client, cloud, monkeypatch):
    """畸形后缀（路径类字符）被替换为 .webm，不炸临时文件创建"""
    monkeypatch.setattr(cloud, "load_model",
                        lambda: (type("S", (), {"run": staticmethod(
                            lambda *_: np.zeros((1, 10), np.float32))})(),) * 2)
    r = client.post("/api/translate/predict",
                    data={"file": (io.BytesIO(b"garbage-bytes"),
                                   "x.<injection>?:")})
    assert r.status_code == 400                     # 解码失败，但不是 500/崩溃


# ─── feedback 畸形输入 ───

def test_feedback_type_confusion_400(client):
    assert client.post("/api/translate/feedback",
                       json={"confidence": "abc"}).status_code == 400
    assert client.post("/api/translate/feedback",
                       json={"timestamp": "xyz"}).status_code == 400
    assert client.post("/api/translate/feedback",
                       json={"confidence": None}).status_code == 400


def test_feedback_long_fields_truncated(client, cloud):
    log = cloud.BASE / "data" / "feedback.jsonl"
    try:
        r = client.post("/api/translate/feedback", json={
            "result_id": "R" * 5000, "label": "L" * 5000,
            "feedback": "F" * 5000, "confidence": 0.9})
        assert r.status_code == 200
        last = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])
        assert len(last["result_id"]) == 128
        assert len(last["label"]) == 64
        assert len(last["feedback"]) == 64
    finally:
        log.unlink(missing_ok=True)


# ─── 口令校验 ───

def test_backup_wrong_key_length_independent(client):
    """错误口令（含超长猜测）一律 403，恒定时间比较不影响结果"""
    for key in ("", "x", "kx8m2p7", "demo-key ", "A" * 10000):
        r = client.get(f"/api/export/backup?key={key}")
        assert r.status_code == 403, key

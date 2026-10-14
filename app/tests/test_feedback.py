"""识别页一键反馈：result_feedback 表（L1）/ 反馈提交接口（L2）/ stats 质量区块（L4）

红线：全部走 conftest 的 /tmp 测试库（YXY_DB_PATH），真实 data/ 零接触；
旧 feedback 表与旧 JSONL 通道的既有用例在原文件，本文件只测新增能力。
"""
import os
import sqlite3

import pytest

LABELS = ["Angry", "Defence", "Fighting", "Happy", "HuntingMind",
          "Mating", "MotherCall", "Paining", "Resting", "Warning"]

# 旧库（M1 之前）的最小镜像：users/sessions/feedback 结构与真实旧 app.db 同型
LEGACY_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, created_at INTEGER);
CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, user_id TEXT, created_at INTEGER);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT REFERENCES users(id),
    result_id TEXT,
    label TEXT,
    confidence REAL,
    feedback TEXT,
    timestamp INTEGER
)
"""


def _conn():
    conn = sqlite3.connect(os.environ["YXY_DB_PATH"])
    conn.row_factory = sqlite3.Row
    return conn


def _insert_fb(result_id, is_correct, correct_label="", note="", source="web",
               device_uid="u_x", created_at=None, updated_at=None):
    """直接 SQL 插一行 result_feedback（L1/L4 造数用，绕过接口）"""
    conn = _conn()
    try:
        conn.execute(
            "INSERT INTO result_feedback (result_id, is_correct, correct_label, note,"
            " source, device_uid, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (result_id, 1 if is_correct else 0, correct_label, note, source,
             device_uid, created_at or 1762512000000, updated_at or 1762512000000))
        conn.commit()
    finally:
        conn.close()


# ─────────────── L1 数据层 ───────────────

def test_l1_table_with_unique_and_index(cloud):
    """全新库导入后：表存在、result_id 唯一约束、created_at 索引都在"""
    conn = _conn()
    try:
        row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table'"
                           " AND name='result_feedback'").fetchone()
        assert row is not None, "result_feedback 表没建出来"
        assert "result_id" in row["sql"] and "UNIQUE" in row["sql"]
        idx = conn.execute("SELECT name FROM sqlite_master WHERE type='index'"
                           " AND tbl_name='result_feedback'").fetchall()
        assert any(i["name"] == "idx_result_feedback_time" for i in idx)
    finally:
        conn.close()


def test_l1_unique_enforced_at_db_level(cloud):
    """唯一约束真实生效：同 result_id 第二次 INSERT 必须 IntegrityError"""
    conn = _conn()
    try:
        args = ("result-uni", 1, "", "", "web", "u_x", 1762512000000, 1762512000000)
        conn.execute("INSERT INTO result_feedback (result_id, is_correct, correct_label,"
                     " note, source, device_uid, created_at, updated_at)"
                     " VALUES (?,?,?,?,?,?,?,?)", args)
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO result_feedback (result_id, is_correct, correct_label,"
                         " note, source, device_uid, created_at, updated_at)"
                         " VALUES (?,?,?,?,?,?,?,?)", args)
    finally:
        conn.close()


def test_l1_column_defaults(cloud):
    """correct_label/note/source/device_uid 默认值与任务书一致"""
    conn = _conn()
    try:
        conn.execute("INSERT INTO result_feedback (result_id, is_correct, created_at,"
                     " updated_at) VALUES (?,?,?,?)",
                     ("result-dft", 0, 1762512000000, 1762512000000))
        conn.commit()
        r = conn.execute("SELECT * FROM result_feedback WHERE result_id='result-dft'").fetchone()
        assert r["correct_label"] == "" and r["note"] == ""
        assert r["source"] == "web" and r["device_uid"] == ""
    finally:
        conn.close()


def test_l1_init_db_legacy_zero_change(cloud, tmp_path, monkeypatch):
    """含旧 feedback（≥1 行）+ cats/records 数据的库副本走一次导入：
    旧表 schema 与内容逐表逐行不变，新表照常建出（旧数据无损口径）"""
    legacy = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(legacy))
    conn.executescript(LEGACY_SCHEMA)
    conn.executescript(cloud.DB_SCHEMA)     # cats/records（已迁移过的库形态）
    now = 1762512000000
    conn.execute("INSERT INTO users VALUES ('u_old', ?)", (now,))
    conn.execute("INSERT INTO sessions VALUES ('s_old', 'u_old', ?)", (now,))
    conn.execute("INSERT INTO feedback (result_id, label, confidence, feedback, timestamp)"
                 " VALUES ('result-old', 'Happy', 0.9, 'good', ?)", (now,))
    conn.execute("INSERT INTO feedback (result_id, label, confidence, feedback, timestamp)"
                 " VALUES ('result-old2', 'Angry', 0.8, 'bad', ?)", (now,))
    conn.execute("INSERT INTO cats (id, name, color, emoji, birthday, note, archived,"
                 " created_at, updated_at) VALUES ('cat-old','老猫','#B5D8CC','🐱','','',0,?,?)",
                 (now, now))
    conn.execute("INSERT INTO records (result_id, cat_id, label, state, confidence,"
                 " source, device_uid, created_at) VALUES"
                 " ('result-old','cat-old','Happy','开心',0.9,'web','u_old',?)", (now,))
    conn.commit()

    def snapshot(c):
        tables = [r[0] for r in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
            " AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        return {t: (c.execute("SELECT sql FROM sqlite_master WHERE name=?", (t,)).fetchone()[0],
                    [tuple(r) for r in c.execute(f"SELECT * FROM {t}")])
                for t in tables}

    before = snapshot(conn)
    conn.close()

    monkeypatch.setattr(cloud, "DB_PATH", legacy)
    cloud.init_db()

    conn = sqlite3.connect(str(legacy))
    after = snapshot(conn)
    conn.close()
    for t, v in before.items():
        assert after.get(t) == v, f"旧表 {t} 的 schema/数据被改动"
    assert "result_feedback" in after, "导入后新表没建出来"


def test_l1_backup_includes_result_feedback(client, cloud, add_record):
    """/api/export/backup：新数组带上，旧 feedback 数组保留"""
    rid = add_record()
    _insert_fb(rid, is_correct=False, correct_label="Angry", note="其实是在生气",
               device_uid="u_bk")
    r = client.get(f"/api/export/backup?key={cloud.STATS_KEY}")
    assert r.status_code == 200
    j = r.get_json()
    assert "feedback" in j, "旧 feedback 数组不能丢"
    assert len(j["result_feedback"]) == 1
    fb = j["result_feedback"][0]
    assert fb["result_id"] == rid and fb["is_correct"] == 0
    assert fb["correct_label"] == "Angry" and fb["note"] == "其实是在生气"


# ─────────────── L2 接口层 ───────────────

def _fb_rows(result_id):
    conn = _conn()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM result_feedback WHERE result_id=?", (result_id,))]
    finally:
        conn.close()


def test_l2_correct_feedback_insert(client, add_record):
    """准确：插入 is_correct=1 一行，updated=false，source/device_uid 落对"""
    rid = add_record()
    r = client.post(f"/api/records/{rid}/feedback?uid=u_ab", json={"is_correct": True})
    assert r.status_code == 200
    assert r.get_json() == {"ok": True, "updated": False}
    rows = _fb_rows(rid)
    assert len(rows) == 1
    assert rows[0]["is_correct"] == 1 and rows[0]["correct_label"] == ""
    assert rows[0]["source"] == "web" and rows[0]["device_uid"] == "u_ab"


def test_l2_uid_falls_back_to_cookie(client, add_record):
    """不带 ?uid= 时从 yxy_uid cookie 取（与 predict 同款逻辑）"""
    rid = add_record()
    client.set_cookie("yxy_uid", "u_from_cookie")
    r = client.post(f"/api/records/{rid}/feedback", json={"is_correct": True})
    assert r.status_code == 200
    assert _fb_rows(rid)[0]["device_uid"] == "u_from_cookie"


def test_l2_wrong_feedback_requires_valid_label(client, add_record):
    """不准确：缺 correct_label / 标签不在 10 类内 → 400 中文，库里零行"""
    rid = add_record()
    r1 = client.post(f"/api/records/{rid}/feedback", json={"is_correct": False})
    assert r1.status_code == 400 and "正确" in r1.get_json()["error"]
    r2 = client.post(f"/api/records/{rid}/feedback",
                     json={"is_correct": False, "correct_label": "Dog"})
    assert r2.status_code == 400 and "情绪" in r2.get_json()["error"]
    assert _fb_rows(rid) == []


def test_l2_wrong_feedback_ok_with_label(client, add_record):
    """不准确 + 合法标签 → 落库 is_correct=0 + correct_label"""
    rid = add_record(label="Happy")
    r = client.post(f"/api/records/{rid}/feedback",
                    json={"is_correct": False, "correct_label": "Angry", "note": "在哈气"})
    assert r.status_code == 200
    row = _fb_rows(rid)[0]
    assert row["is_correct"] == 0 and row["correct_label"] == "Angry"
    assert row["note"] == "在哈气"


def test_l2_is_correct_must_be_json_bool(client, add_record):
    """is_correct 非 JSON 布尔（字符串/数字/缺失）→ 400 中文"""
    rid = add_record()
    for body in ({"is_correct": "true"}, {"is_correct": 1}, {"is_correct": None}, {}):
        r = client.post(f"/api/records/{rid}/feedback", json=body)
        assert r.status_code == 400, body
        assert "布尔" in r.get_json()["error"]
    assert _fb_rows(rid) == []


def test_l2_note_too_long_rejected(client, add_record):
    """note 超 200 字 → 400 中文；正好 200 字放行"""
    rid = add_record()
    r = client.post(f"/api/records/{rid}/feedback",
                    json={"is_correct": True, "note": "好" * 201})
    assert r.status_code == 400 and "200" in r.get_json()["error"]
    r2 = client.post(f"/api/records/{rid}/feedback",
                     json={"is_correct": True, "note": "好" * 200})
    assert r2.status_code == 200


def test_l2_duplicate_submit_updates_not_inserts(client, add_record):
    """防重复：同 result_id 提交两次 → 库里始终 1 行，第二次 updated=true，
    答案以最后一次为准（改答案）"""
    rid = add_record(label="Happy")
    r1 = client.post(f"/api/records/{rid}/feedback?uid=u_1", json={"is_correct": True})
    assert r1.get_json() == {"ok": True, "updated": False}
    r2 = client.post(f"/api/records/{rid}/feedback?uid=u_1",
                     json={"is_correct": False, "correct_label": "Defence"})
    assert r2.status_code == 200 and r2.get_json() == {"ok": True, "updated": True}
    rows = _fb_rows(rid)
    assert len(rows) == 1
    assert rows[0]["is_correct"] == 0 and rows[0]["correct_label"] == "Defence"
    assert rows[0]["updated_at"] >= rows[0]["created_at"]
    # 再改回"准确"：correct_label 应清空
    r3 = client.post(f"/api/records/{rid}/feedback?uid=u_1", json={"is_correct": True})
    assert r3.get_json()["updated"] is True
    row = _fb_rows(rid)[0]
    assert row["is_correct"] == 1 and row["correct_label"] == ""


def test_l2_record_not_found_404(client, add_record):
    """result_id 在 records 里不存在 → 404 中文（有/无反馈两态都试）"""
    add_record()
    r = client.post("/api/records/result-nope/feedback", json={"is_correct": True})
    assert r.status_code == 404 and "没有找到" in r.get_json()["error"]
    r2 = client.post("/api/records/result-nope/feedback",
                     json={"is_correct": False, "correct_label": "Angry"})
    assert r2.status_code == 404


def test_l2_illegal_result_id_400(client):
    """result_id 含非法字符（含 ".."）→ 400，正则同 discard_pending()"""
    for bad in ("res..ult", "abc$def", "a%20b"):
        r = client.post(f"/api/records/{bad}/feedback", json={"is_correct": True})
        assert r.status_code == 400, bad
        assert "不合法" in r.get_json()["error"]


def test_l2_empty_and_malformed_body_400(client, add_record):
    """空 body / 畸形 JSON → 400 中文，不 500（_json_body 范式）"""
    rid = add_record()
    r1 = client.post(f"/api/records/{rid}/feedback", data="")
    assert r1.status_code == 400 and "error" in r1.get_json()
    r2 = client.post(f"/api/records/{rid}/feedback", data="{not-json",
                     content_type="application/json")
    assert r2.status_code == 400 and "error" in r2.get_json()
    r3 = client.post(f"/api/records/{rid}/feedback", json=[1, 2])   # 数组 body 回落空 dict
    assert r3.status_code == 400


def test_l2_uid_truncated_to_64(client, add_record):
    """超长 uid 截断 64（与 records.device_uid 同口径）"""
    rid = add_record()
    r = client.post(f"/api/records/{rid}/feedback?uid={'x' * 100}", json={"is_correct": True})
    assert r.status_code == 200
    assert len(_fb_rows(rid)[0]["device_uid"]) == 64


# ─────────────── L4 统计页"用户反馈质量"区块 ───────────────

def _stats_html(client, cloud):
    r = client.get(f"/stats?key={cloud.STATS_KEY}")
    assert r.status_code == 200
    return r.get_data(as_text=True)


def test_l4_stats_quality_numbers(client, cloud, add_record):
    """种子数据后区块出正确数字：反馈率/对错占比/混淆 TOP（中文情绪名）"""
    add_record(result_id="result-h1", label="Happy")
    add_record(result_id="result-h2", label="Happy")
    add_record(result_id="result-a1", label="Angry")
    _insert_fb("result-h1", is_correct=True)
    _insert_fb("result-h2", is_correct=False, correct_label="Angry", note="x")
    _insert_fb("result-a1", is_correct=False, correct_label="Happy", note="y")
    html = _stats_html(client, cloud)
    assert "用户反馈质量" in html
    assert "100.0%" in html                              # 3 反馈 / 3 记录
    assert "33.3%" in html and "66.7%" in html           # 对/错占比
    assert "预测😊 开心 → 实为😡 生气：1 次" in html       # 混淆（中文情绪名）
    assert "预测😡 生气 → 实为😊 开心：1 次" in html


def test_l4_stats_empty_feedback_no_error(client, cloud):
    """无任何数据：区块显示"暂无反馈"，反馈率按口径显示 "—"，不报错"""
    html = _stats_html(client, cloud)
    assert "暂无反馈" in html
    assert '<div class="num">—</div><div class="lbl">一键反馈率</div>' in html


def test_l4_stats_orphan_feedback_excluded_from_confusion(client, cloud, add_record):
    """识别记录被删后的孤儿反馈：混淆 TOP 不计入（INNER JOIN），对/错分布仍按表行数"""
    add_record(result_id="result-orph", label="Happy")
    add_record(result_id="result-ok", label="Resting")
    _insert_fb("result-orph", is_correct=False, correct_label="Angry")
    _insert_fb("result-ok", is_correct=True)
    assert client.delete("/api/records/result-orph").status_code == 200
    html = _stats_html(client, cloud)
    # 孤儿反馈（Happy→Angry）不再出现在混淆表
    assert "预测😊 开心 → 实为😡 生气" not in html
    assert "暂无反馈" in html
    # 对/错分布仍是 1 对 1 错
    assert "50.0%" in html


def test_l4_stats_wrong_key_still_403(client):
    """错 key 403 沿用现状"""
    assert client.get("/stats").status_code == 403
    assert client.get("/stats?key=wrong").status_code == 403

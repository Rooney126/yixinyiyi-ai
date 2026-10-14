"""L4 导出（CSV BOM/中文表头 / JSON 结构 / 全量备份含 feedback 表）"""
import time
from urllib.parse import unquote


def _mk_record(now, offset_ms):
    return now + offset_ms


def test_export_csv_bom_and_chinese_header(client, make_cat, add_record):
    cat = make_cat(name="导出猫")
    now = int(time.time() * 1000)
    add_record(cat_id=cat["id"], label="Angry", confidence=0.9,
               created_at=now)
    add_record(cat_id=cat["id"], label="Happy", confidence=0.8,
               created_at=now - 60_000)

    r = client.get(f"/api/cats/{cat['id']}/export?format=csv")
    assert r.status_code == 200
    assert "text/csv" in r.headers["Content-Type"]
    assert "attachment" in r.headers["Content-Disposition"]
    assert "导出猫" in unquote(r.headers["Content-Disposition"])  # 中文名附件

    text = r.get_data().decode("utf-8")
    assert text[0] == "\ufeff"                   # UTF-8 BOM：Excel 直开不乱码
    lines = text.lstrip("\ufeff").splitlines()
    header = lines[0].split(",")
    assert header[:3] == ["时间", "情绪(中文)", "情绪(英文)"]
    assert "猫话翻译" in header and "养护建议" in header
    assert len(lines) == 3                       # 表头 + 2 条记录
    assert "开心" in lines[1] and "Happy" in lines[1]      # 按时间旧 → 新
    assert "生气" in lines[2] and "Angry" in lines[2]


def test_export_csv_empty_cat_header_only(client, make_cat):
    cat = make_cat(name="空猫")
    r = client.get(f"/api/cats/{cat['id']}/export?format=csv")
    assert r.status_code == 200
    text = r.get_data().decode("utf-8").lstrip("\ufeff")
    assert len(text.splitlines()) == 1           # 只有表头一行


def test_export_json_structure(client, make_cat, add_record):
    cat = make_cat(name="JSON猫")
    add_record(cat_id=cat["id"], label="Resting", confidence=0.77)
    r = client.get(f"/api/cats/{cat['id']}/export?format=json")
    assert r.status_code == 200
    assert "application/json" in r.headers["Content-Type"]
    j = r.get_json()
    assert j["cat"]["id"] == cat["id"] and j["cat"]["name"] == "JSON猫"
    assert len(j["records"]) == 1
    rec = j["records"][0]
    assert rec["label"] == "Resting"
    assert set(rec) >= {"result_id", "cat_id", "label", "state",
                        "translation", "confidence", "created_at"}
    assert "exported_at" in j
    assert j["app"] == "译芯怡意·猫档案"


def test_export_nonexistent_cat_404(client):
    assert client.get("/api/cats/cat-nope/export").status_code == 404


def test_backup_requires_key(client, make_cat, add_record):
    """安全加固：全量备份必须带 STATS_KEY，无 key / 错 key 一律 403"""
    make_cat(name="备份猫")
    add_record(cat_id=None)
    r = client.get("/api/export/backup")
    assert r.status_code == 403
    assert "口令" in r.get_data(as_text=True)     # 提示带口令访问
    r = client.get("/api/export/backup?key=wrong-key")
    assert r.status_code == 403


def test_backup_with_key_contains_all_data(client, make_cat, add_record,
                                            add_feedback, cloud):
    """带正确 STATS_KEY：猫 + 记录 + 旧 feedback 表，一个 JSON 带走"""
    cat = make_cat(name="备份猫")
    add_record(cat_id=cat["id"], label="Happy")
    add_record(cat_id=cat["id"], label="Angry")
    add_feedback(result_id="result-fb1", label="Happy", feedback="unsure")
    add_feedback(result_id="result-fb2", label="Angry", feedback="wrong")

    r = client.get(f"/api/export/backup?key={cloud.STATS_KEY}")
    assert r.status_code == 200
    assert "application/json" in r.headers["Content-Type"]
    assert "全量备份" in unquote(r.headers["Content-Disposition"])

    j = r.get_json()
    assert j["app"].startswith("译芯怡意")
    assert [c["name"] for c in j["cats"]] == ["备份猫"]
    assert len(j["records"]) == 2
    assert {x["label"] for x in j["records"]} == {"Happy", "Angry"}
    # 旧库 feedback 表数据随备份带走
    fb = {f["result_id"]: f for f in j["feedback"]}
    assert fb["result-fb1"]["feedback"] == "unsure"
    assert fb["result-fb2"]["label"] == "Angry"
    assert "exported_at" in j and "format_version" in j


def test_rate_limit_on_cat_writes(client):
    """裸写接口限流：60 秒内第 11 次创建猫被 429 挡下（与 predict 同机制）"""
    codes = [client.post("/api/cats", json={"name": f"限流猫{i}"}).status_code
             for i in range(11)]
    assert codes[:10] == [201] * 10
    assert codes[10] == 429
    assert "请求太频繁" in client.post(
        "/api/cats", json={"name": "第十二只"}).get_json()["error"]
    # 限流不影响只读接口
    assert client.get("/api/cats").status_code == 200

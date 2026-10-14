"""L1 猫档案：CRUD + 字段校验 + 归档/恢复"""
import time


def test_create_cat_full_fields(client):
    r = client.post("/api/cats", json={
        "name": "橘座", "color": "#F5D3B5", "emoji": "😼",
        "birthday": "2024-05-01", "note": "胖但灵活"})
    assert r.status_code == 201
    cat = r.get_json()["cat"]
    assert cat["name"] == "橘座"
    assert cat["color"] == "#F5D3B5"
    assert cat["emoji"] == "😼"
    assert cat["birthday"] == "2024-05-01"
    assert cat["note"] == "胖但灵活"
    assert cat["archived"] is False
    assert cat["id"].startswith("cat-")
    assert isinstance(cat["created_at"], int)


def test_create_cat_minimal_defaults(client):
    r = client.post("/api/cats", json={"name": "小白"})
    assert r.status_code == 201
    cat = r.get_json()["cat"]
    assert cat["color"] == "#B5D8CC"      # 默认色与站点配色一致
    assert cat["emoji"] == "🐱"
    assert cat["birthday"] == "" and cat["note"] == ""
    assert cat["archived"] is False


def test_create_cat_blank_name_rejected(client):
    r = client.post("/api/cats", json={"name": "   "})
    assert r.status_code == 400
    assert "猫名字" in r.get_json()["error"]


def test_create_cat_missing_name_rejected(client):
    assert client.post("/api/cats", json={}).status_code == 400


def test_create_cat_name_too_long_rejected(client):
    r = client.post("/api/cats", json={"name": "喵" * 21})
    assert r.status_code == 400
    assert "猫名字" in r.get_json()["error"]


def test_create_cat_name_exactly_20_ok(client):
    r = client.post("/api/cats", json={"name": "喵" * 20})
    assert r.status_code == 201           # 20 字边界：合法


def test_create_cat_bad_color_rejected(client):
    for bad in ("red", "#12G45Z", "#12345", "F5D3B5", "#B5D8CC11"):
        r = client.post("/api/cats", json={"name": "小黑", "color": bad})
        assert r.status_code == 400, f"颜色 {bad} 应被拒绝"
        assert "颜色" in r.get_json()["error"]


def test_create_cat_empty_color_gets_default(client):
    r = client.post("/api/cats", json={"name": "小黑", "color": ""})
    assert r.status_code == 201
    assert r.get_json()["cat"]["color"] == "#B5D8CC"


def test_create_cat_emoji_too_long_rejected(client):
    r = client.post("/api/cats", json={"name": "小灰", "emoji": "🐱" * 9})
    assert r.status_code == 400
    assert "表情" in r.get_json()["error"]


def test_create_cat_long_note_truncated(client):
    r = client.post("/api/cats", json={"name": "小灰", "note": "记" * 150})
    assert r.status_code == 201
    assert len(r.get_json()["cat"]["note"]) == 100    # 备注截断到 100 字


def test_patch_cat_updates_fields(client, make_cat):
    cat = make_cat(name="旧名")
    r = client.patch(f"/api/cats/{cat['id']}",
                     json={"name": "新名", "note": "改过的备注"})
    assert r.status_code == 200
    j = r.get_json()["cat"]
    assert j["name"] == "新名"
    assert j["note"] == "改过的备注"
    assert j["updated_at"] >= cat["updated_at"]
    # 没改的字段保持原样
    assert j["color"] == cat["color"]


def test_patch_cat_empty_body_rejected(client, make_cat):
    cat = make_cat()
    r = client.patch(f"/api/cats/{cat['id']}", json={})
    assert r.status_code == 400
    assert "没有要修改" in r.get_json()["error"]


def test_patch_cat_bad_color_rejected(client, make_cat):
    cat = make_cat()
    r = client.patch(f"/api/cats/{cat['id']}", json={"color": "orange"})
    assert r.status_code == 400


def test_patch_nonexistent_cat_404(client):
    assert client.patch("/api/cats/cat-nope",
                        json={"name": "x"}).status_code == 404


def test_archive_and_restore_roundtrip(client, make_cat, add_record):
    cat = make_cat(name="归档猫")
    add_record(cat_id=cat["id"])
    # 归档（软删除）
    r = client.delete(f"/api/cats/{cat['id']}")
    assert r.status_code == 200 and r.get_json()["archived"] is True
    # 默认列表隐藏
    ids = [c["id"] for c in client.get("/api/cats").get_json()["cats"]]
    assert cat["id"] not in ids
    # 全量列表可见，archived 标记，记录数仍计入（数据不丢）
    full = {c["id"]: c
            for c in client.get("/api/cats?include_archived=1").get_json()["cats"]}
    assert full[cat["id"]]["archived"] is True
    assert full[cat["id"]]["record_count"] == 1
    # 恢复
    r = client.post(f"/api/cats/{cat['id']}/restore")
    assert r.status_code == 200
    ids = [c["id"] for c in client.get("/api/cats").get_json()["cats"]]
    assert cat["id"] in ids


def test_archive_nonexistent_404(client):
    assert client.delete("/api/cats/cat-nope").status_code == 404


def test_restore_nonexistent_404(client):
    assert client.post("/api/cats/cat-nope/restore").status_code == 404


def test_list_record_count_and_last_time(client, make_cat, add_record):
    cat = make_cat(name="计数猫")
    now = int(time.time() * 1000)
    add_record(cat_id=cat["id"], created_at=now - 60_000)
    add_record(cat_id=cat["id"], created_at=now)
    cats = {c["id"]: c for c in client.get("/api/cats").get_json()["cats"]}
    c = cats[cat["id"]]
    assert c["record_count"] == 2
    assert c["last_record_at"] == now

"""异常路径测试：畸形上传 / 超长输入 / 并发写 / 畸形 JSON body / 参数边界"""
import io
import threading

import numpy as np


# ─── 畸形上传（load_model 用假会话顶掉，不依赖真模型）───

def _fake_model(cloud, monkeypatch):
    monkeypatch.setattr(cloud, "load_model",
                        lambda: (type("S", (), {"run": staticmethod(
                            lambda *_: np.zeros((1, 10), np.float32))})(),) * 2)


def test_predict_empty_file_400(client, cloud, monkeypatch):
    _fake_model(cloud, monkeypatch)
    r = client.post("/api/translate/predict",
                    data={"file": (io.BytesIO(b""), "empty.wav")})
    assert r.status_code == 400
    assert r.get_json()["error"]


def test_predict_text_as_audio_400(client, cloud, monkeypatch):
    """纯文本冒充音频：解码失败或无猫叫段，总之必须是 400 而非 500"""
    _fake_model(cloud, monkeypatch)
    r = client.post("/api/translate/predict",
                    data={"file": (io.BytesIO("这不是音频文件".encode("utf-8")),
                                   "fake.wav")})
    assert r.status_code == 400
    assert r.get_json()["error"]


def test_predict_truncated_wav_400(client, cloud, monkeypatch):
    """只有 RIFF 头的截断 wav"""
    _fake_model(cloud, monkeypatch)
    r = client.post("/api/translate/predict",
                    data={"file": (io.BytesIO(b"RIFF\x24\x00\x00\x00WAVE"),
                                   "cut.wav")})
    assert r.status_code == 400


# ─── 超长输入与长度边界 ───

def test_cat_name_length_boundaries(client):
    assert client.post("/api/cats", json={"name": "喵" * 20}).status_code == 201
    assert client.post("/api/cats", json={"name": "喵" * 21}).status_code == 400


def test_cat_emoji_and_note_limits(client, make_cat):
    assert client.post("/api/cats", json={"name": "猫", "emoji": "🐱🐱🐱"}).status_code == 201
    assert client.post("/api/cats", json={"name": "猫", "emoji": "e" * 9}).status_code == 400
    cat = make_cat(name="备注猫", note="备" * 500)
    assert cat["note"] == "备" * 100          # 超长备注截断到 100
    cat2 = make_cat(name="生日猫", birthday="2020-01-01" + "x" * 30)
    assert len(cat2["birthday"]) == 20        # 生日截断到 20


def test_device_uid_truncated_to_64(client, cloud, add_record):
    """超长 uid 落库前截断 64：用前 64 字能查到，完整串按截断口径也能查到"""
    long_uid = "u" * 100
    rid = add_record(device_uid=long_uid)
    j = client.get(f"/api/records/unassigned?uid={long_uid}").get_json()
    assert [x["result_id"] for x in j["records"]] == [rid]
    assert len(j["records"][0]["device_uid"]) == 64


def test_very_long_cat_id_404(client):
    assert client.get(f"/api/cats/{'x' * 500}/records").status_code == 404


# ─── 并发写 ───

def test_concurrent_cat_creation(client, cloud):
    """8 线程并发建猫：全部成功、无异常、列表终态一致（8 < 限流阈值 10）"""
    results = []
    barrier = threading.Barrier(8)

    def worker(i):
        c = cloud.app.test_client()
        barrier.wait()
        r = c.post("/api/cats", json={"name": f"并发猫{i}"})
        results.append((i, r.status_code))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert len(results) == 8
    assert {code for _, code in results} == {201}
    names = sorted(c["name"] for c in client.get("/api/cats").get_json()["cats"])
    assert names == sorted(f"并发猫{i}" for i in range(8))


def test_concurrent_assign_same_record(client, cloud, make_cat, add_record):
    """8 线程并发把同一条记录认领给两只猫：无异常，终态唯一且属于其一"""
    a, b = make_cat(name="并发A"), make_cat(name="并发B")
    rid = add_record(cat_id=a["id"])
    codes = []
    barrier = threading.Barrier(8)

    def worker(i):
        c = cloud.app.test_client()
        target = a["id"] if i % 2 == 0 else b["id"]
        barrier.wait()
        r = c.post(f"/api/records/{rid}/assign", json={"cat_id": target})
        codes.append(r.status_code)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert codes and all(code == 200 for code in codes)
    in_a = [x["result_id"] for x in
            client.get(f"/api/cats/{a['id']}/records").get_json()["records"]]
    in_b = [x["result_id"] for x in
            client.get(f"/api/cats/{b['id']}/records").get_json()["records"]]
    # 终态：恰好归属一只猫，无重复无丢失
    if rid in in_a:
        assert rid not in in_b
    else:
        assert rid in in_b


# ─── 畸形 JSON body ───

def test_cats_create_non_dict_bodies_400(client):
    """数组 / 字符串 / null body 都回落空 dict → 走业务校验 400，而不是 500"""
    for body in ([1, 2, 3], "just a string", None, 42):
        r = client.post("/api/cats", json=body)
        assert r.status_code == 400, body
        assert "猫名字" in r.get_json()["error"]


def test_cats_create_numeric_fields_coerced(client):
    """数字字段 str() 强转：name=123 → "123" 建档成功，不炸 500"""
    r = client.post("/api/cats", json={"name": 123, "birthday": 2020, "note": 999})
    assert r.status_code == 201
    assert r.get_json()["cat"]["name"] == "123"
    assert r.get_json()["cat"]["birthday"] == "2020"
    assert r.get_json()["cat"]["note"] == "999"


def test_cats_update_array_body_400(client, make_cat):
    cat = make_cat(name="被改猫")
    r = client.patch(f"/api/cats/{cat['id']}", json=[1, 2, 3])
    assert r.status_code == 400
    assert "没有要修改的内容" in r.get_json()["error"]


def test_assign_array_body_unassigns(client, make_cat, add_record):
    """数组 body 回落 {} → cat_id=None → 等同取消归属（与缺字段行为一致）"""
    cat = make_cat(name="归属猫")
    rid = add_record(cat_id=cat["id"])
    r = client.post(f"/api/records/{rid}/assign", json=["not", "a", "dict"])
    assert r.status_code == 200 and r.get_json()["cat_id"] is None


def test_feedback_array_body_ok(client, cloud):
    log = cloud.BASE / "data" / "feedback.jsonl"
    try:
        r = client.post("/api/translate/feedback", json=[1, 2, 3])
        assert r.status_code == 200 and r.get_json()["ok"] is True
        last = log.read_text(encoding="utf-8").strip().splitlines()[-1]
        assert '"feedback": "unsure"' in last     # 回落默认值落盘
    finally:
        log.unlink(missing_ok=True)


def test_event_numeric_and_array_bodies_ok(client, cloud):
    log = cloud.BASE / "data" / "events.jsonl"
    try:
        assert client.post("/api/event", json=[1]).status_code == 200
        r = client.post("/api/event", json={"event": 123, "path": 456, "ref": 789})
        assert r.status_code == 200              # 数字字段强转，不再 500
        last = log.read_text(encoding="utf-8").strip().splitlines()[-1]
        assert '"event": "123"' in last
    finally:
        log.unlink(missing_ok=True)


# ─── 查询参数边界 ───

def test_timeline_limit_clamps(client, make_cat, add_record):
    cat = make_cat(name="分页猫")
    for i in range(3):
        add_record(cat_id=cat["id"])
    j0 = client.get(f"/api/cats/{cat['id']}/records?limit=0").get_json()
    assert j0["limit"] == 1 and len(j0["records"]) == 1   # 下限钳到 1
    j9 = client.get(f"/api/cats/{cat['id']}/records?limit=99999").get_json()
    assert j9["limit"] == 200                              # 上限钳到 200
    jn = client.get(f"/api/cats/{cat['id']}/records?offset=-50").get_json()
    assert jn["offset"] == 0                               # 负 offset 钳到 0


def test_stats_days_clamps(client, make_cat):
    cat = make_cat(name="统计猫")
    assert client.get(f"/api/cats/{cat['id']}/stats?days=99999") \
                 .get_json()["days"] == 365
    assert client.get(f"/api/cats/{cat['id']}/stats?days=abc") \
                 .get_json()["days"] == 30                  # 非数字回落默认
    assert client.get(f"/api/cats/{cat['id']}/stats?days=0") \
                 .get_json()["days"] == 1                   # 下限钳到 1


def test_unassigned_limit_clamps(client, add_record):
    add_record(device_uid="u_clamp")
    j = client.get("/api/records/unassigned?uid=u_clamp&limit=0").get_json()
    assert j["records"] is not None                          # 不炸，正常返回列表

"""L1 记录归属：认领 / 换归属 / 取消归属 + 时间线分页与 days 过滤"""
import time


def test_assign_unassigned_record(client, make_cat, add_record):
    cat = make_cat(name="橘座")
    rid = add_record(device_uid="u_dev1")          # 识别时未选猫 → 无归属
    # 未归属列表按设备 uid 可见
    un = client.get("/api/records/unassigned?uid=u_dev1").get_json()["records"]
    assert [x["result_id"] for x in un] == [rid]
    # 认领到橘座
    r = client.post(f"/api/records/{rid}/assign", json={"cat_id": cat["id"]})
    assert r.status_code == 200
    assert r.get_json()["cat_id"] == cat["id"]
    # 出现在猫的时间线，从未归属列表消失
    tl = client.get(f"/api/cats/{cat['id']}/records").get_json()
    assert [x["result_id"] for x in tl["records"]] == [rid]
    assert client.get("/api/records/unassigned?uid=u_dev1") \
                 .get_json()["records"] == []


def test_assign_to_nonexistent_cat_404(client, add_record):
    rid = add_record()
    r = client.post(f"/api/records/{rid}/assign", json={"cat_id": "cat-nope"})
    assert r.status_code == 404
    # 记录保持无归属
    assert client.get("/api/records/unassigned?uid=u_test") \
                 .get_json()["records"]


def test_assign_rejects_archived_cat(client, make_cat, add_record):
    """认领/换归属拒绝已归档（软删除）的猫，记录保持原状"""
    archived = make_cat(name="归档猫")
    rid = add_record(device_uid="u_arch")
    client.delete(f"/api/cats/{archived['id']}")      # 归档
    r = client.post(f"/api/records/{rid}/assign",
                    json={"cat_id": archived["id"]})
    assert r.status_code == 400
    assert "归档" in r.get_json()["error"]            # 明确中文提示
    # 未归属记录没有被动过，仍留在认领池
    un = client.get("/api/records/unassigned?uid=u_arch").get_json()["records"]
    assert [x["result_id"] for x in un] == [rid]
    # 换归属到已归档猫同样被拒，原归属不变
    live = make_cat(name="活猫")
    rid2 = add_record(cat_id=live["id"])
    r2 = client.post(f"/api/records/{rid2}/assign",
                     json={"cat_id": archived["id"]})
    assert r2.status_code == 400
    tl = client.get(f"/api/cats/{live['id']}/records").get_json()["records"]
    assert [x["result_id"] for x in tl] == [rid2]
    # 取消归属（cat_id=null）不经过猫校验，不受影响
    r3 = client.post(f"/api/records/{rid2}/assign", json={"cat_id": None})
    assert r3.status_code == 200 and r3.get_json()["cat_id"] is None


def test_assign_nonexistent_record_404(client, make_cat):
    cat = make_cat()
    r = client.post("/api/records/result-nope/assign",
                    json={"cat_id": cat["id"]})
    assert r.status_code == 404


def test_reassign_moves_between_cats(client, make_cat, add_record):
    a, b = make_cat(name="猫A"), make_cat(name="猫B")
    rid = add_record(cat_id=a["id"])
    r = client.post(f"/api/records/{rid}/assign", json={"cat_id": b["id"]})
    assert r.status_code == 200 and r.get_json()["ok"] is True
    assert client.get(f"/api/cats/{a['id']}/records") \
                 .get_json()["records"] == []
    assert [x["result_id"] for x in client.get(f"/api/cats/{b['id']}/records")
            .get_json()["records"]] == [rid]


def test_unassign_returns_to_pool(client, make_cat, add_record):
    cat = make_cat()
    rid = add_record(cat_id=cat["id"], device_uid="u_dev9")
    r = client.post(f"/api/records/{rid}/assign", json={"cat_id": None})
    assert r.status_code == 200 and r.get_json()["cat_id"] is None
    un = client.get("/api/records/unassigned?uid=u_dev9") \
               .get_json()["records"]
    assert [x["result_id"] for x in un] == [rid]


def test_unassigned_empty_without_uid(client, add_record):
    add_record(device_uid="u_dev1")
    r = client.get("/api/records/unassigned")
    assert r.get_json()["records"] == []          # 不带 uid 不给看任何记录


def test_unassigned_scoped_by_uid(client, add_record):
    add_record(device_uid="u_a", label="Angry")
    add_record(device_uid="u_b", label="Happy")
    ids = [x["result_id"] for x in
           client.get("/api/records/unassigned?uid=u_a").get_json()["records"]]
    assert len(ids) == 1                          # 只看到自己设备的


def test_delete_record(client, make_cat, add_record):
    cat = make_cat()
    rid = add_record(cat_id=cat["id"])
    assert client.delete(f"/api/records/{rid}").status_code == 200
    assert client.get(f"/api/cats/{cat['id']}/records") \
                 .get_json()["records"] == []
    assert client.delete(f"/api/records/{rid}").status_code == 404


def test_timeline_pagination_and_order(client, make_cat, add_record):
    cat = make_cat()
    now = int(time.time() * 1000)
    for i in range(7):                            # i=0 最新
        add_record(cat_id=cat["id"], created_at=now - i * 60_000)

    j = client.get(f"/api/cats/{cat['id']}/records?limit=3&offset=0") \
              .get_json()
    assert j["total"] == 7 and j["limit"] == 3 and j["offset"] == 0
    assert len(j["records"]) == 3 and j["has_more"] is True
    ts = [x["created_at"] for x in j["records"]]
    assert ts == sorted(ts, reverse=True)         # 新 → 旧

    mid = client.get(f"/api/cats/{cat['id']}/records?limit=3&offset=3") \
                .get_json()
    assert len(mid["records"]) == 3 and mid["has_more"] is True
    # 页与页衔接：第 2 页最新的比第 1 页最旧的还早
    assert mid["records"][0]["created_at"] < ts[-1]

    last = client.get(f"/api/cats/{cat['id']}/records?limit=3&offset=6") \
                 .get_json()
    assert len(last["records"]) == 1 and last["has_more"] is False


def test_timeline_days_filter_boundaries(client, make_cat, add_record):
    cat = make_cat()
    now = int(time.time() * 1000)
    day = 86_400_000
    r_1d = add_record(cat_id=cat["id"], created_at=now - 1 * day + 30_000)
    r_3d = add_record(cat_id=cat["id"], created_at=now - 3 * day)
    r_in = add_record(cat_id=cat["id"], created_at=now - 7 * day + 30_000)
    r_out = add_record(cat_id=cat["id"], created_at=now - 7 * day - 30_000)
    r_8d = add_record(cat_id=cat["id"], created_at=now - 8 * day)

    j1 = client.get(f"/api/cats/{cat['id']}/records?days=1").get_json()
    assert j1["total"] == 1
    assert [x["result_id"] for x in j1["records"]] == [r_1d]

    # 7 天窗口：边缘内侧（7天整 - 30秒）按 >= 算进来，外侧算不进
    j7 = client.get(f"/api/cats/{cat['id']}/records?days=7").get_json()
    ids7 = {x["result_id"] for x in j7["records"]}
    assert j7["total"] == 3
    assert ids7 == {r_1d, r_3d, r_in}
    assert r_out not in ids7 and r_8d not in ids7

    j30 = client.get(f"/api/cats/{cat['id']}/records?days=30").get_json()
    assert j30["total"] == 5

    j0 = client.get(f"/api/cats/{cat['id']}/records?days=0").get_json()
    assert j0["total"] == 0                       # days=0：窗口为零，全排除


def test_timeline_bad_params_400(client, make_cat):
    cat = make_cat()
    assert client.get(f"/api/cats/{cat['id']}/records?limit=abc") \
                 .status_code == 400
    assert client.get(f"/api/cats/{cat['id']}/records?offset=xyz") \
                 .status_code == 400


def test_timeline_nonexistent_cat_404(client):
    assert client.get("/api/cats/cat-nope/records").status_code == 404

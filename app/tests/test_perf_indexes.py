"""【任务D】性能加固：索引存在性/幂等 + 热查询走索引 + 时间线分页正确性

口径映射（任务书列名 → 本库实际列名）：
  records(cat_id, ts) → idx_records_cat_time (cat_id, created_at)，M1 已有，这里核对
  records(uid)        → idx_records_device_time (device_uid, created_at)，本任务新增
  cats(deleted)       → idx_cats_archived (archived)，本库软删除列叫 archived
"""
import os
import sqlite3
import time


def _db():
    return sqlite3.connect(os.environ["YXY_DB_PATH"])


# ─── 索引存在性与幂等 ───

def test_required_indexes_exist(cloud):
    conn = _db()
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' "
        "AND tbl_name IN ('cats','records')")}
    conn.close()
    assert {"idx_records_cat_time", "idx_records_device_time",
            "idx_cats_archived"} <= names


def test_index_columns_match_spec(cloud):
    conn = _db()
    def cols(idx):
        return [r[2] for r in conn.execute(f"PRAGMA index_info({idx})")]
    assert cols("idx_records_cat_time") == ["cat_id", "created_at"]
    assert cols("idx_records_device_time") == ["device_uid", "created_at"]
    assert cols("idx_cats_archived") == ["archived"]
    conn.close()


def test_schema_idempotent_rerun(cloud):
    cloud.init_db()
    cloud.init_db()                               # 重复执行建表/建索引不报错
    conn = _db()
    n = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='index' "
        "AND name='idx_records_device_time'").fetchone()[0]
    conn.close()
    assert n == 1                                 # IF NOT EXISTS：不因重跑翻倍


# ─── 热查询走索引（EXPLAIN QUERY PLAN，锁死"不全表扫"）───

def test_timeline_query_uses_cat_time_index(client, make_cat, add_record):
    cat = make_cat()
    add_record(cat_id=cat["id"])
    conn = _db()
    plans = conn.execute(
        "EXPLAIN QUERY PLAN SELECT * FROM records WHERE cat_id=? AND created_at>=? "
        "ORDER BY created_at DESC LIMIT ? OFFSET ?", (cat["id"], 0, 50, 0)).fetchall()
    conn.close()
    detail = " ".join(str(r[-1]) for r in plans)
    assert "idx_records_cat_time" in detail       # 时间线命中 (cat_id, created_at)


def test_stats_query_uses_cat_time_index(client, make_cat, add_record):
    cat = make_cat()
    add_record(cat_id=cat["id"])
    conn = _db()
    plans = conn.execute(
        "EXPLAIN QUERY PLAN SELECT * FROM records WHERE cat_id=? AND created_at>=? "
        "ORDER BY created_at ASC", (cat["id"], 0)).fetchall()
    conn.close()
    detail = " ".join(str(r[-1]) for r in plans)
    assert "idx_records_cat_time" in detail       # 近30天统计不是全表扫


def test_unassigned_query_uses_an_index(client, make_cat, add_record):
    add_record(device_uid="u_dev1", cat_id=None)
    conn = _db()
    plans = conn.execute(
        "EXPLAIN QUERY PLAN SELECT * FROM records WHERE cat_id IS NULL AND device_uid=? "
        "ORDER BY created_at DESC LIMIT ?", ("u_dev1", 20)).fetchall()
    conn.close()
    detail = " ".join(str(r[-1]) for r in plans)
    # cat_time / device_time 任一命中均可（cat_id IS NULL 同样是可索引的等值约束），
    # 关键是不落地全表扫
    assert "USING INDEX" in detail


# ─── 时间线分页正确性（SQL LIMIT/OFFSET 真分页）───

def test_pagination_walk_covers_all_no_dup(client, make_cat, add_record):
    cat = make_cat()
    now = int(time.time() * 1000)
    expect = [add_record(cat_id=cat["id"], created_at=now - i * 60_000)
              for i in range(25)]                 # 25 条，页大小 10 → 3 页
    got, offset, has_more = [], 0, True
    while has_more:
        j = client.get(f"/api/cats/{cat['id']}/records?limit=10&offset={offset}") \
                  .get_json()
        got += [x["result_id"] for x in j["records"]]
        has_more = j["has_more"]
        offset += 10
        assert offset <= 100                      # 防止 has_more 永真死循环
    assert got == expect                          # 翻完全量：不重不漏，新→旧


def test_pagination_has_more_boundary(client, make_cat, add_record):
    cat = make_cat()
    now = int(time.time() * 1000)
    for i in range(4):
        add_record(cat_id=cat["id"], created_at=now - i * 60_000)
    exact = client.get(f"/api/cats/{cat['id']}/records?limit=4").get_json()
    assert len(exact["records"]) == 4 and exact["has_more"] is False
    add_record(cat_id=cat["id"], created_at=now + 60_000)   # 补 1 条最新的
    over = client.get(f"/api/cats/{cat['id']}/records?limit=4").get_json()
    assert len(over["records"]) == 4 and over["has_more"] is True
    assert over["total"] == 5


def test_pagination_limit_clamp_and_offset_past_end(client, make_cat, add_record):
    cat = make_cat()
    add_record(cat_id=cat["id"])
    j = client.get(f"/api/cats/{cat['id']}/records?limit=99999").get_json()
    assert j["limit"] == 200                      # limit 服务端钳到上限 200
    empty = client.get(f"/api/cats/{cat['id']}/records?limit=10&offset=500") \
                  .get_json()
    assert empty["records"] == [] and empty["has_more"] is False


def test_pagination_scoped_by_cat_and_days(client, make_cat, add_record):
    a, b = make_cat(name="猫A"), make_cat(name="猫B")
    now = int(time.time() * 1000)
    for i in range(6):                            # 猫A：1~6 小时前各 1 条
        add_record(cat_id=a["id"], created_at=now - (i + 1) * 3_600_000)
    add_record(cat_id=a["id"], created_at=now - 3 * 86_400_000)   # 3 天前，窗口外
    for i in range(3):                            # 猫B：干扰数据
        add_record(cat_id=b["id"], created_at=now - i * 60_000)
    j = client.get(f"/api/cats/{a['id']}/records?days=2&limit=2&offset=0").get_json()
    assert j["total"] == 6                        # total 只算本猫且在窗口内
    assert all(x["cat_id"] == a["id"] for x in j["records"])
    assert all(x["created_at"] >= now - 2 * 86_400_000 for x in j["records"])
    tail = client.get(f"/api/cats/{a['id']}/records?days=2&limit=2&offset=4") \
                 .get_json()
    assert len(tail["records"]) == 2 and tail["has_more"] is False

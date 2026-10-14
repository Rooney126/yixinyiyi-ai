"""译芯怡意 M1 自动化测试 — pytest fixtures

红线保证（任务书要求）：
- 测试库与录音目录一律指向 pytest 的 /tmp 临时目录（YXY_DB_PATH / YXY_RECORDINGS_DIR），
  绝不指向真实 data/
- 服务从 /tmp 里的 cloud_server.py 副本导入（模块内 BASE 指向副本目录），
  即使误触 calls/feedback/events 等落盘逻辑，也全部落在 /tmp，真实 data/ 零写入
- frontend 目录以软链只读引用（主页 / SPA 兜底测试要用）
- 真实 app.db 全程不打开
"""
import os
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest

# 真实项目目录（只读引用）：PROJECT_ROOT\app
REAL_APP_DIR = Path(__file__).resolve().parent.parent

# 镜像真实旧库的 feedback 表结构（全新测试库默认没有旧表，备份导出测试需要它）
FEEDBACK_SCHEMA = """
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


@pytest.fixture(scope="session")
def cloud(tmp_path_factory):
    """在 /tmp 搭测试根目录并导入 cloud_server，返回模块对象。

    路由、常量（STATS_KEY）、工具函数（save_record/_now_ms）、限流器
    都统一从这里取，避免测试文件各自 import 路径不一致。
    """
    root = tmp_path_factory.mktemp("yxy_m1")
    (root / "data").mkdir()
    # 代码副本：让 BASE（=cloud_server.py 所在目录）指向 /tmp，
    # 这样除 DB/录音外的落盘（calls/feedback/events.jsonl）也进不了真实 data/
    shutil.copy2(REAL_APP_DIR / "cloud_server.py", root / "cloud_server.py")
    os.symlink(REAL_APP_DIR / "frontend", root / "frontend")

    # DB 与录音目录必须指向 /tmp —— 双保险断言，指错地方宁可炸在测试里
    db_path = root / "data" / "test.db"
    rec_dir = root / "data" / "recordings"
    assert str(db_path).startswith("/tmp"), f"测试库必须在 /tmp，现在是 {db_path}"
    assert str(rec_dir).startswith("/tmp"), f"录音目录必须在 /tmp，现在是 {rec_dir}"
    os.environ["YXY_DB_PATH"] = str(db_path)
    os.environ["YXY_RECORDINGS_DIR"] = str(rec_dir)

    sys.path.insert(0, str(root))
    import cloud_server  # 导入时自动 init_db() 建表
    conn = sqlite3.connect(str(db_path))
    conn.execute(FEEDBACK_SCHEMA)
    conn.commit()
    conn.close()
    yield cloud_server


@pytest.fixture(scope="session")
def client(cloud):
    """Flask test_client（会话级复用一个应用实例）"""
    return cloud.app.test_client()


@pytest.fixture(autouse=True)
def _isolate(cloud):
    """每个测试前：清空限流器 + 清空 cats/records/feedback，测试间互不干扰。

    （限流器是全模块共享的 store，不清理的话前面的写测试会累计
     触发 429，影响后面的用例。）
    """
    cloud._rate_limit_store.clear()
    conn = sqlite3.connect(os.environ["YXY_DB_PATH"])
    for t in ("records", "cats", "feedback", "result_feedback", "stations"):
        conn.execute(f"DELETE FROM {t}")
    conn.commit()
    conn.close()
    yield
    cloud._rate_limit_store.clear()


@pytest.fixture()
def make_cat(client):
    """工厂：建一只猫并返回 cat dict（默认名"测试猫"，字段可覆盖）"""
    seq = {"n": 0}

    def _make(name=None, **fields):
        seq["n"] += 1
        payload = {"name": name or f"测试猫{seq['n']}"}
        payload.update(fields)
        r = client.post("/api/cats", json=payload)
        assert r.status_code == 201, r.get_data(as_text=True)
        return r.get_json()["cat"]

    return _make


@pytest.fixture()
def add_record(cloud):
    """工厂：走服务自身的 save_record 直接落一条识别记录（不经过模型）"""
    counter = {"n": 0}

    def _add(cat_id=None, label="Happy", confidence=0.9, created_at=None,
             device_uid="u_test", **kw):
        counter["n"] += 1
        rid = kw.pop("result_id", None) or f"result-t{counter['n']:06d}"
        cloud.save_record(
            rid, label,
            kw.pop("state", "开心"), kw.pop("translation", "嘿嘿～好开心！"),
            kw.pop("advice", "多陪它玩吧"), kw.pop("emoji", "😊"),
            confidence,
            n_segments=kw.pop("n_segments", 2),
            gate_score=kw.pop("gate_score", 0.92),
            fuzzy=kw.pop("fuzzy", False),
            source="web", device_uid=device_uid, cat_id=cat_id,
            created_at=created_at)
        return rid

    return _add


@pytest.fixture()
def add_feedback(cloud):
    """工厂：往旧 feedback 表插一行（备份导出测试用）"""
    def _add(result_id="result-fbx", label="Happy", feedback="unsure",
             confidence=0.9, timestamp=None):
        conn = sqlite3.connect(os.environ["YXY_DB_PATH"])
        cur = conn.execute(
            "INSERT INTO feedback (result_id, label, confidence, feedback, timestamp) "
            "VALUES (?,?,?,?,?)",
            (result_id, label, confidence, feedback,
             timestamp or cloud._now_ms()))
        conn.commit()
        rowid = cur.lastrowid
        conn.close()
        return rowid

    return _add

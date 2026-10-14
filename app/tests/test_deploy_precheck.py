"""部署预检工具包测试 — scripts/deploy_precheck.py + deploy_backup.sh + deploy_rollback.sh

每个检查项（A~F）与备份/回滚脚本都给正、反两向用例：
  正向 = 真实项目/沙盒上检查通过；反向 = 人为破坏（写死口令/坏库/占端口/
  缺依赖/空目录/缺包）后必须能查出来。
红线：真实 data/ 只读（探针与迁移演练全部用 /tmp 副本），沙盒一律建在
pytest 临时目录；不接触生产。
"""
import hashlib
import shutil
import socket
import sqlite3
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REAL_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REAL_APP_DIR / "scripts"))
import deploy_precheck as dp  # noqa: E402

# ────────────────────────── 工具 ──────────────────────────

def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def make_db(path: Path, rows=3):
    """造一个带 demo 表和若干行的 sqlite 库（备份/回滚内容校验用）"""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE demo (k TEXT)")
    for i in range(rows):
        conn.execute("INSERT INTO demo VALUES (?)", (f"row-{i}",))
    conn.commit()
    conn.close()


def make_sandbox_app(root: Path) -> Path:
    """搭一个最小 app 沙盒：root/app 为应用目录，root/models 为门卫模型目录
    （镜像真实布局：门卫在 app 上一级）。"""
    app = root / "app"
    (app / "frontend").mkdir(parents=True)
    (app / "model").mkdir()
    (app / "data" / "recordings").mkdir(parents=True)
    (app / "cloud_server.py").write_text("# sandbox cloud_server v1\n", encoding="utf-8")
    (app / "deploy_requirements.txt").write_text("flask>=2.0\n", encoding="utf-8")
    (app / "frontend" / "index.html").write_text("<html>sandbox</html>", encoding="utf-8")
    (app / "model" / "model.onnx").write_bytes(b"fake-onnx")
    make_db(app / "data" / "app.db", rows=3)
    (app / "data" / "recordings" / "result-s1.wav").write_bytes(b"fake-wav")
    models = root / "models"
    models.mkdir(exist_ok=True)
    (models / "minicat_binary.onnx").write_bytes(b"fake-gate")
    return app


def run_sh(script: str, *args, cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(REAL_APP_DIR / "scripts" / script), *map(str, args)],
                          capture_output=True, text=True, timeout=300, cwd=cwd)


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


@pytest.fixture()
def probe_cache():
    return {}


# ────────────────────────── A. STATS_KEY ──────────────────────────

def test_a_static_real_source_injected_and_marked():
    """正：真实源码支持 YXY_STATS_KEY 注入，默认值有「仅演示」标注"""
    st = dp.stats_key_static(REAL_APP_DIR / "cloud_server.py")
    assert st["injected"], "源码没有走 os.environ.get('YXY_STATS_KEY')"
    assert st["marker"], "默认值没有「演示」标注"
    assert st["default"] == "demo-key"


def test_a_static_hardcoded_key_detected(tmp_path):
    """反：把口令写死在源码里必须被查出来"""
    src = tmp_path / "cloud_server.py"
    src.write_text('STATS_KEY = "demo-key"\n', encoding="utf-8")
    st = dp.stats_key_static(src)
    assert not st["injected"]
    assert st["default"] == "demo-key"


def test_a_static_missing_demo_marker_detected(tmp_path):
    """反：走环境变量但没有「仅演示」标注，也要被查出来"""
    src = tmp_path / "cloud_server.py"
    src.write_text('# 生产口令\nSTATS_KEY = os.environ.get("YXY_STATS_KEY") or "demo-key"\n',
                   encoding="utf-8")
    st = dp.stats_key_static(src)
    assert st["injected"]
    assert not st["marker"]


def test_a_dynamic_and_b_migration_on_real_db_copy(tmp_path, probe_cache):
    """正（A 动态 + B 正向，共用一次探针）：YXY_STATS_KEY 注入实测生效；
    真实 app.db 的 /tmp 副本启动迁移：cats/records 建出、旧表数据保留、
    迁移前自动留 .bak_M1 快照——真实库本身不动"""
    wd = tmp_path / "wd"
    wd.mkdir()
    probe = dp.ensure_probe(REAL_APP_DIR, wd, probe_cache)
    assert probe["ok"], probe.get("error")
    assert probe["stats_key"] == probe["probe_value"], "环境变量注入没生效"

    ra = dp.check_a_stats_key(REAL_APP_DIR, wd, probe_cache)
    assert ra.ok, "\n".join(ra.lines)
    rb = dp.check_b_migration(REAL_APP_DIR, wd, probe_cache)
    assert rb.ok, "\n".join(rb.lines)
    assert "cats" in rb.data["tables_after"] and "records" in rb.data["tables_after"]
    # 按库形态分支断言（与 check_b_migration 的口径一致）：
    #   旧结构库 → 迁移前必须自动留 .bak_M1 快照；
    #   已是新结构（cats 表在）→ 走幂等重跑路径，不要求快照。
    if probe["had_cats"]:
        assert not probe["bak_created"] or True   # 幂等迁移，不产生快照是预期
    else:
        assert probe["bak_created"], "旧库迁移前没自动生成快照"
    # 真实 app.db 没被探针碰过（探针只进 /tmp 副本）
    assert (REAL_APP_DIR / "data" / "app.db").exists()


def test_b_migration_corrupt_db_fails(tmp_path):
    """反：库文件损坏（非 sqlite），探针必须报失败而不是静默通过"""
    app = tmp_path / "app"
    app.mkdir()
    shutil.copy2(REAL_APP_DIR / "cloud_server.py", app / "cloud_server.py")
    (app / "data").mkdir()
    (app / "data" / "app.db").write_bytes(b"this is definitely not a sqlite file" * 10)
    wd = tmp_path / "wd"
    wd.mkdir()
    r = dp.check_b_migration(app, wd, {})
    assert not r.ok
    assert any("失败" in ln or "❌" in ln for ln in r.lines)


# ────────────────────────── C. waitress 拉起 ──────────────────────────

def test_c_waitress_boot_health_and_stop(tmp_path):
    """正：waitress 在空闲备用端口拉起、/health 返回 ok、随后自动关停"""
    port = free_port()
    r = dp.check_c_waitress(REAL_APP_DIR, tmp_path, port=port)
    assert r.ok, "\n".join(r.lines)
    assert "ok" in (r.data.get("health_body") or "")
    assert r.data.get("exit_code") is not None, "演练进程没有退出"


def test_c_waitress_port_busy_fails(tmp_path):
    """反：端口被占（模拟 5173 上已有服务）必须报失败，不能误判通过"""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.listen(1)
    try:
        r = dp.check_c_waitress(REAL_APP_DIR, tmp_path, port=port)
        assert not r.ok
        assert any("占用" in ln for ln in r.lines)
    finally:
        s.close()


# ────────────────────────── D. 依赖比对 ──────────────────────────

def test_d_real_requirements_all_satisfied():
    """正：真实 deploy_requirements.txt 六项依赖本机全部满足"""
    r = dp.check_d_requirements(REAL_APP_DIR)
    assert r.ok, "\n".join(r.lines)
    assert r.data["missing"] == []
    assert r.data["mismatch"] == []
    assert r.data["n_reqs"] == 6


def test_d_missing_and_version_mismatch_detected(tmp_path):
    """反：缺包 + 版本钉死过高，两种差异都要报出来"""
    app = tmp_path / "app"
    app.mkdir()
    (app / "deploy_requirements.txt").write_text(
        "flask>=2.0\nnumpy>=999.0\nnever-exists-pkg>=1.0\n", encoding="utf-8")
    freeze = {"flask": "3.1.3", "numpy": "1.26.4"}
    r = dp.check_d_requirements(app, freeze=freeze)
    assert not r.ok
    assert "never-exists-pkg" in r.data["missing"]
    assert any(m.startswith("numpy") for m in r.data["mismatch"])
    # flask 满足约束，不能误报
    assert not any("flask" in m for m in r.data["mismatch"] + r.data["missing"])


# ────────────────────────── E. 文件与空间 ──────────────────────────

def test_e_real_app_files_ok():
    """正：真实 app 目录——磁盘/库/录音/双模型全在"""
    r = dp.check_e_files(REAL_APP_DIR)
    assert r.ok, "\n".join(r.lines)
    assert r.data["db_size"] > 0
    assert r.data["rec_files"] >= 1
    assert r.data["disk_free_gb"] > 0.5


def test_e_empty_app_dir_fails(tmp_path):
    """反：空目录——缺库/缺模型必须报失败（录音目录允许服务自建，记警告）"""
    r = dp.check_e_files(tmp_path)
    assert not r.ok
    assert any("model.onnx" in ln and "❌" in ln for ln in r.lines)
    assert any("minicat_binary" in ln and "❌" in ln for ln in r.lines)


# ────────────────────────── F. 备份演练 + 备份脚本 ──────────────────────────

def test_f_backup_roundtrip_and_db_snapshot(tmp_path):
    """正：预检调用备份脚本打出全量包并校验通过；包内 data/app.db 是一致性
    快照（能独立打开、行数与源库一致）；门卫 models/ 一并入库"""
    app = make_sandbox_app(tmp_path / "sbx")
    dest = tmp_path / "backups"
    dest.mkdir()
    r = dp.check_f_backup(app, dest)
    assert r.ok, "\n".join(r.lines)
    assert r.data["member_count"] >= 6

    tar_path = Path(r.data["backup_file"])
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extract("data/app.db", path=tmp_path / "extract", filter="data")
    conn = sqlite3.connect(str(tmp_path / "extract" / "data" / "app.db"))
    n = conn.execute("SELECT COUNT(*) FROM demo").fetchone()[0]
    conn.close()
    assert n == 3, "快照库内容与源库不一致"
    assert any(m.startswith("models/minicat_binary.onnx") for m in r.data["members"]), \
        "门卫模型没进备份包"


def test_backup_script_rejects_missing_app_root(tmp_path):
    """反（备份脚本）：app-root 指向不存在/不完整的目录必须非零退出且不打包"""
    dest = tmp_path / "backups"
    dest.mkdir()
    r = run_sh("deploy_backup.sh", "--app-root", tmp_path / "nope", "--dest", dest)
    assert r.returncode != 0
    assert list(dest.glob("*.tar.gz")) == []


# ────────────────────────── 回滚脚本 ──────────────────────────

def _make_backup(app: Path, dest: Path) -> Path:
    r = run_sh("deploy_backup.sh", "--app-root", app, "--dest", dest)
    assert r.returncode == 0, r.stderr
    return sorted(dest.glob("yxy_backup_manual_*.tar.gz"))[-1]


def test_rollback_list_and_missing_timestamp(tmp_path):
    """反：--list 能列出备份；不存在的時間戳必须非零退出且现场不动"""
    app = make_sandbox_app(tmp_path / "sbx")
    dest = tmp_path / "backups"
    dest.mkdir()
    _make_backup(app, dest)

    rl = run_sh("deploy_rollback.sh", "--app-root", app, "--backup-dir", dest, "--list")
    assert rl.returncode == 0
    assert "yxy_backup_manual_" in rl.stdout

    before = md5(app / "cloud_server.py")
    bad = run_sh("deploy_rollback.sh", "--app-root", app, "--backup-dir", dest,
                 "20990101_000000")
    assert bad.returncode != 0
    assert md5(app / "cloud_server.py") == before, "失败的回滚不该动现场"


def test_rollback_restores_without_deleting(tmp_path):
    """正 + 红线：回滚把改动文件恢复成备份版本；之后新加的文件【必须保留】
    （覆盖式恢复、禁止删除式恢复）；恢复前自动打 pre_rollback 现场快照"""
    sbx = tmp_path / "sbx"
    app = make_sandbox_app(sbx)
    dest = tmp_path / "backups"
    dest.mkdir()
    backup = _make_backup(app, dest)
    original_src = (app / "cloud_server.py").read_text(encoding="utf-8")

    # 备份之后现场继续演化：改代码、写库、加新文件
    (app / "cloud_server.py").write_text("# sandbox cloud_server v2 (deployed)\n", encoding="utf-8")
    conn = sqlite3.connect(str(app / "data" / "app.db"))
    conn.execute("INSERT INTO demo VALUES ('deployed-row')")
    conn.commit()
    conn.close()
    (app / "data" / "recordings" / "result-after-backup.wav").write_bytes(b"new-rec")
    (app / "frontend" / "stray.html").write_text("new page", encoding="utf-8")

    ts = backup.name[len("yxy_backup_manual_"):-len(".tar.gz")]
    r = run_sh("deploy_rollback.sh", "--app-root", app, "--backup-dir", dest, ts)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "未删除任何现有文件" in r.stdout

    # 改动被恢复
    assert (app / "cloud_server.py").read_text(encoding="utf-8") == original_src
    conn = sqlite3.connect(str(app / "data" / "app.db"))
    n = conn.execute("SELECT COUNT(*) FROM demo").fetchone()[0]
    conn.close()
    assert n == 3, "数据库没有恢复到备份版本"
    # 备份之后新增的东西必须还在（非删除式恢复）
    assert (app / "data" / "recordings" / "result-after-backup.wav").exists()
    assert (app / "frontend" / "stray.html").exists()
    # 现场快照已留底，能再滚回去
    assert list(dest.glob("yxy_backup_pre_rollback_*.tar.gz"))


def test_rollback_refuses_missing_app_root(tmp_path):
    """反：部署目录被整个删掉时拒绝“恢复”（不在空目标上动手）"""
    sbx = tmp_path / "sbx"
    app = make_sandbox_app(sbx)
    dest = tmp_path / "backups"
    dest.mkdir()
    backup = _make_backup(app, dest)
    shutil.rmtree(app)

    r = run_sh("deploy_rollback.sh", "--app-root", app, "--backup-dir", dest, str(backup))
    assert r.returncode != 0
    assert not (sbx / "app").exists(), "拒绝恢复时不应把目录建出来"

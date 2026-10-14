"""旧接口回归：主页 / 档案页 / SPA 路由兜底 / API 404 / pending 删除 / health"""
import time


def test_homepage(client):
    r = client.get("/")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "译芯怡意" in html
    assert "yxy_current_cat" in html             # M1 fetch 钩子仍在（X-Cat-Id 注入）


def test_cat_archive_page(client):
    r = client.get("/cats")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "猫档案" in html
    assert "全量备份" in html                     # 导出视图入口在


def test_spa_fallback_serves_index(client):
    for path in ("/translate", "/profile", "/no-such-page-xyz"):
        r = client.get(path)
        assert r.status_code == 200, path
        html = r.get_data(as_text=True)
        assert "yxy_current_cat" in html         # 兜底回落的是主站 index.html
        assert "猫档案" not in html[:200]        # 不是档案页


def test_static_file_still_served(client):
    r = client.get("/cat-archive.html")
    assert r.status_code == 200
    assert "猫档案" in r.get_data(as_text=True)


def test_unknown_api_route_json_404(client):
    r = client.get("/api/no/such/api")
    assert r.status_code == 404
    assert r.get_json() == {"error": "not found"}


def test_health(client):
    assert client.get("/health").get_json() == {"status": "ok"}


def test_pending_delete_removes_file_in_tmp_recordings(client, cloud):
    """pending 删除只动 YXY_RECORDINGS_DIR（/tmp 副本），不碰真实录音"""
    rid = f"result-pending-{int(time.time() * 1000)}"
    f = cloud.RECORDINGS_DIR / f"{rid}.wav"
    f.write_bytes(b"RIFFfake-wav-bytes")
    assert f.exists()
    r = client.delete(f"/api/translate/pending/{rid}")
    assert r.status_code == 200 and r.get_json()["ok"] is True
    assert not f.exists()                        # 删的是 /tmp 副本目录里的文件


def test_pending_delete_missing_file_still_ok(client):
    r = client.delete("/api/translate/pending/result-never-existed")
    assert r.status_code == 200
    assert r.get_json()["ok"] is True

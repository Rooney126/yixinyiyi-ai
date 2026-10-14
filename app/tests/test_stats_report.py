"""L2 统计数值 + L3 周报（404 / dl=1 下载头 / 上周环比）"""
import time
from datetime import datetime


def test_stats_numbers(client, make_cat, add_record):
    cat = make_cat(name="统计猫")
    # 时间戳一律相对"今天零点"构造，与测试运行的具体时刻无关：
    # 今天 4 条（Happy×2 + Angry + Resting）、昨天 1 条（Angry）、3 天前 1 条（Happy）
    day = 86_400_000
    midnight = int(datetime.now().replace(
        hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    today, yesterday = (datetime.fromtimestamp((midnight + o) / 1000).strftime("%Y-%m-%d")
                        for o in (0, -day))
    add_record(cat_id=cat["id"], label="Happy", confidence=0.9,
               created_at=midnight + 3600_000)
    add_record(cat_id=cat["id"], label="Happy", confidence=0.8,
               created_at=midnight + 7200_000)
    add_record(cat_id=cat["id"], label="Angry", confidence=0.85,
               created_at=midnight + 5400_000)
    add_record(cat_id=cat["id"], label="Resting", confidence=0.66,
               created_at=midnight + 5400_000)
    add_record(cat_id=cat["id"], label="Angry", confidence=0.95,
               created_at=midnight - day + 3600_000)
    add_record(cat_id=cat["id"], label="Happy", confidence=0.7,
               created_at=midnight - 3 * day + 3600_000)

    j = client.get(f"/api/cats/{cat['id']}/stats?days=30").get_json()
    assert j["days"] == 30 and j["total"] == 6
    assert j["negative_count"] == 2              # 异常口径：生气/防御/战斗/警告/疼痛
    assert abs(j["avg_confidence"] - 0.81) < 0.001

    # 分布按次数降序，中文情绪名 + emoji + 占比 + 平均置信度
    assert [d["label"] for d in j["distribution"]] == \
        ["Happy", "Angry", "Resting"]
    dist = {d["label"]: d for d in j["distribution"]}
    assert dist["Happy"]["state"] == "开心" and dist["Happy"]["emoji"] == "😊"
    assert dist["Angry"]["state"] == "生气" and dist["Angry"]["emoji"] == "😡"
    assert dist["Happy"]["count"] == 3 and dist["Happy"]["pct"] == 50.0
    assert abs(dist["Happy"]["avg_confidence"] - 0.8) < 0.001
    assert dist["Angry"]["count"] == 2 and dist["Angry"]["pct"] == 33.3
    assert dist["Resting"]["count"] == 1 and dist["Resting"]["pct"] == 16.7

    # 每日趋势：今天 4 条（1 条异常），昨天 1 条（异常），分类计数正确
    daily = {d["date"]: d for d in j["daily"]}
    assert daily[today]["total"] == 4
    assert daily[today]["negative"] == 1
    assert daily[today]["counts"]["Happy"] == 2
    assert daily[today]["counts"]["Angry"] == 1
    assert daily[today]["counts"]["Resting"] == 1
    assert daily[yesterday]["total"] == 1
    assert daily[yesterday]["negative"] == 1


def test_stats_daily_window_zero_padded(client, make_cat):
    cat = make_cat(name="空数据猫")
    j30 = client.get(f"/api/cats/{cat['id']}/stats?days=30").get_json()
    assert len(j30["daily"]) == 31               # 30 天窗口零填充 31 格（含今天）
    assert all(d["total"] == 0 for d in j30["daily"])
    j7 = client.get(f"/api/cats/{cat['id']}/stats?days=7").get_json()
    assert len(j7["daily"]) == 8
    assert j7["total"] == 0
    assert j7["distribution"] == [] and j7["avg_confidence"] == 0
    assert j7["negative_count"] == 0


def test_stats_bad_days_falls_back_to_30(client, make_cat):
    cat = make_cat()
    j = client.get(f"/api/cats/{cat['id']}/stats?days=abc").get_json()
    assert j["days"] == 30


def test_stats_nonexistent_cat_404(client):
    assert client.get("/api/cats/cat-nope/stats").status_code == 404


# ─── L3 周报 ───

def test_weekly_report_404_for_missing_cat(client):
    r = client.get("/api/cats/cat-nope/report/weekly")
    assert r.status_code == 404
    assert r.get_json()["error"] == "没有找到这只猫"


def test_weekly_report_html_layout(client, make_cat, add_record):
    cat = make_cat(name="咪咪酱")
    add_record(cat_id=cat["id"], label="Fighting", confidence=0.91,
               created_at=int(time.time() * 1000))
    r = client.get(f"/api/cats/{cat['id']}/report/weekly")
    assert r.status_code == 200
    assert "text/html" in r.headers["Content-Type"]
    html = r.get_data(as_text=True)
    assert "咪咪酱" in html and "情绪周报" in html
    assert "window.print" in html                # 打印 / 存 PDF 按钮
    assert "本周 vs 上周" in html and "异常明细" in html
    assert "<svg" in html                        # 每日趋势 SVG
    assert "战斗" in html                        # 异常明细里出现中文名


def test_weekly_report_dl_attachment_header(client, make_cat):
    cat = make_cat(name="下载猫")
    r = client.get(f"/api/cats/{cat['id']}/report/weekly?dl=1")
    assert r.status_code == 200
    cd = r.headers["Content-Disposition"]
    assert cd.startswith("attachment")
    assert "filename*=UTF-8''" in cd             # 中文文件名走 RFC 5987
    assert "weekly_report.html" in cd            # ASCII 兜底名
    from urllib.parse import unquote
    decoded = unquote(cd)
    assert "下载猫" in decoded and "情绪周报" in decoded
    assert time.strftime("%Y%m%d") in decoded


def test_weekly_report_offset_last_week(client, make_cat, add_record):
    cat = make_cat(name="环比猫")
    now = int(time.time() * 1000)
    add_record(cat_id=cat["id"], label="Happy", created_at=now)
    add_record(cat_id=cat["id"], label="Happy",
               created_at=now - 7 * 86_400_000)  # 上周同一天
    html_now = client.get(f"/api/cats/{cat['id']}/report/weekly") \
                     .get_data(as_text=True)
    assert "本周（进行中）" in html_now
    html_prev = client.get(f"/api/cats/{cat['id']}/report/weekly?offset=1") \
                      .get_data(as_text=True)
    assert "上周" in html_prev
    assert "平均置信度" in html_prev              # 环比表

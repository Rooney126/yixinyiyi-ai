"""CSV 公式注入防护：导出时以 = + - @ 开头的单元格前缀单引号（Excel 按文本展示）"""
import csv
import io


def _export_rows(client, cat_id):
    r = client.get(f"/api/cats/{cat_id}/export?format=csv")
    assert r.status_code == 200
    assert "text/csv" in r.headers["Content-Type"]
    text = r.get_data().decode("utf-8").lstrip("\ufeff")
    return list(csv.reader(io.StringIO(text)))


def test_csv_formula_cells_neutralized(client, make_cat, add_record):
    cat = make_cat(name="注入猫")
    add_record(cat_id=cat["id"], label="Happy",
               translation="=1+1", advice="@SUM(A1)")
    add_record(cat_id=cat["id"], label="Angry",
               translation="+HYPERLINK(\"http://x\",\"y\")", advice="-2+3")
    rows = _export_rows(client, cat["id"])
    assert rows[0][0] == "时间"                       # 表头不受影响
    translations = {r[4] for r in rows[1:]}          # 猫话翻译列
    advices = {r[5] for r in rows[1:]}               # 养护建议列
    assert "'=1+1" in translations
    assert "'+HYPERLINK" in next(t for t in translations if t.startswith("'+"))
    assert "'@SUM(A1)" in advices
    assert "'-2+3" in advices
    for cell in translations | advices:
        assert cell.startswith(("'=", "'+", "'-", "'@")), cell


def test_csv_normal_cells_untouched(client, make_cat, add_record):
    """正常文本与数值不加前缀，避免破坏 Excel 里的展示与再计算"""
    cat = make_cat(name="普通猫")
    add_record(cat_id=cat["id"], label="Happy", translation="嘿嘿～好开心！")
    rows = _export_rows(client, cat["id"])
    assert rows[1][4] == "嘿嘿～好开心！"
    assert rows[1][3] == "0.900"                     # 置信度数值列不受影响
    assert rows[1][9].startswith("result-")          # 记录ID 不带前缀

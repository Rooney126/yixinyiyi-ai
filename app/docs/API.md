# 译芯怡意 · 接口文档（cloud_server.py）

> 以代码实际行为为准（2026-09-07 冲刺版）。
> 服务：Flask，默认端口 `5173`；生产建议 `waitress-serve --port=5173 cloud_server:app`。
> 数据库/录音目录可用环境变量替换：`YXY_DB_PATH`、`YXY_RECORDINGS_DIR`；管理口令 `YXY_STATS_KEY`（不设则用源码内置演示值）。

## 通用约定

- **限流**：标注 🔒 的接口按 IP（优先取 `X-Forwarded-For` 首段）限 **10 次/分钟**，超限返回 `429 {"error": "请求太频繁，请稍后再试"}`。
- **口令**：标注 🔑 的接口需 `?key=` 访问口令（`YXY_STATS_KEY`），错误一律 `403`。
- **上传体积**：全站请求上限 32MB，超限 `413`。
- 所有时间戳均为**毫秒**（除非另注明）；所有接口返回 JSON（除页面/导出/周报）。

---

## 页面与健康检查

### GET `/`
主站单页应用（frontend/index.html）。响应：HTML。

### GET `/cats`
猫档案页（frontend/cat-archive.html）。响应：HTML。

### GET `/health`
健康检查。响应：`{"status": "ok"}`

### GET `/<path>`
静态文件（仅 frontend/ 目录内）；路径穿越（含 `..`）返回 `404 {"error": "not found"}`；不存在的前端文件回落主站 index.html（SPA 兜底）；`/api/*` 前缀返回 `404 {"error": "not found"}`。

### GET `/stats?key=…` 🔑
数据监控页（管理员）。无/错 key 返回 403 文本提示。正确时返回统计 HTML（识别结果分布、来源、家庭视图、最近 100 条请求明细，IP 脱敏），含"用户反馈质量"区块（反馈率 = result_feedback 行数 ÷ records 行数、对/错分布、混淆 TOP10 = "不准确"反馈按 records.label × correct_label 聚合，仅统计识别记录仍存在的反馈）。

---

## 识别（旧功能）

### POST `/api/translate/predict` 🔒
猫叫音频 → 情绪识别。`multipart/form-data`：

| 参数 | 位置 | 说明 |
|---|---|---|
| `file` | form-data | 音频文件（webm/wav 等），后缀白名单外的自动回落 `.webm` |
| `X-Cat-Id` | header | 可选。当前猫 id；猫存在且未归档时识别记录自动归属它，否则落为未归属 |
| `uid` | query | 可选。设备匿名 uid（≤64 字符），用于事后认领；缺省取 cookie `yxy_uid` |

成功 200（字段含义见 index.html 渲染逻辑）：

```json
{
  "id": "result-1762512000000",
  "timestamp": 1762512000000,
  "emotionState": "开心",
  "emotionEmoji": "😊",
  "catIntent": "它很开心",
  "behaviorAdvice": "猫咪现在很开心，趁这个状态多陪它玩吧",
  "confidence": 0.912,
  "label": "Happy",
  "top3": [{"label": "Happy", "prob": "91.2%"}],
  "translation": "嘿嘿～好开心！陪我玩嘛～",
  "confidence_level": "high",
  "alternative": null,
  "distribution": [{"label": "Happy", "prob": 0.912, "emoji": "😊", "translation": "嘿嘿～好开心！陪我玩嘛～", "state": "开心"}],
  "n_segments": 2,
  "gate_score": 0.95,
  "gate_filtered": 1,
  "privacy_version": "v1"
}
```

**`privacy_version`**：当前隐私声明版本。前端把该值与 localStorage 里已确认的版本比较，不一致时强制重新弹出隐私提示（声明升级通道，见 `docs/隐私说明.md`）。仅在识别成功（200）响应中携带。

错误（均 400，`id` 为 null）：`没有音频文件`（缺 file）/ `无法解析音频: …`（解码失败）/ `没有听到猫叫声，请靠近猫咪等它叫`（VAD 无段）/ `听到声音了，但不像是猫叫`（门卫拦截）/ `音频无法识别`（NaN）。其他异常 500 只返回 `{"error": "…"}`（不含 traceback）。超过 32MB 返回 413。

识别成功即写 records 表（置信度 ≥0.8 时另存 .npy/.json 特征到录音目录）。

### POST `/api/translate/feedback`
用户反馈落 JSONL。body（JSON）：`result_id`(≤128)、`label`(≤64)、`confidence`(数字)、`feedback`(≤64)、`timestamp`(整数毫秒)。
- 200：`{"ok": true}`
- 400：`confidence 要是数字` / `timestamp 要是整数毫秒`（类型不对时）

### DELETE `/api/translate/pending/<result_id>`
删除一条录音（`.wav`）。result_id 白名单 `[A-Za-z0-9._-]{1,80}` 且禁 `..`：违规 400 `{"error": "记录ID不合法"}`；文件不存在也算成功，200 `{"ok": true}`。

### POST `/api/event`
匿名埋点。body：`event`(≤50)、`path`(≤100)、`ref`(≤200)。落 events.jsonl。响应 `{"ok": true}`（异常 500 `{"ok": false}`）。

---

## 猫档案（M1）

### GET `/api/cats`
猫列表。query：`include_archived=1` 时包含已归档。
每只含 `record_count`（关联记录数）与 `last_record_at`（最近记录毫秒，无为 0）。

```json
{"cats": [{"id": "cat-1762512000000ab", "name": "测试猫", "color": "#B5D8CC",
  "emoji": "🐱", "birthday": "", "note": "", "archived": false,
  "created_at": 1762512000000, "updated_at": 1762512000000,
  "record_count": 3, "last_record_at": 1762515000000}]}
```

### POST `/api/cats` 🔒
建猫。body（JSON）：

| 字段 | 校验 | 默认 |
|---|---|---|
| `name` | 必填，1~20 字 | — |
| `color` | `#RRGGBB`，否则 400 `颜色格式不对` | `#B5D8CC` |
| `emoji` | ≤8 字 | `🐱` |
| `birthday` | 截断 20 字 | `""` |
| `note` | 截断 100 字 | `""` |

- 201：`{"cat": {…同上结构}}`
- 400：`猫名字要 1~20 个字` / `颜色格式不对` / `表情太长了`

### PATCH `/api/cats/<cat_id>` 🔒
改猫资料（部分更新，传哪个字段改哪个，校验同上）。空 body 400 `没有要修改的内容`；猫不存在 404 `没有找到这只猫`。成功 200 返回更新后的 `{"cat": …}`。

### DELETE `/api/cats/<cat_id>` 🔒
归档（软删除，保留档案与历史记录）。200 `{"ok": true, "archived": true}`。

### POST `/api/cats/<cat_id>/restore` 🔒
恢复归档。200 `{"ok": true, "cat": {…}}`。

### GET `/api/cats/<cat_id>/records`
时间线（新→旧）。query：`limit`（默认 50，钳 1~200）、`offset`（默认 0）、`days`（按天数过滤，`created_at >= now-days*86400000`）。

```json
{"records": [{"result_id": "result-…", "cat_id": "cat-…", "label": "Happy",
  "state": "开心", "translation": "嘿嘿～好开心！", "advice": "…", "emoji": "😊",
  "confidence": 0.9, "n_segments": 2, "gate_score": 0.92, "fuzzy": false,
  "source": "web", "device_uid": "u_x", "created_at": 1762512000000}],
 "total": 1, "limit": 50, "offset": 0, "has_more": false}
```

limit/offset 非数字 → 400 `参数不对`；猫不存在 → 404。

### GET `/api/cats/<cat_id>/stats`
近 N 天统计。query：`days`（默认 30，钳 1~365，非数字回落 30）。

```json
{"days": 30, "total": 5, "negative_count": 1, "avg_confidence": 0.88,
 "distribution": [{"label": "Happy", "state": "开心", "emoji": "😊", "count": 4,
   "pct": 80.0, "avg_confidence": 0.9}],
 "daily": [{"date": "2026-09-01", "label": "09-01", "total": 2, "negative": 0,
   "counts": {"Happy": 2}}]}
```

异常情绪口径（negative）：Angry / Defence / Fighting / Warning / Paining。

### GET `/api/cats/<cat_id>/report/weekly`
可打印 HTML 周报（独立文档）。query：`offset`（0=本周，1=上周…钳 0~52，周一起算）、`dl=1`（作为附件下载，中文文件名）。猫名/表情等用户输入已 HTML 转义。猫不存在 404。

### GET `/api/cats/<cat_id>/export`
导出该猫全部记录（旧→新）。query：`format=csv|json`（默认 csv）。
- CSV：UTF-8 带 BOM，中文表头（时间/情绪(中文)/情绪(英文)/置信度/猫话翻译/养护建议/叫声段数/门卫分/模糊结果/记录ID）；以 `= + - @` 开头的单元格前缀单引号防 Excel 公式注入。
- JSON：`{"cat": {…}, "records": […], "exported_at": 毫秒, "app": "译芯怡意·猫档案"}`。
- 附件中文名（filename* UTF-8）。

---

## 记录归属

### POST `/api/records/<result_id>/assign` 🔒
改归属。body：`{"cat_id": "cat-…"}` 关联；`{"cat_id": null}` 取消归属。
- 200：`{"ok": true, "cat_id": "…"}`
- 404：`没有找到这只猫` / `没有找到这条记录`
- 400：`这只猫的档案已归档，请先恢复档案再认领记录`

### DELETE `/api/records/<result_id>` 🔒
删除一条识别记录。200 `{"ok": true}`；不存在 404 `没有找到这条记录`。

### GET `/api/records/unassigned`
本设备未归属记录（新→旧）。query：`uid`（必填，≤64；不带返回空列表）、`limit`（默认 20，钳 1~50，非数字回落 20）。按设备隔离，看不到其他设备的记录。

---

## 识别结果反馈（一键反馈）

### POST `/api/records/<result_id>/feedback` 🔒
识别结果一键反馈，结构化落 `result_feedback` 表（一条识别只留一条反馈，重复提交=改答案，答案以最后一次为准）。与旧通道 `POST /api/translate/feedback`（JSONL）并行，互不影响。

| 字段 | 位置 | 说明 |
|---|---|---|
| `is_correct` | body | 必填，JSON 布尔：`true`=识别准确，`false`=不准确；非布尔 400 |
| `correct_label` | body | `is_correct=false` 时必填，10 类英文标签之一（Angry/Defence/Fighting/Happy/HuntingMind/Mating/MotherCall/Paining/Resting/Warning），否则 400；`true` 时存空串 |
| `note` | body | 可选备注，≤200 字，超长 400 |
| `uid` | query | 可选。设备匿名 uid（>64 截断）；缺省回落 cookie `yxy_uid`（与 predict 同款逻辑） |

- 200：`{"ok": true, "updated": false}`（首次反馈）/ `{"ok": true, "updated": true}`（改答案）
- 400：`记录ID不合法`（result_id 白名单 `[A-Za-z0-9._-]{1,80}` 且禁 `..`）/ `is_correct 要是布尔值：true=准确，false=不准确` / `请从 10 类情绪里选择正确的类别` / `备注最多 200 字`
- 404：`没有找到这条识别记录`

入库字段：`result_id`（UNIQUE，对应 records.result_id）、`is_correct`（1/0）、`correct_label`、`note`、`source`（'web'）、`device_uid`、`created_at`/`updated_at`（毫秒）。随 `/api/export/backup` 的 `result_feedback` 数组导出；`/stats` 页"用户反馈质量"区块以此为口径。

---

## 驻置模式（旧手机常驻听猫叫）

页面：`GET /station`（常驻采集页）与 `GET /station-board`（看板），独立静态页，显式路由（不会被 SPA 兜底接走）。
识别入库复用 `POST /api/translate/predict`，驻置页上传时带 `X-Source: station` 请求头（白名单只有 `web`/`station`，其余值与缺省一律落 `web`）+ `?uid=station-…` 设备号 + 可选 `X-Cat-Id`，records.source 相应为 `station`。

### POST `/api/station/heartbeat` 🔒
驻置设备心跳，upsert `stations` 表（首插写 first_seen，更新保留 first_seen）。采集页启动即发 1 次、此后每 30s 一次。body（JSON）：

| 字段 | 校验 | 默认 |
|---|---|---|
| `device_uid` | 必填，`[A-Za-z0-9._-]{1,64}` 且禁 `..` | — |
| `name` | ≤20 字（超长截断） | `""` |
| `battery` | 0~1 数字；缺省存 NULL | NULL |
| `charging` | 任意（真值→1，其余→0）；缺省 NULL | NULL |
| `uptime_s` / `triggered_total` / `uploaded_total` | 非负整数 | 0 |

- 200：`{"ok": true}`
- 400：`设备号要 1~64 位字母数字（station- 前缀）` / `battery 要是 0~1 的数字` / `battery 要在 0~1 之间` / `uptime_s 要是非负整数` 等

### GET `/api/station/status?key=…` 🔑
看板数据：设备列表 + BP 指标。无/错 key 403 JSON（与 /stats 同口令）。

```json
{"devices": [{"device_uid": "station-xxx", "name": "驻置设备",
  "first_seen": 1790300000000, "last_seen": 1790300000000,
  "battery": 0.66, "charging": 1, "uptime_s": 3600,
  "triggered_total": 12, "uploaded_total": 10,
  "online": true, "today_count": 3, "accepted_total": 10}],
 "metrics": {"health_today": 1, "health_week": 2,
   "daily": [{"date": "2026-09-25", "label": "09-25", "total": 3,
              "health": 1, "avg_confidence": 0.62}]},
 "server_time": 1790300000000, "online_window_ms": 120000}
```

`online` = 最近心跳距今 < 2 分钟；`today_count` = 该设备今日 records(source='station') 条数；`accepted_total` = 该设备累计入库条数；指标口径只算 source='station'，健康相关 = NEG_LABELS（Angry/Defence/Fighting/Warning/Paining），`daily` 为近 7日本地日界零填充序列（无记录日 avg_confidence 为 null）。

### GET `/api/station/events?key=…&hours=24&uid=&limit=200` 🔑
驻置事件流（records.source='station'，复用记录结构），`created_at` 倒序。query：`hours`（1~720，默认 24）、`uid`（可选，按设备过滤）、`limit`（1~200，默认 200，超界回落 200）。无/错 key 403。

```json
{"events": [{"result_id": "result-…", "cat_id": "cat-…", "label": "Happy",
  "state": "开心", "confidence": 0.9, "source": "station",
  "device_uid": "station-xxx", "created_at": 1790300000000, "…": "…同 /api/cats/<id>/records 条目"}],
 "total": 3, "hours": 24, "limit": 200}
```

`total` 为窗口内总条数（不受 limit 截断）；hours/limit 非整数 400 `hours/limit 要是整数`。

---

## 备份

### GET `/api/export/backup?key=…` 🔑
全量备份（猫 + 记录 + 旧 feedback 表 + result_feedback 一键反馈表）单个 JSON 附件下载。无/错 key 403（文本提示）；正确返回 `{"app": "译芯怡意·猫档案 全量备份", "format_version": 1, "exported_at": …, "cats": […], "records": […], "feedback": […], "result_feedback": […]}`。

---

## 状态码速查

| 码 | 出现场景 |
|---|---|
| 400 | 参数/校验类错误（中文提示）、识别失败类 |
| 403 | /stats、/api/export/backup 口令错误 |
| 404 | 猫/记录不存在、api 未知路径、路径穿越 |
| 405 | 请求方法不匹配（如 DELETE 到含 `/` 的 pending id） |
| 413 | 请求体超过 32MB |
| 429 | 触发限流（🔒 接口，10 次/分/IP） |
| 500 | 服务端异常（predict 的 500 不含 traceback） |

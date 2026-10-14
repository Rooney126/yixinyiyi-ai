# 猫健康看板 · 用户版（部署后挂 your-server.example/app）

「我家猫今天怎么样」——为养猫用户做的移动优先看板：打开就知道猫今天的情绪轨迹、健康相关事件、设备是否在线。**原生 HTML/CSS/JS 单文件实现，零外部依赖**（无 CDN / 框架 / 字体 / 图标库 / 图表库，图标全用 emoji 与内联 SVG），可在离线内网环境运行。

## 产出文件

| 文件 | 说明 |
|---|---|
| `index.html` | 看板全部页面与逻辑（内联 CSS/JS，约 520 行） |
| `dev_server.py` | 仅本地开发用：静态服务 + `/api/*` 反向代理，**不参与部署** |
| `README.md` | 本文件 |

## 页面结构

```
① 口令门（无口令/口令失效时全屏）
   输入访问口令 → 探测 GET /stats?key=<输入>（同一 STATS_KEY）
   200 → 存 localStorage（键 yxy_userboard_key）进入看板
   403 → 「口令错误」；网络失败 → 「网络异常 + 重试」

② 主看板（375px 起步，桌面居中限宽 720px，暗色主题）
   ③ 猫切换器        横向滑动 chips，默认选列表第一只（建档最早）
   ④ 健康概览卡      当前猫的今日/近7日健康相关事件数
   ⑤ 今日情绪时间线  当天识别记录（新→旧）：时间+emoji+中文解读+置信度
   ⑥ 设备状态条      每台设备一行：三态徽标/电量/运动态/今日次数/心跳时间；
                     fence_alert=1 时行内红条「最近越界 + 时间」
   ⑦ 近 7 日趋势     识别量柱（CSS）+ 负面占比线（内联 SVG）
```

每个数据分区（④⑤⑥⑦）有独立的加载骨架、空态与「加载失败 [重试]」占位，只重试本区，任何情况下不白屏。所有请求带 10 秒超时。

## 接口依赖清单（全部相对路径，与 /api 同源）

| 接口 | 用途 | 需要口令 |
|---|---|---|
| `GET /api/cats` | 猫切换器（id/name/emoji/color/record_count） | 否 |
| `GET /api/cats/<id>/records?days=1&limit=200` | 今日时间线（label/state/translation/emoji/confidence/fuzzy） | 否 |
| `GET /api/cats/<id>/stats?days=7` | 健康卡（negative_count、daily）+ 趋势图（daily 末 7 格） | 否 |
| `GET /api/station/status?key=` | 设备状态（devices[]）+ 口令验证探测 | **是** |

页面内嵌的映射表（与 `cloud_server.py` 保持同一口径，改后端需同步）：

- 情绪文案 `DOME_MAP`：10 类 label → [emoji, 描述, 猫话翻译, 中文名]；模糊态 `PAIN_HAPPY_FUZZY`（😶/不确定）。
- 健康事件口径 `NEGATIVE_LABELS`：Angry / Defence / Fighting / Warning / Paining。
- 项圈运动态 `motion_state` → 中文：resting 静卧 / walking 行走 / running 奔跑 / jumping 跳跃 / impact 碰撞。

## 口令机制

- 与现有 `/stats` 运维看板**同一套 `STATS_KEY`**（后端 `YXY_STATS_KEY`），以 `?key=` 查询参数传输，后端 `hmac.compare_digest` 校验。
- 首访输入 → 用 `GET /stats?key=` 探测（同一 STATS_KEY，且新旧版后端都有此路由）→ 通过后存 `localStorage`（键 `yxy_userboard_key`），之后免输；右上 ⚙️ 可清除并重输。
- 口令错误（403）有明确提示；已存口令直接进入（不重复探测），口令失效由设备接口返回 403 时兜底退回口令门。
- 若后端还没部署 `/api/station/*`（旧版会 404），口令门可正常通过（探测走 /stats），仅「设备状态」分区独立显示加载失败，其余分区不受影响。
- 说明：猫档案/时间线/统计这三个接口后端本就不校验口令，口令门是页面层准入（任务书要求），口令真正作用于设备接口。

## 判定口径与已知边界

- 设备三态按**服务器时钟** `server_time − last_seen` 计算（防手机时钟不准）：<2min 在线；2~35min 休眠（深睡设备）；≥35min 离线。
- `fence_alert` 是闩锁字段（后端 upsert 逻辑决定一旦 =1 不会被心跳清零），故文案为「**最近越界** + fence_alert_ms 时间」而非「正在越界」。
- 「今天」按用户手机本地零点过滤（接口 `days=1` 是滚动 24h 窗口，前端再按本地零点筛）。
- 时间线上限 200 条/天（接口 limit 上限），超出显示「仅显示最近 200 条」提示。
- `index.html` 内不含任何 `http(s)://` 外链资源（趋势图的 SVG 由 JS 以数字拼出，无图表库、无外链）。

## 本地运行

```bash
cd webapp
python dev_server.py
# 浏览器打开 http://127.0.0.1:8000
```

- `/api/*` 与 `/stats`（口令探测用）被反向代理到 `https://your-server.example`（透传 query、key 与客户端 User-Agent——线上有 Cloudflare，需真实 UA 才放行），静态文件本地服务，因此本地即可连**真实数据**验证。
- 环境变量：`DEV_UPSTREAM`（换上游，如指向本地 mock 服务器）、`DEV_PORT`（默认 8000）。
- 代理仅支持 GET（看板只用 GET）；该文件为标准库实现（≤40 行），**部署不含它**。

## 部署（人工操作）

把 `index.html` 挂到 cloud_server 的 `/app` 路由即可——单文件自包含，无构建步骤；`dev_server.py` 与 `README.md` 不参与部署。

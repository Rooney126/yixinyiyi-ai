# 译芯怡意 · 猫语通（YixinYiyi CatVoice AI）

多模态 AI 驱动的宠物全生命周期服务平台。从「猫叫声理解」出发，让养宠人真正听懂自家猫：通过端侧实时识别 + 云端细粒度分析 + 数据闭环迭代，把猫的情绪、健康信号变成养宠人看得懂的日常信息。

> 学生创业项目（物联网工程方向）。本仓库公开训练、服务端、前端全部核心代码；模型权重与真实用户数据不在此仓库。

## 亮点

- **端云协同**：MCU（ESP32-S3 等）上跑蒸馏后的小模型做唤醒/门卫过滤，云端跑 10 类细粒度叫声分类
- **模型压缩落地**：教师-学生知识蒸馏（`kd_tiny_mcu_distill.py`），把 CNN / DS-ResNet 蒸馏到微型 MCU 可部署的尺寸，导出 TFLite / ONNX
- **真实数据闭环**：自采标注管线 + 内测家庭反馈回流，形成「采集 → 标注 → 训练 → 上线 → 反馈」飞轮
- **工程完整**：单元测试、安全测试、部署预检脚本齐全；Flask 生产服务 + SQLite

## 架构

```
[猫项圈 / 驻置设备]
        │  实时音频
        ▼
[MCU 端侧模型]  ← 蒸馏后的小模型（唤醒 / 二分类门卫，TFLite）
        │  触发事件
        ▼
[云端服务 Flask] ─── SQLite（用户 / 事件 / 反馈）
        │
        ├── 10 类叫声识别（FDAP CNN / DS-ResNet）
        ├── 情绪与健康事件看板
        └── 数据飞轮（复核标注 → 再训练）
```

## 目录结构

```
train/          训练与研究（PyTorch）
  train/        主训练脚本：FDAP CNN、DS-ResNet、知识蒸馏（kd_tiny_mcu_distill.py）、TFLite/ONNX 导出
  experiments/  数据增强与事件分割实验
  eval/         评估与推理（含 Wav2Vec2/HuBERT 对比实验）
  analysis/     嵌入空间与 t-SNE 可视化分析
app/            应用服务
  server.py     本地版服务（Flask，Windows 原生版 v2）
  docs/         技术文档（API、标注管线、使用说明）
  tests/        单元测试 / 安全测试 / 部署预检测试
frontend/       用户看板前端（Vue 3 + Pinia + Vite + TS）
webapp/         移动优先看板（原生 HTML/CSS/JS 单文件，零外部依赖）
docs/           技术方案沉淀、数据集报告、界面截图
```

## 技术栈

| 层 | 技术 |
|---|---|
| 训练 | Python · PyTorch · librosa · noisereduce |
| 端侧 | 知识蒸馏 · TFLite / ONNX 导出 · MCU 部署 |
| 服务端 | Flask · SQLite · waitress |
| 前端 | Vue 3 · Pinia · Vite · TypeScript；原生单文件版（webapp/） |
| 工程 | pytest · 安全测试 · 部署预检脚本 · 数据脱敏 |

## 快速开始

```bash
# 1. 训练（以 FDAP CNN 为例）
python train/train/train_fdap_cnn_v2_ch64.py

# 2. 启动本地服务
python app/server.py

# 3. 打开 http://127.0.0.1:5173 体验
```

详细说明见 `app/docs/` 与 `docs/技术方案沉淀_v1.md`。

## 数据与隐私

- 真实用户/内测数据不进入本仓库；接口层做了口令保护、限流、录音短存与脱敏
- 训练脚本中的数据集路径为占位符，需按你的实际路径调整

## License

MIT © Rooney126

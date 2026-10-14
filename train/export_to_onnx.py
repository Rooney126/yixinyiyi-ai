r"""
FDAPMobileCatCNN PyTorch → ONNX 导出
用于阿里云 CPU 推理部署

用法:
    python train/export_to_onnx.py

输入:  app/model/fold3_best.pth
输出:  app/model/model.onnx  (用于云服务器)
       app/model/model.quant.onnx  (可选: int8 量化版)
"""

import os, sys, torch
import numpy as np

# ── 添加路径 ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "train"))

from train.train_cnn_pipeline_v2_vadexp import FDAPMobileCatCNN

# ── 配置 ──
CKPT_PATH = os.path.join(PROJECT_DIR, "app", "model", "fold3_best.pth")
ONNX_PATH = os.path.join(PROJECT_DIR, "app", "model", "model.onnx")
ONNX_QUANT_PATH = os.path.join(PROJECT_DIR, "app", "model", "model.quant.onnx")

# ── 1. 加载 PyTorch 模型 ──
print("🚀 加载 PyTorch 模型...")
device = torch.device("cpu")
model = FDAPMobileCatCNN(num_classes=10).to(device)
state = torch.load(CKPT_PATH, map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()
print(f"  ✅ 已加载: {CKPT_PATH}")

# ── 2. 验证前向推理 ──
dummy = torch.randn(1, 1, 128, 80)
with torch.no_grad():
    ref_out = model(dummy)
ref_probs = torch.softmax(ref_out, dim=1)
ref_pred = int(torch.argmax(ref_probs[0]))
print(f"  ✅ PyTorch 参考推理: 类 {ref_pred}, 置信度 {ref_probs[0][ref_pred]:.4f}")

# ── 3. 导出 ONNX ──
print("\n📦 导出 ONNX...")
torch.onnx.export(
    model,
    dummy,
    ONNX_PATH,
    input_names=["mel"],           # (N, 1, 128, 80) Mel频谱
    output_names=["logits"],       # (N, 10) 原始logits
    dynamic_axes={
        "mel": {0: "batch"},       # 批量可变的段数
        "logits": {0: "batch"},
    },
    opset_version=17,              # 广泛兼容
    do_constant_folding=True,
)
print(f"  ✅ ONNX 已导出: {ONNX_PATH}")

# ── 4. 验证 ONNX 输出 ──
print("\n🔍 验证 ONNX 输出...")
import onnxruntime as ort

sess = ort.InferenceSession(ONNX_PATH)
ort_out = sess.run(None, {"mel": dummy.numpy()})[0]
ort_probs = torch.softmax(torch.tensor(ort_out), dim=1)
ort_pred = int(torch.argmax(ort_probs[0]))

diff = np.abs(ref_out.numpy() - ort_out).max()
print(f"  ONNX 推理: 类 {ort_pred}, 置信度 {ort_probs[0][ort_pred]:.4f}")
print(f"  输出最大差异: {diff:.6f}")

if diff < 1e-4:
    print("  ✅ 输出一致，导出成功！")
else:
    print(f"  ⚠️ 差异 {diff:.6f} > 1e-4，建议检查")

# ── 5. (可选) int8 量化 ──
print("\n📦 导出 int8 量化版 (可选)...")
try:
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType

    model_onnx = onnx.load(ONNX_PATH)
    quantize_dynamic(
        ONNX_PATH,
        ONNX_QUANT_PATH,
        weight_type=QuantType.QInt8,
    )
    quant_size = os.path.getsize(ONNX_QUANT_PATH)
    origin_size = os.path.getsize(ONNX_PATH)
    print(f"  原始大小: {origin_size / 1024:.1f} KB")
    print(f"  量化大小: {quant_size / 1024:.1f} KB")
    print(f"  压缩比: {origin_size / quant_size:.1f}x")
    print(f"  ✅ int8 量化版: {ONNX_QUANT_PATH}")
except Exception as e:
    print(f"  ⚠️ 量化失败 (非必须): {e}")

print("\n🎉 完成！")

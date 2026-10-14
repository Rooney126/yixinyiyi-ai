r"""
将 FDAPMobileCatCNN PyTorch 模型导出为 TFLite int8 格式
用于 T5 AI Board 的 TFLite Micro 推理

用法:
    python export_to_tflite.py

输入:  train/output/fdap_cnn_ch64_A_aug/fold3_best.pth  (87.9% F1)
输出:  train/output/fdap_cnn_ch64_A_aug/model.tflite
"""

import os, sys, torch
import numpy as np
import json

# ── 添加模型定义所在目录到路径 ──────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "train"))

from train_cnn_pipeline_v2_vadexp import FDAPMobileCatCNN

# ── 配置 ─────────────────────────────────────────────
MODEL_DIR = os.path.join(SCRIPT_DIR, "output", "fdap_cnn_ch64_A_aug")
CKPT_PATH = os.path.join(MODEL_DIR, "fold3_best.pth")
TFLITE_PATH = os.path.join(MODEL_DIR, "model.tflite")

# ── 加载 PyTorch 模型并提取权重 ─────────────────────
print("加载 PyTorch 模型...")
device = torch.device("cpu")
pt_model = FDAPMobileCatCNN(num_classes=10).to(device)
state = torch.load(CKPT_PATH, map_location=device, weights_only=True)
pt_model.load_state_dict(state)
pt_model.eval()
print(f"  ✅ 已加载: {CKPT_PATH}")

# 用随机输入跑一次，看预测结果作为基准
dummy_input = torch.randn(1, 1, 128, 80)
with torch.no_grad():
    pt_out = pt_model(dummy_input)
pt_probs = torch.softmax(pt_out, dim=1)
pt_pred = int(torch.argmax(pt_probs[0]))
print(f"  参考推理: 类 {pt_pred}, 概率 {pt_probs[0][pt_pred]:.4f}")

# ── 提取权重 ────────────────────────────────────────
print("\n提取权重...")
weights = {}
for name, param in pt_model.named_parameters():
    if param.requires_grad:
        weights[name] = param.detach().numpy()
        print(f"  {name}: {param.shape}")

# ── 在 TF 中重建等效模型 ────────────────────────────
print("\n在 TensorFlow 中重建模型...")
import tensorflow as tf
tf.config.run_functions_eagerly(True)

def fdap_fn(x):
    # x shape: (b, h, w, c) = (b, freq, time, ch)
    h = tf.shape(x)[1]
    bs = h // 4
    b0 = tf.reduce_mean(x[:, 0:bs, :, :], axis=[1, 2])
    b1 = tf.reduce_mean(x[:, bs:2*bs, :, :], axis=[1, 2])
    b2 = tf.reduce_mean(x[:, 2*bs:3*bs, :, :], axis=[1, 2])
    b3 = tf.reduce_mean(x[:, 3*bs:, :, :], axis=[1, 2])
    return tf.concat([b0, b1, b2, b3], axis=1)

model = tf.keras.Sequential([
    # Conv1: 1→16ch, 3x3, pad=1, BN, ReLU, MP2
    tf.keras.layers.Conv2D(16, 3, padding="same", input_shape=(128, 80, 1), name="conv1_conv"),
    tf.keras.layers.BatchNormalization(axis=-1, epsilon=1e-5, name="conv1_bn"),
    tf.keras.layers.ReLU(name="conv1_relu"),
    tf.keras.layers.MaxPool2D(2, name="conv1_pool"),

    # Conv2: 16→32ch
    tf.keras.layers.Conv2D(32, 3, padding="same", name="conv2_conv"),
    tf.keras.layers.BatchNormalization(axis=-1, epsilon=1e-5, name="conv2_bn"),
    tf.keras.layers.ReLU(name="conv2_relu"),
    tf.keras.layers.MaxPool2D((1, 2), name="conv2_pool"),

    # Conv3: 32→48ch
    tf.keras.layers.Conv2D(48, 3, padding="same", name="conv3_conv"),
    tf.keras.layers.BatchNormalization(axis=-1, epsilon=1e-5, name="conv3_bn"),
    tf.keras.layers.ReLU(name="conv3_relu"),
    tf.keras.layers.MaxPool2D((1, 2), name="conv3_pool"),

    # Conv4: 48→64ch
    tf.keras.layers.Conv2D(64, 3, padding="same", name="conv4_conv"),
    tf.keras.layers.BatchNormalization(axis=-1, epsilon=1e-5, name="conv4_bn"),
    tf.keras.layers.ReLU(name="conv4_relu"),

    # FDAP: 频率分带均值池化
    tf.keras.layers.Lambda(fdap_fn, name="fdap"),

    # Classifier
    tf.keras.layers.Dense(96, activation="relu", name="fc1"),
    tf.keras.layers.Dense(10, name="fc2"),
])

# ── 加载 PyTorch 权重到 TF 模型 ────────────────────
print("\n加载权重到 TF 模型...")

# Conv 权重: PyTorch 格式 (out, in, h, w) → TF 格式 (h, w, in, out)
conv_pairs = [
    ("conv1", "conv1_conv"),
    ("conv2", "conv2_conv"),
    ("conv3", "conv3_conv"),
    ("conv4", "conv4_conv"),
]
for pt_prefix, tf_name in conv_pairs:
    w = weights.get(f"{pt_prefix}.0.weight")
    b = weights.get(f"{pt_prefix}.0.bias")
    if w is not None:
        model.get_layer(tf_name).set_weights([
            np.transpose(w, (2, 3, 1, 0)),  # Conv weight
            b,
        ])
        print(f"  ✅ {tf_name}: w={w.shape}→{np.transpose(w, (2,3,1,0)).shape}")

# BN 权重: PyTorch → TF 格式相同
bn_pairs = [
    ("conv1", "conv1_bn"),
    ("conv2", "conv2_bn"),
    ("conv3", "conv3_bn"),
    ("conv4", "conv4_bn"),
]
for pt_prefix, tf_name in bn_pairs:
    gamma = state.get(f"{pt_prefix}.1.weight")
    beta = state.get(f"{pt_prefix}.1.bias")
    mean = state.get(f"{pt_prefix}.1.running_mean")
    var = state.get(f"{pt_prefix}.1.running_var")
    if gamma is not None:
        model.get_layer(tf_name).set_weights([gamma.numpy(), beta.numpy(), mean.numpy(), var.numpy()])
        print(f"  ✅ {tf_name}")

# FC 权重
fc_pairs = [
    ("classifier.1", "fc1"),
    ("classifier.4", "fc2"),
]
for pt_name, tf_name in fc_pairs:
    w = weights.get(f"{pt_name}.weight")
    b = weights.get(f"{pt_name}.bias")
    if w is not None:
        model.get_layer(tf_name).set_weights([np.transpose(w), b])
        print(f"  ✅ {tf_name}: w={w.shape}→转置")

# ── 验证 TF 输出 ────────────────────────────────────
print("\n验证 TF 输出...")
# PyTorch: NCHW (1,1,128,80) → TF: NHWC (1,128,80,1)
tf_input = np.transpose(dummy_input.numpy(), (0, 2, 3, 1))
tf_out = model(tf_input, training=False).numpy()
diff = np.max(np.abs(tf_out - pt_out.numpy()))
print(f"  PyTorch vs TF 最大差异: {diff:.6f}")
assert diff < 0.01, f"❌ TF 输出不一致! 差异={diff:.6f}"
print("  ✅ TF 输出一致")

# ── 导出 TFLite int8 ────────────────────────────────
print("\n导出 TFLite int8...")

# 保存 SavedModel
save_path = os.path.join(os.environ.get('TEMP', '.'), 'catcnn_savedmodel')
import shutil
if os.path.exists(save_path):
    shutil.rmtree(save_path)
model.export(save_path)
print(f"  SavedModel: {save_path}")


# float32 导出（不量化）
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)
print(f"  TFLite: {TFLITE_PATH} ({len(tflite_model)//1024}KB)")

# ── 验证 TFLite 推理 ────────────────────────────────
print("\n验证 TFLite 推理...")
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_scale, input_zero = input_details[0]["quantization"]
output_scale, output_zero = output_details[0]["quantization"]
print(f"  输入: scale={input_scale:.6f}, zero={input_zero}")
print(f"  输出: scale={output_scale:.6f}, zero={output_zero}")

# float32 推理
interpreter.set_tensor(input_details[0]["index"], tf_input.astype(np.float32))
interpreter.invoke()
tflite_out = interpreter.get_tensor(output_details[0]["index"]).astype(np.float32)

# softmax 对比
tflite_probs = np.exp(tflite_out) / np.sum(np.exp(tflite_out), axis=1, keepdims=True)
prob_diff = np.max(np.abs(pt_probs.numpy() - tflite_probs))
tflite_pred = int(np.argmax(tflite_probs[0]))

print(f"  PyTorch → 类 {pt_pred}  概率 {pt_probs[0][pt_pred]:.4f}")
print(f"  TFLite  → 类 {tflite_pred}  概率 {tflite_probs[0][tflite_pred]:.4f}")
print(f"  最大概率差异: {prob_diff:.4f}")

if pt_pred == tflite_pred and prob_diff < 0.05:
    print("  ✅ TFLite 导出成功，结果一致！")
else:
    print("  ⚠️ 预测不一致，检查权重加载")

# 保存量化参数供 T5 端使用
params = {
    "input_scale": float(input_scale),
    "input_zero_point": int(input_zero),
    "output_scale": float(output_scale),
    "output_zero_point": int(output_zero),
}
params_path = TFLITE_PATH.replace(".tflite", "_params.json")
with open(params_path, "w") as f:
    json.dump(params, f, indent=2)
print(f"\n量化参数: {params_path}")
print(f"  {params}")
print(f"\n完成！")

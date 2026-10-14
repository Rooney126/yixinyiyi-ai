r"""
导出 TinyMCUCatCNN (Student) → TFLite int8 + C数组
===============================================
用法:
  cd PROJECT_ROOT\train
  /path/to/user\AppData\Local\Programs\Python\Python310\python.exe export_student_tflite.py

输出:
  train\output\kd_tiny_mcu\student_model.tflite
  train\output\kd_tiny_mcu\student_model_data.c    (C数组, 用于T5固件)
  train\output\kd_tiny_mcu\student_model_params.json

依赖:
  pip install tensorflow onnx onnx2tf
"""

import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # TFLite导出走CPU

from pathlib import Path

import numpy as np
import torch
from torch import nn

# ═══════════════════════════════════════
#  配置
# ═══════════════════════════════════════

OUTPUT_DIR = Path('PROJECT_ROOT/train/output/kd_tiny_mcu')
MODEL_DIR = Path('PROJECT_ROOT/train/output/kd_tiny_mcu')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 最佳折权重 — 用折1（目前最高81.5%段级F1），或者你也可以指定其他折
# 后续可改为自动选最佳折
FOLD = 1
CKPT_PATH = OUTPUT_DIR / f'fold{FOLD}_best.pth'

TFLITE_PATH = OUTPUT_DIR / 'student_model.tflite'
C_ARRAY_PATH = OUTPUT_DIR / 'student_model_data.c'
PARAMS_PATH = OUTPUT_DIR / 'student_model_params.json'

DEVICE = 'cpu'

LABELS = ["Angry","Defence","Fighting","Happy","HuntingMind",
          "Mating","MotherCall","Paining","Resting","Warning"]

# ═══════════════════════════════════════
#  Student 模型定义
# ═══════════════════════════════════════

class TinyMCUCatCNN(nn.Module):
    """
    3Conv+GAP, 32×40 Mel (直接输入)
    目标: arena <50KB, 全 SRAM, 推理 ~50ms @M33F
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(),
            nn.MaxPool2d(2))                    # → 8×32×20
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2))                    # → 16×16×10
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2))                    # → 32×8×5
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)


# ═══════════════════════════════════════
#  PyTorch → ONNX → TFLite int8
# ═══════════════════════════════════════

print(f'\n{"█" * 55}')
print('  导出 TinyMCUCatCNN → TFLite int8')
print(f'  Fold: {FOLD}')
print('  Input: 32×40 Mel (1ch)')
print(f'  Output: 10类 ({len(LABELS)})')
print(f'{"█" * 55}\n')

# ── 1. 加载 PyTorch 权重 ──
print(f'[1] 加载折 {FOLD} 权重...')
if not CKPT_PATH.exists():
    print(f'  ❌ 权重不存在: {CKPT_PATH}')
    print('  可用的折:')
    for f in sorted(OUTPUT_DIR.glob('fold*_best.pth')):
        print(f'    {f.name}')
    sys.exit(1)

model = TinyMCUCatCNN(num_classes=10).to(DEVICE)
state = torch.load(str(CKPT_PATH), map_location=DEVICE, weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()
print(f'  ✅ 加载完成 ({CKPT_PATH.name})')

n_params = sum(p.numel() for p in model.parameters())
print(f'  参数量: {n_params:,}')

# ── 2. 测试推理 ──
print('\n[2] 测试推理 (随机输入)...')
dummy = torch.randn(1, 1, 64, 40).to(DEVICE)
with torch.no_grad():
    logits = model(dummy)
    probs = torch.softmax(logits, dim=1)
    pred = int(torch.argmax(probs[0]))
    conf = float(probs[0][pred])
    print(f'  随机输入 → 类 {pred} ({LABELS[pred]}) 置信度 {conf:.4f}')

# ── 3. ONNX 导出 ──
print('\n[3] 导出 ONNX...')
ONNX_PATH = OUTPUT_DIR / 'student_model.onnx'
torch.onnx.export(
    model,
    dummy,
    str(ONNX_PATH),
    input_names=['mel'],
    output_names=['logits'],
    opset_version=18,
    dynamic_axes={'mel': {0: 'batch'}, 'logits': {0: 'batch'}},
)
print(f'  ONNX: {ONNX_PATH} ({ONNX_PATH.stat().st_size//1024}KB)')

# ── 4. TFLite int8 导出 (Keras 手动重建) ──
print('\n[4] 导出 TFLite int8...')
try:
    import tensorflow as tf
except ImportError:
    print('  ❌ 需要安装 tensorflow: pip install tensorflow')
    sys.exit(1)

# 代表数据集 (Mel 空间随机采样)
def representative_dataset():
    for _ in range(200):
        yield [np.random.rand(1, 32, 40, 1).astype(np.float32)]

# ── 手动构建 Keras 等效模型 ──
print('  构建 Keras 模型...')
inputs = tf.keras.Input(shape=(32, 40, 1), name='mel')
x = tf.keras.layers.Conv2D(8, 3, padding='same', use_bias=False, name='conv1')(inputs)
x = tf.keras.layers.BatchNormalization(epsilon=1e-5, name='bn1')(x)
x = tf.keras.layers.ReLU(name='relu1')(x)
x = tf.keras.layers.MaxPool2D(2, name='pool1')(x)
x = tf.keras.layers.Conv2D(16, 3, padding='same', use_bias=False, name='conv2')(x)
x = tf.keras.layers.BatchNormalization(epsilon=1e-5, name='bn2')(x)
x = tf.keras.layers.ReLU(name='relu2')(x)
x = tf.keras.layers.MaxPool2D(2, name='pool2')(x)
x = tf.keras.layers.Conv2D(32, 3, padding='same', use_bias=False, name='conv3')(x)
x = tf.keras.layers.BatchNormalization(epsilon=1e-5, name='bn3')(x)
x = tf.keras.layers.ReLU(name='relu3')(x)
x = tf.keras.layers.MaxPool2D(2, name='pool3')(x)
x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)
outputs = tf.keras.layers.Dense(10, name='fc')(x)
tf_model = tf.keras.Model(inputs=inputs, outputs=outputs)

# ── 搬运 PyTorch 权重 ──
pt_state = model.state_dict()

# Conv2D (PT → TF)
for conv_name, pt_key, ch_out, ch_in in [
    ('conv1', 'conv1.0.weight', 8, 1),
    ('conv2', 'conv2.0.weight', 16, 8),
    ('conv3', 'conv3.0.weight', 32, 16),
]:
    w = pt_state[pt_key].cpu().numpy()  # (out, in, H, W)
    w = w.transpose(2, 3, 1, 0)          # (H, W, in, out)
    tf_model.get_layer(conv_name).set_weights([w])

# BatchNorm
for bn_name, pt_prefix in [('bn1', 'conv1.1'), ('bn2', 'conv2.1'), ('bn3', 'conv3.1')]:
    gamma = pt_state[f'{pt_prefix}.weight'].cpu().numpy()
    beta = pt_state[f'{pt_prefix}.bias'].cpu().numpy()
    running_mean = pt_state[f'{pt_prefix}.running_mean'].cpu().numpy()
    running_var = pt_state[f'{pt_prefix}.running_var'].cpu().numpy()
    # Keras BN: [gamma, beta, mean, variance]
    tf_model.get_layer(bn_name).set_weights([gamma, beta, running_mean, running_var])

# FC
fc_w = pt_state['fc.weight'].cpu().numpy().T  # (32, 10) → (10, 32)
fc_b = pt_state['fc.bias'].cpu().numpy()
tf_model.get_layer('fc').set_weights([fc_w, fc_b])
print('  ✅ 权重搬运完成')

# ── 验证权重一致 ──
print('  验证权重...')
test_in = np.random.rand(1, 32, 40, 1).astype(np.float32)
with torch.no_grad():
    pt_out = model(torch.FloatTensor(test_in.transpose(0, 3, 1, 2)))
    pt_probs = torch.softmax(pt_out, dim=1).numpy()
tf_out = tf_model.predict(test_in, verbose=0)
tf_probs = np.exp(tf_out) / np.sum(np.exp(tf_out), axis=1, keepdims=True)
diff = np.max(np.abs(pt_probs - tf_probs))
print(f'  PT vs TF 最大概率差异: {diff:.6f}')
if diff > 0.01:
    print('  ⚠️ 权重搬运偏差较大，检查网络结构')
else:
    print('  ✅ PT → TF 权重一致')

# 转换 TFLite int8 — 直接从 Keras 模型转换
print('  TFLite int8 转换...')
converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter._experimental_default_to_single_batch_in_tensor_list_ops = True

try:
    tflite_model = converter.convert()
    with open(str(TFLITE_PATH), 'wb') as f:
        f.write(tflite_model)
    print(f'  ✅ TFLite int8: {TFLITE_PATH} ({len(tflite_model)//1024}KB)')
except Exception as e:
    print(f'  ❌ int8 失败: {e}')
    print('  回退到 FP16 量化...')
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    with open(str(TFLITE_PATH), 'wb') as f:
        f.write(tflite_model)
    print(f'  ✅ TFLite FP16: {TFLITE_PATH} ({len(tflite_model)//1024}KB)')

# ── 5. 验证 ──
print('\n[5] 验证 TFLite 推理精度...')
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_scale, input_zero = input_details[0]['quantization']
output_scale, output_zero = output_details[0]['quantization']
print(f'  输入量化: scale={input_scale:.6f}, zero={input_zero}')
print(f'  输出量化: scale={output_scale:.6f}, zero={output_zero}')
print(f'  输入 shape: {input_details[0]["shape"]}')
print(f'  输出 shape: {output_details[0]["shape"]}')

# 多随机样本验证
n_test = 10
max_diff = 0.0
for i in range(n_test):
    test_input = np.random.rand(32, 40).astype(np.float32)
    
    # PyTorch 推理
    with torch.no_grad():
        pt_out = model(torch.FloatTensor(test_input).unsqueeze(0).unsqueeze(0))
        pt_probs = torch.softmax(pt_out, dim=1).numpy().squeeze()
    
    # TFLite 推理
    tflite_input = (test_input / input_scale + input_zero).astype(np.int8)
    tflite_input = tflite_input.reshape(1, 32, 40, 1)
    interpreter.set_tensor(input_details[0]['index'], tflite_input)
    interpreter.invoke()
    tflite_out = interpreter.get_tensor(output_details[0]['index']).astype(np.float32)
    tflite_out = (tflite_out - output_zero) * output_scale
    
    # softmax
    tflite_exp = np.exp(tflite_out - np.max(tflite_out))
    tflite_probs = tflite_exp / np.sum(tflite_exp)
    
    diff = np.max(np.abs(pt_probs - tflite_probs))
    max_diff = max(max_diff, diff)

print(f'  PT vs TFLite 最大概率差异: {max_diff:.6f}')
print(f'  {"✅ 验证通过" if max_diff < 0.1 else "⚠️ 差异较大，请检查量化参数"}')

# ── 6. 导出 C 数组 ──
print('\n[6] 导出 C 数组 (用于 T5 固件)...')
with open(str(TFLITE_PATH), 'rb') as f:
    model_bytes = f.read()

n_bytes = len(model_bytes)
hex_lines = []
for i in range(0, n_bytes, 16):
    chunk = model_bytes[i:i+16]
    hex_str = ', '.join(f'0x{b:02x}' for b in chunk)
    hex_lines.append(f'  {hex_str},')

c_array_name = 'g_student_model_tflite'
c_code = f"""/*
 * TinyMCUCatCNN Student TFLite int8 模型
 * 输入: 32×40 Mel (直接)
 * 输出: 10 类 (Angry/Defence/Fighting/Happy/HuntingMind/…)
 * 参数量: {n_params:,}
 * TFLite 大小: {n_bytes} bytes ({n_bytes//1024}KB)
 * 生成: {__file__}
 */

#include <stdint.h>

#define STUDENT_MODEL_TFLITE_LEN {n_bytes}

const unsigned char {c_array_name}[] = {{
{chr(10).join(hex_lines)}
}};

"""

# 去掉最后一行的尾逗号
c_code = c_code.replace(',\n};\n', '\n};\n')

with open(str(C_ARRAY_PATH), 'w', encoding='utf-8') as f:
    f.write(c_code)
print(f'  C 数组: {C_ARRAY_PATH} ({n_bytes} bytes)')

# ── 7. 保存参数 JSON ──
print('\n[7] 保存参数...')
params = {
    'model': 'TinyMCUCatCNN',
    'fold': FOLD,
    'n_params': n_params,
    'input_shape': [1, 32, 40],
    'output_classes': 10,
    'labels': LABELS,
    'tflite_size': n_bytes,
    'tflite_path': str(TFLITE_PATH),
    'c_array_path': str(C_ARRAY_PATH),
    'input_scale': input_scale,
    'input_zero_point': input_zero,
    'output_scale': output_scale,
    'output_zero_point': output_zero,
    'pt_vs_tflite_max_diff': float(max_diff),
}
with open(str(PARAMS_PATH), 'w', encoding='utf-8') as f:
    json.dump(params, f, indent=2, ensure_ascii=False)
print(f'  参数: {PARAMS_PATH}')

# ── 汇总 ──
print(f'\n{"=" * 55}')
print('  导出完成!')
print(f'  TFLite:  {TFLITE_PATH} ({n_bytes//1024}KB)')
print(f'  C 数组:  {C_ARRAY_PATH}')
print(f'  参数:    {PARAMS_PATH}')
print(f'  验证:    {"✅ PT vs TFLite 差异 < 0.1" if max_diff < 0.1 else "⚠️ 差异较大"}')
print('  ~~~~ 后续步骤 ~~~~')
print(f'  1. T5 固件: 将 {C_ARRAY_PATH.name} 加入项目')
print('  2. 预处理 32×40 Mel (与 vadexp 管线一致)')
print('  3. TFLite Micro 加载 g_student_model_tflite')
print('  4. 预期 arena: <50KB, 推理: ~50ms')
print(f'{"=" * 55}')

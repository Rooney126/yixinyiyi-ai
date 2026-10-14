"""
跨数据集诊断 — Per-frame Majority Voting
========================================
对比两种池化方式在跨数据集上的性能：

  模式 1: mean pooling (当前, 基线)   — 池化后单向量跑 SVM
  模式 2: per-frame majority vote     — 每帧跑 SVM → 多数投票

如果模式 2 显著高于模式 1，说明 mean pooling 丢信息，改 attention 池化有效。
如果差不多，说明根因是嵌入空间本身没对齐。

用法:
  python test_perframe_vote.py --dir "原猫语分类/情绪分类" --report
  python test_perframe_vote.py --dir "新猫语40/<目录>" --report
# 逐帧投票：HuBERT 每帧分类后投票合并结果
"""

import os, sys, time
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from pathlib import Path
import numpy as np
import torch
import librosa
import joblib
from collections import Counter
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import argparse

# ── 配置 ──
SR = 16000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYERS = [3, 4, 5]  # 12层扫描最优: layer 3-5

BASE = Path(r'/path/to/dataset')
# 使用 10 类 freeze 模型 (不要用 6 类合并的, 看原始可分性)
MODEL_DIR = BASE / '模型代码/训练结果/hubert_v2/freeze_model_hubert_20260618_111717'

# 10类 Pandeya 映射 (用于合并统计)
CN_LABELS = [
    'Angry', 'Defence', 'Fighting', 'Happy', 'HuntingMind',
    'Mating', 'MotherCall', 'Paining', 'Resting', 'Warning',
]
MERGE_6 = {
    'Angry':'Aggressive', 'Defence':'Aggressive',
    'Fighting':'Aggressive', 'Warning':'Aggressive',
    'Happy':'Happy', 'HuntingMind':'Happy',
    'Mating':'Mating', 'MotherCall':'MotherCall',
    'Paining':'Paining', 'Resting':'Resting',
}

# 中文文件夹名 → 6类标签 (来自 test_orig163.py)
CN_MAP = {
    '交配（低沉嘶吼）': 'Mating',
    '休息': 'Resting',
    '快乐': 'Happy',
    '愤怒（呜哇哇哇）': 'Angry',
    '打架（尖锐刺耳）': 'Fighting',
    '母体呼叫（警笛）': 'MotherCall',
    '狩猎（短促的嗷呜）': 'HuntingMind',
    '疼痛': 'Paining',
    '警告（呜~）（咕噜）': 'Warning',
    '防御(响尾蛇嘶哈)': 'Defence',
}

print(f'[加载] HuBERT + SVM (10类, freeze)')
print(f'  模型: {MODEL_DIR}')
print(f'  设备: {DEVICE}')
t0 = time.time()

# 加载 SVM 分类器
data = joblib.load(str(MODEL_DIR / 'svm.pkl'))
svm = data['svm']
scaler = data['scaler']
le = data['label_encoder']
print(f'  SVM: {len(le.classes_)} 类 → {list(le.classes_)}')

# 加载 HuBERT
model = HubertModel.from_pretrained('facebook/hubert-base-ls960', local_files_only=True)
model.to(DEVICE)
model.eval()
for p in model.parameters():
    p.requires_grad = False
processor = Wav2Vec2FeatureExtractor.from_pretrained(
    'facebook/hubert-base-ls960', local_files_only=True
)
print(f'  加载完成 ({time.time()-t0:.1f}s)')


def preprocess(y):
    """峰值归一化"""
    peak = np.max(np.abs(y))
    return y / peak if peak > 1e-10 else np.zeros_like(y)


@torch.no_grad()
def extract_frame_embeddings(audio_path):
    """
    返回帧级嵌入: (T, 768)
    T = 时间帧数 (=音频长度/20ms 约)
    不做 mean pooling, 保留时间维度
    """
    y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    y = preprocess(y)
    
    inputs = processor(
        [y], sampling_rate=SR, return_tensors="pt", padding=True
    ).input_values.to(DEVICE)
    
    outputs = model(inputs, output_hidden_states=True)
    hidden = outputs.hidden_states  # list of 13 tensors: [0]=CNN, [1..12]=transformer
    
    # 平均指定层 → 保留时间维
    layer_stack = torch.stack([hidden[i] for i in LAYERS], dim=0)  # (n_layers, 1, T, 768)
    emb = layer_stack.mean(dim=0)  # (1, T, 768)
    
    return emb.squeeze(0).cpu().numpy()  # (T, 768)


def predict_mean_pool(audio_path, merge_map=None):
    """模式 1: mean pooling + SVM (当前方法, 基线)"""
    y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    y = preprocess(y)
    
    with torch.no_grad():
        inputs = processor([y], sampling_rate=SR, return_tensors="pt", padding=True).input_values.to(DEVICE)
        outputs = model(inputs, output_hidden_states=True)
        hs = outputs.hidden_states
        emb = torch.stack([hs[i] for i in LAYERS], dim=0).mean(dim=0).mean(dim=1).cpu().numpy()  # (1, 768)
    
    emb_s = scaler.transform(emb)
    scores = svm.decision_function(emb_s)[0]
    scores -= scores.max()
    probs = np.exp(scores) / np.exp(scores).sum()
    top_idx = np.argmax(probs)
    pred = le.classes_[top_idx]
    merged = merge_map.get(pred, pred) if merge_map else pred
    return merged, float(probs[top_idx])


def predict_perframe_majority(audio_path, merge_map=None):
    """模式 2: per-frame majority voting

    1. 提取 (T, 768) 帧级嵌入
    2. 每帧跑 SVM → T 个预测
    3. 多数投票 → 最终预测
    """
    frames = extract_frame_embeddings(audio_path)  # (T, 768)
    n_frames = len(frames)
    
    if n_frames == 0:
        return None, 0.0, 0
    
    # 全量帧级 SVM 推理 (T 帧一次性)
    frames_s = scaler.transform(frames)          # (T, 768)
    frame_preds = svm.predict(frames_s)          # (T,) 每帧的预测标签
    
    # 多数投票
    vote_counter = Counter(frame_preds)
    winner_label, winner_votes = vote_counter.most_common(1)[0]
    
    # 计算合并后结果
    merged = merge_map.get(winner_label, winner_label) if merge_map else winner_label
    
    # 置信度 = 胜选帧占比
    confidence = winner_votes / n_frames
    
    # 额外统计: 胜选帧 vs 非胜选帧分布
    top3 = vote_counter.most_common(5)
    
    return merged, confidence, n_frames, top3, frame_preds


def scan_thresholds(audio_path, merge_map=None, thresholds=None):
    """模式 3: 阈值投票 — 只有 >= threshold% 的帧一致才采纳, 否则输出分布
    
    用于找出最佳投票阈值
    """
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    frames = extract_frame_embeddings(audio_path)
    n_frames = len(frames)
    if n_frames == 0:
        return {}
    
    frames_s = scaler.transform(frames)
    frame_preds = svm.predict(frames_s)
    vote_counter = Counter(frame_preds)
    total = sum(vote_counter.values())
    
    results = {}
    for t in thresholds:
        winner, votes = vote_counter.most_common(1)[0]
        ratio = votes / total
        if ratio >= t:
            label = winner
        else:
            # 低于阈值 → 输出"不确定"的最高类
            label = f"UNCERTAIN({winner})"
        
        merged = merge_map.get(winner, winner) if merge_map else winner
        if ratio < t:
            merged = f"??{merged}"
        
        results[t] = {
            'pred': winner,
            'merged': merged,
            'confidence': ratio,
            'votes': votes,
            'total': total,
        }
    
    return results


def run_directory_test(test_dir, merge_map=None):
    """
    在指定目录上跑所有 3 种模式, 输出对比
    test_dir 结构: <label_chinese>/<files.wav>
    """
    test_dir = Path(test_dir)
    if not test_dir.exists():
        print(f'❌ 目录不存在: {test_dir}')
        return
    
    print(f'\n{"="*60}')
    print(f'  测试集: {test_dir}')
    print(f'{"="*60}')
    
    total_files = 0
    mean_correct = 0
    perframe_correct = 0
    perframe_detail = {}  # label -> {mean_ok, pf_ok, mean_for_pf_ok, ...}
    
    for cn_dir in sorted(test_dir.iterdir()):
        if not cn_dir.is_dir():
            continue
        
        # 中文目录 → 期望标签
        cn_name = cn_dir.name
        expected_en = CN_MAP.get(cn_name)
        if expected_en is None:
            print(f'  [!] 跳过未映射目录: {cn_name}')
            continue
        
        expected_merged = merge_map.get(expected_en, expected_en) if merge_map else expected_en
        
        files = sorted(cn_dir.glob('*.wav')) + sorted(cn_dir.glob('*.mp3'))
        if not files:
            continue
        
        n_files = len(files)
        total_files += n_files
        
        mean_ok = 0
        pf_ok = 0
        pf_diffs = []
        file_results = []
        
        for fp in files:
            # 模式 1: mean pooling
            pred_mean, prob_mean = predict_mean_pool(fp, merge_map)
            mean_match = (pred_mean == expected_merged)
            if mean_match:
                mean_ok += 1
            
            # 模式 2: per-frame majority
            result = predict_perframe_majority(fp, merge_map)
            if result[0] is None:
                continue
            pred_pf, conf_pf, n_frames, top3, all_preds = result
            pf_match = (pred_pf == expected_merged)
            if pf_match:
                pf_ok += 1
            else:
                pf_diffs.append((fp.name, pred_pf, expected_merged, top3, n_frames))
            
            file_results.append({
                'file': fp.name,
                'mean': pred_mean,
                'mean_ok': mean_match,
                'pf': pred_pf,
                'pf_ok': pf_match,
                'pf_conf': conf_pf,
                'pf_frames': n_frames,
                'pf_top3': top3,
            })
        
        mean_acc = mean_ok / n_files
        pf_acc = pf_ok / n_files
        
        perframe_detail[expected_merged] = {
            'n': n_files,
            'mean_acc': mean_acc,
            'pf_acc': pf_acc,
            'diffs': pf_diffs[:5],  # 显示前 5 条差异
        }
        
        # 输出
        diff_emoji = '✅' if pf_acc >= mean_acc else '⚠️'
        print(f'\n  {expected_merged:<12} ({len(files)}条)')
        print(f'    均值池化 (基线): {mean_ok}/{n_files} = {mean_acc:.0%}')
        print(f'  {diff_emoji} 逐帧投票:       {pf_ok}/{n_files} = {pf_acc:.0%}')
        if pf_acc < mean_acc and pf_diffs:
            print(f'    ─ 差异文件 (per-frame 失败):')
            for name, got, exp, top3, nf in pf_diffs[:3]:
                top3_str = ', '.join([f'{k}={v}' for k, v in top3[:3]])
                print(f'      {name}: 期望={exp}, 模型={got}  (帧投票: {top3_str}, 共{nf}帧)')
    
    # 汇总
    print(f'\n  {"─"*50}')
    print(f'  总计: {total_files} 条音频')
    print(f'  均值池化: {mean_correct}/{total_files} = {mean_correct/total_files:.1%}')
    print(f'  Per-frame: {perframe_correct}/{total_files} = {perframe_correct/total_files:.1%}')
    
    # 按类对比
    print(f'\n  ── 各类对比（均值 vs Per-frame）──')
    print(f'  {"类别":<14} {"均值":>8} {"PerFrame":>10} {"差异":>8}')
    print(f'  {"─"*42}')
    for cls in sorted(perframe_detail.keys()):
        d = perframe_detail[cls]
        diff = d['pf_acc'] - d['mean_acc']
        print(f'  {cls:<14} {d["mean_acc"]:>7.0%}  {d["pf_acc"]:>7.0%}  {"+" if diff>0 else ""}{diff:>+6.0%}')
    
    return perframe_detail


def run_single_file_test(audio_path, merge_map=None):
    """单条音频详细诊断"""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f'❌ 文件不存在: {audio_path}')
        return
    
    print(f'\n{"="*60}')
    print(f'  详细诊断: {audio_path.name}')
    print(f'{"="*60}')
    
    # 模式 1
    pred_mean, prob_mean = predict_mean_pool(audio_path, merge_map)
    print(f'\n  均值池化: {pred_mean} (置信度 {prob_mean:.1%})')
    
    # 模式 2
    result = predict_perframe_majority(audio_path, merge_map)
    if result[0]:
        pred_pf, conf_pf, n_frames, top3, all_preds = result
        print(f'  Per-frame 投票 ({n_frames} 帧):')
        print(f'    胜选: {pred_pf} ({conf_pf:.1%} 帧支持)')
        print(f'    帧分布:')
        for label, count in top3:
            bar = '█' * int(count / n_frames * 40)
            print(f'      {label:<14} {count:>4}/{n_frames} ({count/n_frames:>5.1%}) {bar}')
        
        # 帧级标签序列（可视化前 60 帧）
        first60 = all_preds[:60]
        print(f'\n    前 {min(60, n_frames)} 帧标签序列:')
        print(f'    {"".join(l[0] for l in first60)}')
        
        # 检查帧标签的稳定性（每秒的标签分布）
        frames_per_second = 50  # HuBERT ~50Hz frame rate
        if n_frames >= frames_per_second:
            print(f'\n    逐秒标签分布:')
            for sec in range(min(5, n_frames // frames_per_second)):
                start = sec * frames_per_second
                end = min((sec + 1) * frames_per_second, n_frames)
                sec_preds = all_preds[start:end]
                sec_counter = Counter(sec_preds)
                sec_total = len(sec_preds)
                
                dominant = sec_counter.most_common(1)[0]
                sec_str = ', '.join([f'{k}({v/sec_total:.0%})' for k, v in sec_counter.most_common(3)])
                print(f'      第{sec+1}秒 ({start}-{end}帧): 主流={dominant[0]}({dominant[1]/sec_total:.0%})  [{sec_str}]')
    
    # 阈值扫描
    print(f'\n  阈值扫描:')
    thresh_results = scan_thresholds(audio_path, merge_map)
    for t, r in sorted(thresh_results.items()):
        print(f'    阈值 >= {t:.0%} → {r["merged"]} (支持 {r["confidence"]:.0%})')
    
    # 帧嵌入的统计特征
    frames = extract_frame_embeddings(audio_path)
    print(f'\n  帧嵌入统计 (T={len(frames)}, D=768):')
    print(f'    帧间余弦相似度均值: {np.mean([np.dot(frames[i], frames[i+1])/(np.linalg.norm(frames[i])*np.linalg.norm(frames[i+1])+1e-10) for i in range(min(100, len(frames)-1))]):.3f}')
    print(f'    帧嵌入标准差均值:    {np.mean(np.std(frames, axis=0)):.4f}')
    print(f'    帧嵌入标准差分布:    min={np.min(np.std(frames, axis=0)):.4f}, max={np.max(np.std(frames, axis=0)):.4f}')


def main():
    parser = argparse.ArgumentParser(description='Per-frame majority voting 诊断')
    parser.add_argument('--dir', type=str, help='批量测试目录 (<类别中文名>/<音频>)')
    parser.add_argument('--audio', type=str, help='单条音频详细诊断')
    parser.add_argument('--report', action='store_true', help='显示汇总报告')
    args = parser.parse_args()
    
    if not args.dir and not args.audio:
        parser.print_help()
        print('\n❌ 指定 --dir (批量) 或 --audio (单条)')
        return
    
    merge_map = MERGE_6  # 使用 6 类合并输出
    
    if args.dir:
        run_directory_test(args.dir, merge_map)
    
    if args.audio:
        run_single_file_test(args.audio, merge_map)


if __name__ == '__main__':
    main()

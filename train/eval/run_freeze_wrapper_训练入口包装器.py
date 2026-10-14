# 训练入口的简化包装器（命令行快捷调用）
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
log = open(r'/path/to/dataset\模型代码\训练结果\freeze_output.txt', 'w', buffering=1, encoding='utf-8')
sys.stdout = log
sys.stderr = log
with open(r'/path/to/dataset\模型代码\训练入口hubert_train.py', 'r', encoding='utf-8') as f:
    code = f.read()
exec(code)
log.close()

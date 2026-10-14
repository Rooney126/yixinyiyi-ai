import librosa
import torch

print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'device={torch.cuda.get_device_name(0)}')
print(f'librosa={librosa.__version__}')
print('All packages OK')

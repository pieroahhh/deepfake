"""Fail before training when the selected host/container cannot execute CUDA."""
import json
import torch

if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; check the pinned environment, NVIDIA driver and Docker/WSL2 GPU access')
    result = (torch.ones(8, device='cuda') * 2).sum().item()
    if result != 16:
        raise RuntimeError('CUDA arithmetic check failed')
    print(json.dumps({'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                      'device': torch.cuda.get_device_name(0),
                      'vram_gib': round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
                      'arithmetic_check': 'PASSED'}))

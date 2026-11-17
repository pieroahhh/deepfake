# Local machine inventory

Observed on 11 November 2026. This describes the current guest environment, not the physical host or the four planned experiment machines.

| Item | Observed value |
| --- | --- |
| OS | Kali GNU/Linux Rolling 2026.2, x86_64 |
| Virtualization | VMware virtual machine |
| Reported CPU model | AMD Ryzen 5 5600X 6-Core Processor |
| CPUs available to guest | 4 logical CPUs |
| Guest RAM | Approximately 1.9 GiB total; 936 MiB available at measurement |
| Guest swap | 953 MiB total; 692 MiB used at measurement |
| Project filesystem | 79 GiB total; 54 GiB available at measurement |
| /tmp | RAM-backed tmpfs, approximately 967 MiB capacity |
| Display adapter | VMware SVGA II Adapter |
| NVIDIA/CUDA exposure | No NVIDIA device listed by lspci; /dev/nvidia0 absent; nvidia-smi and nvcc unavailable |
| Python | 3.13.12 |
| pip | 26.1.1 |
| Git | Executable available |
| FFmpeg / FFprobe | Not found on PATH |
| uv / conda | Not found on PATH |

Installed package metadata: NumPy 2.3.5, SciPy 1.16.3, pandas 2.3.3, pypdf 6.9.2. No installed distribution metadata found for torch, torchvision, torchaudio, transformers, librosa, opencv-python, or scikit-learn. These were metadata checks, not import/runtime validation.

## Readiness

This guest can support document, metadata and lightweight code work. It does not currently expose the GPU or memory resources specified for the training machines in the sprint plan. Physical-host GPU, RAM and storage have not been inspected. Training feasibility has not been established.

Next: identify an accessible GPU training environment and inventory it; audit dataset access and metadata before selecting pilot media. Resolve per-method Python and dependency versions before installing model environments. No packages or datasets were installed/downloaded during this inventory.

## Source fingerprint discrepancy

Actual SHA-256 values:

```text
64d4dd72589d00e4f68b633bcbce91a7dd1cf90278c8c00fcec056809ee8d83f  DeepFakeDetection_AjOhmRev.pdf
b761456da20989dcad126af3b6be7a5127085b108a2978ea57bbccc4ff4eb0e8  AVFakeNet.pdf
be7eda054e2824aaf86c291e575a0ccc3c14c7f6d3445f6ab84a476a1381f197  AVTENet.pdf
```

The implementation plan appendix lists manuscript hash `a3d1b85c3fcbc5eb275861dcf3334c52c398bbc3527b939063f1b958569509ca`, which differs from the current manuscript. Both baseline PDF hashes match the appendix. The cause and content differences are unknown; the differently hashed manuscript copy is not present for comparison. Do not claim the current manuscript is the exact revision fingerprinted in that appendix.

## Inspection commands

`uname -a`, `lscpu`, `free -h`, `df -h . /tmp`, `nvidia-smi`, `lspci`, device-path checks, `/etc/os-release`, Python/pip versions, executable discovery, Python importlib.metadata distribution checks, and `sha256sum *.pdf`.

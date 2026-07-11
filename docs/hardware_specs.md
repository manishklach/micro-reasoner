# Hardware Specifications

## System Overview

| Component | Detail |
|-----------|--------|
| CPU | AMD Ryzen 3 5300U (4 cores, 8 threads) |
| GPU | AMD Radeon Graphics (integrated) |
| GPU VRAM | ~512 MB dedicated, shared system memory |
| System RAM | 16 GB |
| Disk Free | ~65 GB (Windows C:), ~868 GB (WSL virtual disk) |
| CUDA | Not available |
| OS | Windows 11 (Build 26100) |

## Training Environment

- **Environment**: WSL2 + Ubuntu 24.04 LTS
- **Python**: 3.12.3
- **Torch**: 2.13.0+cpu (CPU-only)
- **CPU Threads**: 4

## Limitation

No NVIDIA GPU is present. All training runs on CPU.
This constrains us to small models (1B-3B parameters) with QLoRA.
Recommended approach: CPU-only training with efficient CPU inference via llama.cpp.

## Installed Packages

See requirements.txt for full freeze.

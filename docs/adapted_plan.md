# CPU-Only Adaptation of Single-GPU Reasoning Model Plan

## Key Difference

The original plan assumed an NVIDIA GPU with CUDA. This system has:
- **AMD Ryzen 3 5300U** (no NVIDIA GPU)
- **16 GB RAM** (7.4 GB available to WSL)
- **CPU-only PyTorch** training

## Adapted Strategy

### Model Size
We must target **1B to 3B parameter models** (not 7B/8B as the original plan suggested for Tier 3).

### Training
- **QLoRA on CPU** — slower but functional for small models
- Use itsandbytes 4-bit quantization during training to reduce memory
- Small batch sizes, gradient accumulation
- Expect 10-50x slower than GPU training

### Inference
- Use 	ransformers with CPU optimizations
- Optionally use llama.cpp after quantizing to GGUF for faster CPU inference

### What Changes
- No Triton kernels (CUDA-only)
- No flash-attn (CUDA-only)
- No vLLM (CUDA-only)
- Focus on smaller, more efficient models
- Expect longer training times

## Phases Modified

| Phase | Modification |
|-------|-------------|
| 1-2 | Same, environment validated |
| 3-5 | Same, but model size capped at 3B |
| 6 | Profile CPU bottlenecks instead |
| 7 | CPU-specific optimizations (thread tuning, AVX, etc.) |
| 8 | Skip — Triton requires CUDA |
| 9-10 | DPO still possible on CPU with small models |
| 11 | GRPO likely impractical on CPU |
| 12 | Quantization to GGUF for CPU inference |
| 13-14 | Same, ship quantized model |

## Hardware Tiers (Adapted)

| Tier | Parameters | Approach |
|------|-----------|----------|
| 1B class | 1-2B | Comfortable, reasonable training speed |
| 3B class | 2.7-3.8B | Tight but feasible with QLoRA |
| 7B+ | 7B+ | Impractical for training (could infer only) |

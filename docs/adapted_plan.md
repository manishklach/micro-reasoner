# CPU-Only Adaptation of Single-GPU Reasoning Model Plan

## Key Difference
The original plan assumed an NVIDIA GPU with CUDA. This system has:
- **AMD Ryzen 3 5300U** (no NVIDIA GPU)
- **16 GB RAM** (7.4 GB available to WSL)
- **CPU-only PyTorch** training

## Current Track
**SmolLM2-360M-Instruct** — a CPU-first reasoning-model lab.
- Target: 360M parameters with LoRA adapters
- GSM8K math reasoning dataset
- SFT → DPO → eval → deploy pipeline

## Adapted Strategy

### Model Size
Current: **360M** (SmolLM2). Stretch: **1.5B** (Qwen2.5). Beyond 3B is not feasible for CPU training.

### Training
- **LoRA on CPU** — verified forward/backward at ~10s/step
- Batch size 1, gradient accumulation 8
- Full epoch ~18h on CPU
- No bitsandbytes quantization currently (float32 fits with 360M)

### Inference
- transformers with greedy decoding — ~6.6 tok/s on CPU
- GGUF/llama.cpp export possible for faster CPU inference

### What Changes from Original Plan
- No Triton kernels (CUDA-only)
- No flash-attn (CUDA-only)
- No vLLM (CUDA-only)
- No GRPO (impractical on CPU — DPO is the ceiling)
- Focus on small models, quick iteration

## Phases Modified

| Phase | Modification |
|-------|-------------|
| Environment | Validated WSL2 + CPU-only PyTorch |
| Data | GSM8K only (not OpenThoughts) |
| SFT | LoRA on CPU, small batch sizes |
| DPO | Synthetic preference pairs (placeholder) |
| GRPO | Skipped — impractical on CPU |
| GGUF | Manual guidance only; adapter merge not automated |

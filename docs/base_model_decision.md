# Base Model Decision

## Hardware Constraint
- CPU-only (no NVIDIA GPU)
- 7.4 GB RAM available in WSL
- Target: 360M-1.5B parameter models with LoRA

## Tiers

| Tier | Model | Params | Inference | Training | Status |
|------|-------|--------|-----------|----------|--------|
| Ship (current) | SmolLM2-360M-Instruct | 360M | 6.62 tok/s | ~22 min/step | ✅ Active |
| Stretch | Qwen2.5-1.5B-Instruct | 1.5B | 3.9 tok/s | ~40+ min/step | ⚠️ Feasible but slow |

## Chosen Model (Current Track)
**SmolLM2-360M-Instruct**
- Fits 7.4 GB RAM comfortably in float32 with LoRA
- Fast training iteration (~22 min/step on CPU)
- Full pipeline (SFT → DPO → eval → serve) verified end-to-end
- Strong baseline: 6.62 tok/s inference on CPU

## Stretch Target: Qwen2.5-1.5B-Instruct
- **1543.7M params** — fits in RAM with LoRA (verified: 0.67 GB RSS, 6.0 GB available)
- **Inference**: 3.3-4.0 tok/s on CPU (greedy, 128 new tokens)
- **Training**: Verified feasible but ~2-3x slower than SmolLM2 per step
- Training practical for 5-10 step demos; full SFT impractical on CPU
- Better reasoning capability than SmolLM2
- Permissive license (Apache 2.0)
- Chat template: `<|im_start|>user/assistant` format

## Rejected
- **Models >3B**: Too large for comfortable CPU training even with QLoRA
- TinyLlama-1.1B: Weaker reasoning than Qwen at similar size

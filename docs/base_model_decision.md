# Base Model Decision

## Hardware Constraint
- CPU-only (no NVIDIA GPU)
- 7.4 GB RAM available in WSL
- Target: 360M-1.5B parameter models with LoRA

## Tiers

| Tier | Model | Params | Inference | Status |
|------|-------|--------|-----------|--------|
| Ship (current) | SmolLM2-360M-Instruct | 360M | 6.62 tok/s | ✅ Active |
| Stretch | Qwen2.5-1.5B-Instruct | 1.5B | 3.9 tok/s | ✅ Benchmark complete |

## Chosen Model (Current Track)
**SmolLM2-360M-Instruct**
- Fits 7.4 GB RAM comfortably in float32 with LoRA
- Fast training iteration (~22 min/step on CPU)
- Full pipeline (SFT → DPO → eval → serve) verified end-to-end
- Strong baseline: 6.62 tok/s inference on CPU

## Stretch Target: Qwen2.5-1.5B-Instruct
- **1543.7M params** — fits in RAM with ~2GB overhead
- **Inference**: 3.3-4.0 tok/s on CPU (greedy, 128 new tokens)
- **First-token latency**: ~10s for short prompts
- Better reasoning capability than SmolLM2
- Permissive license (Apache 2.0)
- Chat template: `<|im_start|>user/assistant` format
- **Next step**: Verify training feasibility (RAM during LoRA SFT)

## Rejected
- **Models >3B**: Too large for comfortable CPU training even with QLoRA
- TinyLlama-1.1B: Weaker reasoning than Qwen at similar size

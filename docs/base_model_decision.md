# Base Model Decision

## Hardware Constraint
- CPU-only (no NVIDIA GPU)
- 7.4 GB RAM available in WSL
- Target: 360M-1.5B parameter models with LoRA

## Tiers

| Tier | Model | Params | Status | Notes |
|------|-------|--------|--------|-------|
| Current (ship) | SmolLM2-360M-Instruct | 360M | ✅ Active | Fits comfortably on CPU, end-to-end verified |
| Stretch | Qwen2.5-1.5B-Instruct | 1.5B | ⏳ Planned | Next upgrade if CPU training remains tolerable |

## Chosen Model (Current Track)
**SmolLM2-360M-Instruct**
- Fits 7.4 GB RAM comfortably in float32 with LoRA
- Fast training iteration (~18h/epoch on CPU)
- Full pipeline (SFT → DPO → eval → serve) verified end-to-end
- Strong baseline: 6.62 tok/s inference on CPU

## Stretch Target
**Qwen2.5-1.5B-Instruct**
- Better reasoning capability
- ~4x larger → slower but more capable
- Requires QLoRA (4-bit) to fit in RAM
- Permissive license (Apache 2.0)

## Rejected
- **Models >3B**: Too large for comfortable CPU training even with QLoRA
- **Llama-3.2-3B**: Requires too much RAM for CPU training

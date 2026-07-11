# Base Model Decision

## Hardware Constraint
- CPU-only (no NVIDIA GPU)
- 7.4 GB RAM available in WSL
- Target: 1B-3B parameter models with QLoRA

## Candidates

| Model | Params | Reasoning | VRAM Fit | Ecosystem | Quant | License | Score |
|-------|--------|-----------|----------|-----------|-------|---------|-------|
| Qwen2.5-1.5B-Instruct | 1.5B | 4 | 5 | 5 | 5 | 4 | 28 |
| Qwen2.5-3B-Instruct | 3B | 4 | 4 | 5 | 5 | 4 | 27 |
| Llama-3.2-3B-Instruct | 3B | 4 | 4 | 5 | 5 | 3 | 26 |
| Phi-3-mini-4k-instruct | 3.8B | 5 | 3 | 4 | 4 | 4 | 24 |
| Gemma-2-2B-it | 2B | 3 | 5 | 4 | 4 | 3 | 22 |

Scored 1-5 per column.

## Chosen Model
**Qwen2.5-1.5B-Instruct**
- Best balance of reasoning capability and resource fit for CPU-only training
- Strong ecosystem support (HuggingFace, PEFT, TRL all well-tested)
- Excellent quantization support
- Permissive license (Apache 2.0)
- Fast inference on CPU

## Runner-Up
**Qwen2.5-3B-Instruct**
- More capable but tighter on RAM during training
- Good fallback if 1.5B proves too weak

## Fallback
**Llama-3.2-3B-Instruct**
- If Qwen doesn't work well, this is the next best option
- Slightly weaker reasoning but better instruction following

## Rejected
- **Phi-3-mini (3.8B)**: Too large for comfortable CPU training with QLoRA
- **Gemma-2-2B**: Weaker reasoning capability compared to Qwen at similar size
- **Models >4B**: Not feasible for training on this hardware

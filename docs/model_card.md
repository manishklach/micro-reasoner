# Model Card: SmolLM2-360M-Reasoner

## Model Details
- **Base Model**: HuggingFaceTB/SmolLM2-360M-Instruct
- **Architecture**: 360M parameter transformer (LoRA adapters)
- **License**: Apache 2.0 (base model)
- **Hardware**: AMD Ryzen 3 5300U (CPU), 16GB RAM

## Training
| Phase | Method | Data | Epochs |
|-------|--------|------|--------|
| SFT   | LoRA (r=8) | GSM8K (6.7K math problems) | 1 |
| DPO   | LoRA (r=8) | UltraFeedback subset (100 pairs) | 1 |

## Performance
| Metric | Baseline | SFT | DPO |
|--------|----------|-----|-----|
| Tokens/sec | 6.62 | TBD | TBD |
| Avg latency | 23.6s | TBD | TBD |

## Usage
```bash
# Interactive
python scripts/serve.py --adapter outputs/sft/final_adapter

# Single prompt
python scripts/serve.py --adapter outputs/sft/final_adapter --prompt "What is 15 * 7?"
```

## Prompt Format
```
<|user|>
{prompt}
<|assistant|>
{response}</s>
```

## Limitations
- CPU-only inference (6-7 tok/s)
- 360M parameter capacity limits complex reasoning
- Trained on GSM8K math only - narrow domain

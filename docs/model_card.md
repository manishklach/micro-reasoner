# Model Card: SmolLM2-360M-Reasoner

## Model Details
- **Base Model**: HuggingFaceTB/SmolLM2-360M-Instruct
- **Architecture**: 360M parameter transformer (LoRA adapters)
- **License**: Apache 2.0 (base model)
- **Hardware**: AMD Ryzen 3 5300U (CPU), 16GB RAM

## Training
| Phase | Method | Data | Epochs | Status |
|-------|--------|------|--------|--------|
| SFT   | LoRA (r=8) | GSM8K (6.7K math problems) | 1 | Partial run: 7/50 steps completed |
| DPO   | LoRA (r=8) | Synthetic pairs (100 examples) | 1 | Placeholder — real data needed |

## SFT Loss Curve (Partial)
| Step | Loss | Grad Norm | LR |
|------|------|-----------|----|
| 1    | 1.338 | 0.142 | 1e-05 |
| 2    | 1.584 | 0.198 | 2e-05 |
| 3    | 1.686 | 0.229 | 3e-05 |
| 4    | 1.383 | 0.153 | 4e-05 |
| 5    | 1.359 | 0.164 | 5e-05 |
| 6    | 1.380 | 0.178 | 6e-05 |

Loss trending downward after initial spike. Averaging ~22 min/step (gradient accumulation 8).

## Performance
| Metric | Baseline |
|--------|----------|
| Tokens/sec | 6.62 |
| Avg latency | 23.6s |

## Usage
```bash
# Interactive (base model only — no adapter weights saved from partial run)
python scripts/chat_local.py

# Evaluate baseline
python scripts/evaluate.py --max-prompts 20
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
- Trained on GSM8K math only — narrow domain
- DPO currently uses synthetic preference pairs
- SFT run was partial (7/50 steps); adapter weights not saved

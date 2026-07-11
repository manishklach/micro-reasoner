# Data

| Directory | Contents |
|-----------|----------|
| `processed/` | Parquet files for SFT training/eval |
| `eval/` | JSON eval prompts with reference answers |

## Current Datasets
- **GSM8K** — 6,725 train / 748 eval / 100 test prompts for math reasoning
- **UltraFeedback (DPO)** — 500 preference pairs from HuggingFaceH4/ultrafeedback_binarized

## Generating
```bash
# SFT data
python scripts/prepare_data.py

# DPO preference data
python scripts/prepare_dpo_data.py --dataset ultrafeedback --max-samples 500
```

# Data

| Directory | Contents |
|-----------|----------|
| `processed/` | Parquet files for SFT training/eval |
| `eval/` | JSON eval prompts with reference answers |

## Current Datasets
- **GSM8K** — 6,725 train / 748 eval / 100 test prompts for math reasoning

## Generating
```bash
python scripts/prepare_data.py
```

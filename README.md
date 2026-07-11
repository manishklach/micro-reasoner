# micro-reasoner

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-orange)](https://pytorch.org/)
[![Platform](https://img.shields.io/badge/Platform-CPU%20(WSL2)-lightgrey)](https://learn.microsoft.com/en-us/windows/wsl/)
[![Model](https://img.shields.io/badge/Model-SmolLM2--360M-green)](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
[![Last Commit](https://img.shields.io/github/last-commit/manishklach/micro-reasoner)](https://github.com/manishklach/micro-reasoner)

**CPU-first reasoning-model lab** — train a small instruct model into a measurable reasoning assistant using SFT, lightweight preference tuning, and local deployment.

**Current best-supported model**: `SmolLM2-360M-Instruct` (360M params, float32)

## Hardware Constraints

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 3 5300U (4 cores, 8 threads) |
| RAM | 16 GB (7.4 GB available to WSL) |
| GPU | None (AMD Radeon integrated — no CUDA) |
| Training | CPU-only PyTorch, LoRA adapters |

## Quickstart

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download + prepare GSM8K data
python scripts/prepare_data.py

# 3. Baseline benchmark
python scripts/benchmark_base.py --max-prompts 20

# 4. SFT training (20 steps demo; full run: --max-steps 500)
python scripts/sft_train.py --max-steps 20

# 5. Evaluate SFT
python scripts/evaluate.py --adapter outputs/sft/final_adapter

# 6. DPO tuning (synthetic pairs — replace for production)
python scripts/dpo_train.py

# 7. Interactive chat
python scripts/chat_local.py --adapter outputs/sft/final_adapter
```

For long training runs: `nohup python scripts/sft_train.py > outputs/sft/train.log 2>&1 &`

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Baseline benchmark | ✅ Complete | 6.62 tok/s, 23.6s avg latency |
| SFT training | ⚠️ Partial | 7/50 steps completed, loss 1.34→1.38, ~22 min/step |
| DPO training | ⚠️ Synthetic data | Uses truncated correct answers — real preference pairs needed |
| Evaluation | ✅ Script ready | Regex-based answer extraction, accuracy + speed metrics |
| GGUF export | 📋 Guide only | Manual llama.cpp conversion; adapter merge not automated |
| Qwen2.5-1.5B track | ⏳ Planned | Next upgrade candidate if CPU training remains tolerable |

## Project Structure

```
micro-reasoner/
├── scripts/           # All training, eval, inference scripts
│   ├── config.py      # Centralized paths and defaults
│   ├── prepare_data.py
│   ├── sft_train.py
│   ├── dpo_train.py
│   ├── chat_local.py
│   ├── benchmark_base.py
│   ├── evaluate.py
│   ├── benchmark_compare.py
│   └── gguf_instructions.py
├── docs/              # Design docs and model card
├── data/              # Datasets (GSM8K; processed Parquet files)
├── models/            # Base model weights (gitignored)
├── outputs/           # Training outputs and benchmarks
└── requirements.txt
```

## What's Real vs Planned

**Real today:**
- GSM8K math dataset with SFT formatting
- LoRA SFT training (verified forward/backward on CPU, partial run completed)
- Baseline benchmark with token/sec and latency metrics
- Eval script with answer extraction and accuracy scoring
- Chat interface with optional adapter loading
- Comparison table across model stages

**Planned next:**
- Full SFT training run (~18h on CPU, or overnight nohup)
- Real preference data (UltraFeedback subset or manually filtered pairs)
- DPO with real chosen/rejected pairs
- Qwen2.5-1.5B-Instruct track
- GGUF export with automatic adapter merging

## License

Apache 2.0

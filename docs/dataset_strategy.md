# Dataset Strategy

## Current: GSM8K (Math Reasoning)
- **Source**: `openai/gsm8k` — 8.5K grade-school math word problems with step-by-step solutions
- **SFT split**: 6,725 train / 748 eval (90/10 split)
- **Eval**: 100 held-out GSM8K test prompts with reference answers
- **Format**: `instruction` (question) + `response` (answer with chain-of-thought)
- **Storage**: Parquet in `data/processed/`

## Planned
- **Preference data**: Real DPO pairs from UltraFeedback or manually filtered self-generated pairs
- **Additional domains**: Code, instruction-following, safety/refusal

## Data Processing
- Loaded via HuggingFace `datasets`
- Formatted with SmolLM2 chat template (`<|user|>`, `<|assistant|>`)
- Tokenized to max 512 tokens
- Saved as Parquet for fast loading

# Dataset Strategy

## SFT Data: GSM8K + OpenThoughts (reasoning subset)
- **GSM8K**: 8.5K math word problems with step-by-step solutions
- **OpenThoughts**: ~10K high-quality reasoning traces
- Format: instruction + chain-of-thought + answer
- Train/val split: 90/10

## Preference Data: UltraFeedback (subset)
- 10K preference pairs with chosen/rejected responses
- Focus on reasoning and instruction-following categories
- Fields: prompt, chosen, rejected, source

## Evaluation Set: 100 held-out prompts
- 40 math/reasoning
- 20 instruction following
- 20 code/debugging
- 10 safety/refusal
- 10 long-form structured

## Data Processing
- All datasets loaded via HuggingFace datasets
- Preprocessing: tokenize, format with chat template, filter length
- Saved as Parquet in data/processed/

import os, argparse
from pathlib import Path
os.environ['HF_HUB_DISABLE_HF_TRANSFER'] = '1'
from datasets import load_dataset
import json
from config import PROJECT_ROOT

def main():
    parser = argparse.ArgumentParser(description="Download and prepare training/eval data")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data"), help="Output data directory")
    parser.add_argument("--max-eval", type=int, default=100, help="Number of eval prompts to extract")
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k"], help="Dataset to prepare")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    processed_dir = output_dir / "processed"
    eval_dir = output_dir / "eval"
    processed_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset}...")
    gsm8k = load_dataset('openai/gsm8k', 'main', split='train')
    print(f"GSM8K train: {len(gsm8k)} examples")

    def format_gsm8k(example):
        return {
            'instruction': example['question'],
            'response': example['answer'],
            'source': 'gsm8k'
        }

    sft_data = gsm8k.map(format_gsm8k, remove_columns=gsm8k.column_names)
    split = sft_data.train_test_split(test_size=0.1, seed=42)
    train_data = split['train']
    eval_data = split['test']

    print(f"SFT train: {len(train_data)}, eval: {len(eval_data)}")

    train_data.to_parquet(str(processed_dir / "sft_train.parquet"))
    eval_data.to_parquet(str(processed_dir / "sft_eval.parquet"))

    print("Loading GSM8K test set for evaluation...")
    gsm8k_test = load_dataset('openai/gsm8k', 'main', split='test')
    eval_prompts = gsm8k_test.select(range(min(args.max_eval, len(gsm8k_test))))

    eval_formatted = []
    for ex in eval_prompts:
        eval_formatted.append({
            'prompt': ex['question'],
            'reference': ex['answer'],
            'category': 'math/reasoning'
        })

    with open(str(eval_dir / "eval_prompts.json"), 'w') as f:
        json.dump(eval_formatted, f, indent=2)

    print(f"Evaluation set: {len(eval_formatted)} prompts saved")
    print("Data preparation complete!")

if __name__ == "__main__":
    main()

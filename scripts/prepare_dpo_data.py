"""Download and prepare real preference data for DPO training.
Uses HuggingFaceH4/ultrafeedback_binarized (already formatted as chosen/rejected pairs).

Usage:
  python scripts/prepare_dpo_data.py --max-samples 500
  python scripts/prepare_dpo_data.py --dataset orca_dpo_pairs --max-samples 300
"""
import os, argparse
from pathlib import Path
from datasets import load_dataset
from config import PROJECT_ROOT

def format_smollm2(example):
    prompt = example.get("prompt") or example.get("question", "")
    chosen = example.get("chosen") or example.get("chosen", "")
    rejected = example.get("rejected") or example.get("rejected", "")

    # Format with SmolLM2 chat template
    formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
    formatted_chosen = f"{chosen}</s>"
    formatted_rejected = f"{rejected}</s>"

    return {
        "prompt": formatted_prompt,
        "chosen": formatted_chosen,
        "rejected": formatted_rejected,
        "source": args.dataset,
    }

def main():
    parser = argparse.ArgumentParser(description="Prepare DPO preference data")
    parser.add_argument("--dataset", default="ultrafeedback",
                        choices=["ultrafeedback", "orca_dpo_pairs"],
                        help="Preference dataset source")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Number of preference pairs to extract")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "processed"),
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "ultrafeedback":
        print("Loading HuggingFaceH4/ultrafeedback_binarized (train_prefs split)...")
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
        ds = ds.select(range(min(args.max_samples, len(ds))))
        source = "ultrafeedback"

    elif args.dataset == "orca_dpo_pairs":
        print("Loading Intel/orca_dpo_pairs...")
        ds = load_dataset("Intel/orca_dpo_pairs", split="train")
        ds = ds.select(range(min(args.max_samples, len(ds))))
        source = "orca_dpo_pairs"

    print(f"Loaded {len(ds)} examples from {source}")

    def format_row(example):
        prompt = example["prompt"]
        chosen = example["chosen"]
        rejected = example["rejected"]
        return {
            "prompt": f"<|user|>\n{prompt}\n<|assistant|>\n",
            "chosen": f"{chosen}</s>",
            "rejected": f"{rejected}</s>",
            "source": source,
        }

    formatted = ds.map(format_row, remove_columns=ds.column_names)

    out_path = output_dir / f"dpo_pairs_{source}.parquet"
    formatted.to_parquet(str(out_path))

    # Also save a smaller sample for quick testing
    sample = formatted.select(range(min(100, len(formatted))))
    sample_path = output_dir / f"dpo_pairs_{source}_sample.parquet"
    sample.to_parquet(str(sample_path))

    print(f"Saved {len(formatted)} pairs to {out_path}")
    print(f"Saved {len(sample)} sample pairs to {sample_path}")
    print("Schema: prompt (str), chosen (str), rejected (str), source (str)")

if __name__ == "__main__":
    main()

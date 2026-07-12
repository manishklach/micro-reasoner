"""DPO preference tuning with real preference data.
Downloads UltraFeedback or Orca DPO pairs if not cached.

Usage:
  python scripts/dpo_train.py --max-steps 20
  python scripts/dpo_train.py --dataset orca_dpo_pairs --max-samples 300 --max-steps 30
"""
import json, time, os, argparse, sys
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from trl import DPOTrainer, DPOConfig
from config import PROJECT_ROOT, DEFAULT_MODEL, DEFAULT_SFT_OUTPUT, DEFAULT_DPO_OUTPUT, get_chat_template

def main():
    parser = argparse.ArgumentParser(description="DPO preference tuning")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=None, help="SFT adapter path (optional)")
    parser.add_argument("--output-dir", default=DEFAULT_DPO_OUTPUT)
    parser.add_argument("--dataset", default="ultrafeedback",
                        choices=["ultrafeedback", "orca_dpo_pairs", "synthetic"],
                        help="Preference data source")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model from {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(args.model, low_cpu_mem_usage=True)

    if args.adapter:
        adapter_path = Path(args.adapter)
        if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(model, str(adapter_path))
            print(f"Loaded SFT adapter from {args.adapter}")

    lora_cfg = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    tmpl = get_chat_template(args.model)

    if args.dataset == "synthetic":
        print("WARNING: Using synthetic preference data (truncated correct answers).")
        print("Use --dataset ultrafeedback or orca_dpo_pairs for real data.")
        from datasets import load_dataset as ld
        raw = ld("parquet", data_files={"train": str(PROJECT_ROOT / "data/processed/sft_train.parquet")}, split="train")
        raw = raw.select(range(min(args.max_samples, len(raw))))

        def build_preference(example):
            prompt = f"{tmpl['user_prefix']}{example['instruction']}{tmpl['user_suffix']}"
            resp = example['response']
            chosen = resp
            rejected = resp[:len(resp)//2] + "[incomplete]"
            return {"prompt": prompt, "chosen": chosen + tmpl['assistant_suffix'], "rejected": rejected + tmpl['assistant_suffix']}

        dpo_data = raw.map(build_preference, remove_columns=raw.column_names)
    else:
        cache_path = PROJECT_ROOT / "data" / "processed" / f"dpo_pairs_{args.dataset}.parquet"
        if cache_path.exists():
            print(f"Loading cached DPO data from {cache_path}...")
            dpo_data = load_dataset("parquet", data_files={"train": str(cache_path)}, split="train")
        else:
            print(f"Downloading {args.dataset} preference data...")
            if args.dataset == "ultrafeedback":
                ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
            elif args.dataset == "orca_dpo_pairs":
                ds = load_dataset("Intel/orca_dpo_pairs", split="train")
            ds = ds.select(range(min(args.max_samples, len(ds))))

            def format_row(example):
                prompt = example["prompt"]
                chosen = example["chosen"]
                rejected = example["rejected"]
                return {
                    "prompt": f"{tmpl['user_prefix']}{prompt}{tmpl['user_suffix']}",
                    "chosen": f"{chosen}{tmpl['assistant_suffix']}",
                    "rejected": f"{rejected}{tmpl['assistant_suffix']}",
                }
            dpo_data = ds.map(format_row, remove_columns=ds.column_names)

        print(f"Loaded {len(dpo_data)} preference pairs from {args.dataset}")

    training_args = DPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        max_length=512,
        num_train_epochs=1,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="no",
        learning_rate=args.lr,
        report_to=[],
        beta=args.beta,
        use_cpu=True,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dpo_data,
        processing_class=tokenizer,
    )

    print(f"Starting DPO training with {args.dataset} data ({args.max_steps} steps)...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"DPO done in {elapsed:.1f}s")

    model.save_pretrained(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))

    summary = {
        "model": Path(args.model).name,
        "dataset": args.dataset,
        "num_pairs": len(dpo_data),
        "train_time_sec": round(elapsed, 1),
        "max_steps": args.max_steps,
    }
    with open(str(output_dir / "dpo_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {json.dumps(summary, indent=2)}")

if __name__ == "__main__":
    main()

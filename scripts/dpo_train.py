import json, time, os, argparse
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from trl import DPOTrainer, DPOConfig
from config import PROJECT_ROOT, DEFAULT_MODEL, DEFAULT_SFT_OUTPUT, DEFAULT_DPO_OUTPUT, DEFAULT_SFT_DATA

def main():
    parser = argparse.ArgumentParser(description="DPO preference tuning")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model path")
    parser.add_argument("--adapter", default=DEFAULT_SFT_OUTPUT + "/final_adapter", help="SFT adapter path")
    parser.add_argument("--output-dir", default=DEFAULT_DPO_OUTPUT, help="Output directory")
    parser.add_argument("--train-data", default=DEFAULT_SFT_DATA, help="Source data for synthetic pairs")
    parser.add_argument("--max-samples", type=int, default=100, help="Number of DPO pairs")
    parser.add_argument("--max-steps", type=int, default=20, help="Max training steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(args.model, low_cpu_mem_usage=True)

    adapter_path = Path(args.adapter)
    if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(adapter_path))
        print(f"Loaded SFT adapter from {adapter_path}")

    lora_cfg = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # NOTE: Preference data is currently synthetic (truncated correct answers).
    # This is a placeholder until real preference data (e.g. UltraFeedback) is added.
    print("Loading/simulating preference data (SYNTHETIC - see warning above)...")
    raw = load_dataset("parquet", data_files={"train": args.train_data}, split="train")
    raw = raw.select(range(min(args.max_samples, len(raw))))

    def build_preference(example):
        prompt = f"<|user|>\n{example['instruction']}\n<|assistant|>\n"
        resp = example['response']
        chosen = resp
        rejected = resp[:len(resp)//2] + "[incomplete]"
        return {"prompt": prompt, "chosen": chosen + "</s>", "rejected": rejected + "</s>"}

    dpo_data = raw.map(build_preference, remove_columns=raw.column_names)

    training_args = DPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        max_length=512,
        max_prompt_length=256,
        num_train_epochs=1,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="no",
        learning_rate=args.lr,
        report_to=[],
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        beta=args.beta,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dpo_data,
        tokenizer=tokenizer,
    )

    print("Starting DPO training...")
    t0 = time.time()
    trainer.train()
    print(f"DPO done in {time.time()-t0:.1f}s")

    model.save_pretrained(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))

    summary = {
        "model": Path(args.model).name,
        "num_pairs": len(dpo_data),
        "train_time_sec": round(time.time() - t0, 1),
        "max_steps": args.max_steps,
        "data_source": f"synthetic (from {Path(args.train_data).name})",
        "warning": "Preference pairs are synthetic (truncated answers). Replace with real data for production use."
    }
    with open(str(output_dir / "dpo_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved!")

if __name__ == "__main__":
    main()

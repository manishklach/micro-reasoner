import json, time, argparse
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from config import PROJECT_ROOT, DEFAULT_MODEL, DEFAULT_SFT_DATA, DEFAULT_EVAL_DATA, DEFAULT_SFT_OUTPUT, get_chat_template

def main():
    parser = argparse.ArgumentParser(description="LoRA SFT training")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model path")
    parser.add_argument("--train-data", default=DEFAULT_SFT_DATA, help="Training data (parquet)")
    parser.add_argument("--eval-data", default=DEFAULT_EVAL_DATA, help="Eval data (parquet)")
    parser.add_argument("--output-dir", default=DEFAULT_SFT_OUTPUT, help="Output directory")
    parser.add_argument("--max-samples", type=int, default=500, help="Max training samples (0 = all)")
    parser.add_argument("--max-steps", type=int, default=50, help="Max training steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading model (float32, CPU)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"Model loaded in {time.time()-t0:.1f}s, params: {model.num_parameters()/1e6:.1f}M")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading training data...")
    data_files = {"train": args.train_data}
    if args.eval_data:
        data_files["eval"] = args.eval_data

    raw_train = load_dataset("parquet", data_files={"train": args.train_data}, split="train")
    if args.max_samples and args.max_samples > 0:
        raw_train = raw_train.select(range(min(args.max_samples, len(raw_train))))
    print(f"Train samples: {len(raw_train)}")

    raw_eval = None
    if args.eval_data:
        raw_eval = load_dataset("parquet", data_files={"eval": args.eval_data}, split="eval")
        print(f"Eval samples: {len(raw_eval)}")

    tmpl = get_chat_template(args.model)

    def format_and_tokenize(examples):
        texts = []
        for inst, resp in zip(examples["instruction"], examples["response"]):
            text = f"{tmpl['user_prefix']}{inst}{tmpl['user_suffix']}{resp}{tmpl['assistant_suffix']}"
            texts.append(text)
        tokenized = tokenizer(texts, truncation=True, padding=False, max_length=512)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    print("Tokenizing...")
    train_tok = raw_train.map(format_and_tokenize, batched=True, remove_columns=raw_train.column_names, num_proc=2)
    eval_tok = None
    if raw_eval:
        eval_tok = raw_eval.map(format_and_tokenize, batched=True, remove_columns=raw_eval.column_names, num_proc=2)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=1,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no" if eval_tok is None else "steps",
        eval_steps=50 if eval_tok else None,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        learning_rate=args.lr,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        report_to=[],
        optim="adamw_torch",
        max_grad_norm=0.3,
        gradient_checkpointing=False,
        use_cpu=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
    eval_dataset = eval_tok.select(range(50)) if eval_tok else None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print("Starting SFT training...")
    t0 = time.time()
    trainer.train()
    total_time = time.time() - t0
    print(f"Training complete in {total_time:.1f}s")

    model.save_pretrained(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))

    summary = {
        "model": Path(args.model).name,
        "train_samples": len(train_tok),
        "eval_samples": len(eval_tok) if eval_tok else 0,
        "train_time_sec": round(total_time, 1),
        "trainable_params": model.num_parameters(only_trainable=True),
        "total_params": model.num_parameters(),
        "max_steps": args.max_steps,
        "learning_rate": args.lr,
    }
    with open(str(output_dir / "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {json.dumps(summary, indent=2)}")

if __name__ == "__main__":
    main()

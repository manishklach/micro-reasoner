import json
import time
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

os.chdir("/home/manishkl/single-gpu-reasoner")
os.makedirs("outputs/sft", exist_ok=True)

model_path = "models/smollm2-360m"
output_dir = "outputs/sft"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading model (float32, CPU)...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_path,
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

print("Loading training data (subset for demo)...")
train_data = load_dataset("parquet", data_files="data/processed/sft_train.parquet", split="train").select(range(500))
eval_data = load_dataset("parquet", data_files="data/processed/sft_eval.parquet", split="train")

def format_and_tokenize(examples):
    texts = []
    for inst, resp in zip(examples["instruction"], examples["response"]):
        text = f"<|user|>\n{inst}\n<|assistant|>\n{resp}</s>"
        texts.append(text)
    tokenized = tokenizer(texts, truncation=True, padding=False, max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Tokenizing...")
train_tok = train_data.map(format_and_tokenize, batched=True, remove_columns=train_data.column_names, num_proc=2)
eval_tok = eval_data.map(format_and_tokenize, batched=True, remove_columns=eval_data.column_names, num_proc=2)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    max_steps=50,
    logging_steps=1,
    save_strategy="no",
    eval_strategy="no",
    save_total_limit=2,
    remove_unused_columns=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    learning_rate=2e-4,
    warmup_steps=20,
    lr_scheduler_type="cosine",
    report_to=[],
    optim="adamw_torch",
    max_grad_norm=0.3,
    gradient_checkpointing=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok.select(range(50)),
    data_collator=data_collator,
    processing_class=tokenizer,
)

print("Starting SFT training...")
t0 = time.time()
trainer.train()
total_time = time.time() - t0
print(f"Training complete in {total_time:.1f}s")

model.save_pretrained(f"{output_dir}/final_adapter")
tokenizer.save_pretrained(f"{output_dir}/final_adapter")

summary = {
    "model": "SmolLM2-360M-Instruct",
    "train_samples": len(train_tok),
    "eval_samples": len(eval_tok),
    "train_time_sec": round(total_time, 1),
    "trainable_params": model.num_parameters(only_trainable=True),
    "total_params": model.num_parameters(),
}
with open(f"{output_dir}/training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary: {json.dumps(summary, indent=2)}")

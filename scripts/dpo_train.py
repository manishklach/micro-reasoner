import json, time, os, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig

os.chdir("/home/manishkl/single-gpu-reasoner")
os.makedirs("outputs/dpo", exist_ok=True)

model_path = "models/smollm2-360m"
adapter_path = "outputs/sft/final_adapter"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True)

# Load SFT adapter if available
if os.path.exists(f"{adapter_path}/adapter_config.json"):
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter_path)
    print("Loaded SFT adapter")

lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj"], task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# Load preference data (sample if no real data exists)
print("Loading/simulating preference data...")
raw = load_dataset("parquet", data_files="data/processed/sft_train.parquet", split="train").select(range(100))

def build_preference(example):
    prompt = f"<|user|>\n{example['instruction']}\n<|assistant|>\n"
    # chosen = correct answer, rejected = shortened/wrong version
    resp = example['response']
    chosen = resp
    rejected = resp[:len(resp)//2] + "[incomplete]"
    return {"prompt": prompt, "chosen": chosen + "</s>", "rejected": rejected + "</s>"}

dpo_data = raw.map(build_preference, remove_columns=raw.column_names)

training_args = DPOConfig(
    output_dir="outputs/dpo",
    per_device_train_batch_size=1,
    max_length=512,
    max_prompt_length=256,
    num_train_epochs=1,
    max_steps=20,
    logging_steps=1,
    save_strategy="no",
    learning_rate=1e-5,
    report_to=[],
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    beta=0.1,
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

model.save_pretrained("outputs/dpo/final_adapter")
tokenizer.save_pretrained("outputs/dpo/final_adapter")
print("Saved!")

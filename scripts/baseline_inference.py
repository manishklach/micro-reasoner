import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

os.makedirs("outputs/benchmarks", exist_ok=True)

model_path = "models/smollm2-360m"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("Loading baseline model...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.eval()
print(f"Loaded in {time.time()-t0:.1f}s, params: {model.num_parameters()/1e6:.1f}M")

# Load eval prompts
with open("data/eval/eval_prompts.json", "r") as f:
    eval_prompts = json.load(f)

print(f"Running baseline on {min(20, len(eval_prompts))} prompts...")
results = []
total_tokens = 0
total_time = 0

for i, item in enumerate(eval_prompts[:20]):
    prompt = item["prompt"]
    text = f"<|user|>\n{prompt}\n<|assistant|>\n"
    inputs = tokenizer(text, return_tensors="pt")
    
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=False,
        )
    elapsed = time.time() - start
    
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    num_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    
    total_tokens += num_tokens
    total_time += elapsed
    
    results.append({
        "prompt": prompt,
        "generated": generated,
        "tokens": num_tokens,
        "time_sec": round(elapsed, 2),
        "tokens_per_sec": round(num_tokens / elapsed, 2),
    })
    
    if (i + 1) % 5 == 0:
        print(f"  [{i+1}/20] {elapsed:.1f}s, {num_tokens} tok, {num_tokens/elapsed:.1f} tok/s")

metrics = {
    "model": "SmolLM2-360M-Instruct",
    "num_prompts": len(results),
    "avg_latency_sec": round(total_time / len(results), 2),
    "avg_tokens_per_sec": round(total_tokens / total_time, 2),
    "avg_output_length": round(total_tokens / len(results), 1),
    "total_time_sec": round(total_time, 2),
    "total_tokens": total_tokens,
    "device": "cpu",
}

with open("outputs/benchmarks/baseline_results.json", "w") as f:
    json.dump({"metrics": metrics, "results": results}, f, indent=2)

print("\n=== Baseline Metrics ===")
for k, v in metrics.items():
    print(f"  {k}: {v}")
print("\nResults saved.")

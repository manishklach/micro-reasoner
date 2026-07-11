"""Baseline inference benchmark for the base model.
Replaces baseline_inference.py. Run with: python scripts/benchmark_base.py
"""
import json, time, argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import PROJECT_ROOT, DEFAULT_MODEL, DEFAULT_EVAL_PROMPTS

def main():
    parser = argparse.ArgumentParser(description="Baseline inference benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model path")
    parser.add_argument("--eval-file", default=DEFAULT_EVAL_PROMPTS, help="Eval prompts JSON")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "benchmarks"), help="Output directory")
    parser.add_argument("--max-prompts", type=int, default=20, help="Number of prompts to benchmark")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--do-sample", action="store_true", help="Use sampling (default: greedy)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading baseline model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"Loaded in {time.time()-t0:.1f}s, params: {model.num_parameters()/1e6:.1f}M")

    with open(args.eval_file, "r") as f:
        eval_prompts = json.load(f)

    eval_prompts = eval_prompts[:args.max_prompts]
    print(f"Running baseline on {len(eval_prompts)} prompts...")

    results = []
    total_tokens = 0
    total_time = 0

    for i, item in enumerate(eval_prompts):
        prompt = item["prompt"]
        text = f"<|user|>\n{prompt}\n<|assistant|>\n"
        inputs = tokenizer(text, return_tensors="pt")

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                do_sample=args.do_sample,
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
            print(f"  [{i+1}/{len(eval_prompts)}] {elapsed:.1f}s, {num_tokens} tok, {num_tokens/elapsed:.1f} tok/s")

    metrics = {
        "model": Path(args.model).name,
        "num_prompts": len(results),
        "avg_latency_sec": round(total_time / len(results), 2),
        "avg_tokens_per_sec": round(total_tokens / total_time, 2),
        "avg_output_length": round(total_tokens / len(results), 1),
        "total_time_sec": round(total_time, 2),
        "total_tokens": total_tokens,
        "device": "cpu",
    }

    with open(str(output_dir / "baseline_results.json"), "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)

    print("\n=== Baseline Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nResults saved to {output_dir / 'baseline_results.json'}")

if __name__ == "__main__":
    main()

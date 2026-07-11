"""Evaluate model accuracy on GSM8K-style math problems.
Extracts final numeric answers from generations and compares to references.

Usage:
  python scripts/evaluate.py --model models/smollm2-360m
  python scripts/evaluate.py --model models/smollm2-360m --adapter outputs/sft/final_adapter
"""
import json, re, time, argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import PROJECT_ROOT, DEFAULT_MODEL, DEFAULT_EVAL_PROMPTS, get_chat_template

def extract_answer(text):
    """Extract final numeric answer from GSM8K-style response."""
    text = text.strip()
    patterns = [
        r"####\s*(-?\d+\.?\d*)",
        r"The answer is\s*(-?\d+\.?\d*)",
        r"answer is\s*(-?\d+\.?\d*)",
        r"=\s*(-?\d+\.?\d*)\s*$",
        r"\b(\d+)\s*(?:\n|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.MULTILINE)
        if m:
            return m.group(1)
    return None

def evaluate(prompts, model, tokenizer, max_tokens=256, temperature=0.0, chat_tmpl=None):
    results = []
    correct = 0
    total_gen_tokens = 0
    total_time = 0
    repetitions = 0

    for item in prompts:
        prompt = item["prompt"]
        reference = item.get("reference", "")
        ref_answer = extract_answer(reference)

        text = f"{chat_tmpl['user_prefix']}{prompt}{chat_tmpl['user_suffix']}"
        inputs = tokenizer(text, return_tensors="pt")

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
            )
        elapsed = time.time() - t0

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        num_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        gen_answer = extract_answer(generated)

        is_correct = False
        if ref_answer and gen_answer and ref_answer == gen_answer:
            is_correct = True
            correct += 1

        gen_lower = generated.lower()
        if any(phrase in gen_lower for phrase in gen_lower.count(s) > 3 for s in [".", "!", "?"]):
            repetitions += 1

        total_gen_tokens += num_tokens
        total_time += elapsed

        results.append({
            "prompt": prompt[:80],
            "reference_answer": ref_answer,
            "generated_answer": gen_answer,
            "correct": is_correct,
            "tokens": num_tokens,
            "time_sec": round(elapsed, 2),
        })

    accuracy = correct / len(prompts) if prompts else 0
    avg_tok_per_sec = total_gen_tokens / total_time if total_time else 0
    avg_output_len = total_gen_tokens / len(prompts) if prompts else 0

    return {
        "num_prompts": len(prompts),
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "avg_tokens_per_sec": round(avg_tok_per_sec, 2),
        "avg_output_length": round(avg_output_len, 1),
        "total_time_sec": round(total_time, 2),
        "repetition_failures": repetitions,
        "details": results,
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate model accuracy on math reasoning")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--eval-file", default=DEFAULT_EVAL_PROMPTS)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "benchmarks"))
    parser.add_argument("--max-prompts", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, low_cpu_mem_usage=True)

    if args.adapter:
        adapter_path = Path(args.adapter)
        if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(model, str(adapter_path))
            print(f"Loaded adapter from {args.adapter}")
    model.eval()

    with open(args.eval_file) as f:
        all_prompts = json.load(f)
    prompts = all_prompts[:args.max_prompts]
    print(f"Evaluating on {len(prompts)} prompts...")

    tmpl = get_chat_template(args.adapter if args.adapter else args.model)
    results = evaluate(prompts, model, tokenizer, args.max_tokens, chat_tmpl=tmpl)

    label = Path(args.adapter).parent.name if args.adapter else "baseline"
    out_file = output_dir / f"eval_{label}.json"
    with open(str(out_file), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Evaluation Results ({label}) ===")
    print(f"  Accuracy:  {results['correct']}/{results['num_prompts']} = {results['accuracy']:.2%}")
    print(f"  Tok/s:     {results['avg_tokens_per_sec']}")
    print(f"  Avg len:   {results['avg_output_length']} tok")
    print(f"  Repetitions: {results['repetition_failures']}")
    print(f"\nSaved to {out_file}")

if __name__ == "__main__":
    main()

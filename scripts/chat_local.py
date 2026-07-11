"""Local inference server for the fine-tuned model.
Usage:
  python scripts/chat_local.py
  python scripts/chat_local.py --adapter outputs/sft/final_adapter --prompt "What is 15 * 7?"
"""
import argparse, json, time, torch, sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import PROJECT_ROOT, DEFAULT_MODEL

def main():
    parser = argparse.ArgumentParser(description="Local model inference")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model path")
    parser.add_argument("--adapter", default=None, help="Adapter path (e.g. outputs/sft/final_adapter)")
    parser.add_argument("--prompt", default=None, help="Single prompt (non-interactive mode)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    args = parser.parse_args()

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

    if args.prompt:
        single = args.prompt
        text = f"<|user|>\n{single}\n<|assistant|>\n"
        inputs = tokenizer(text, return_tensors="pt")
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=args.temperature > 0, temperature=args.temperature if args.temperature > 0 else None)
        elapsed = time.time() - t0
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        tok_count = outputs.shape[1] - inputs["input_ids"].shape[1]
        print(f"\n{response}\n")
        print(f"[{tok_count} tok in {elapsed:.1f}s, {tok_count/elapsed:.1f} tok/s]")
        return

    print("Ready. Type your prompts (Ctrl+C to exit).\n")
    while True:
        try:
            prompt = input(">>> ")
        except EOFError:
            break

        text = f"<|user|>\n{prompt}\n<|assistant|>\n"
        inputs = tokenizer(text, return_tensors="pt")

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
        elapsed = time.time() - t0

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        tok_count = outputs.shape[1] - inputs["input_ids"].shape[1]
        print(f"\n{response}\n")
        print(f"[{tok_count} tok in {elapsed:.1f}s, {tok_count/elapsed:.1f} tok/s]\n")

if __name__ == "__main__":
    main()

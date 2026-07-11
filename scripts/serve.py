"""Local inference server for the fine-tuned model.
Usage: python scripts/serve.py [--model models/smollm2-360m] [--adapter outputs/sft/final_adapter]
"""
import argparse, json, time, torch, sys, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.chdir("/home/manishkl/single-gpu-reasoner")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/smollm2-360m")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, low_cpu_mem_usage=True)
    if args.adapter and os.path.exists(f"{args.adapter}/adapter_config.json"):
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"Loaded adapter from {args.adapter}")
    model.eval()

    print("Ready. Type your prompts (Ctrl+C to exit).\n")
    while True:
        if args.prompt:
            prompt = args.prompt
            args.prompt = None
        else:
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

"""Export a fine-tuned model to GGUF format for llama.cpp inference.
Merges LoRA adapter into base model, then converts to GGUF.

Requires: llama.cpp cloned and built locally, OR use pre-quantized HF weights.

Usage:
  # Merge adapter + show conversion command
  python scripts/export_gguf.py --adapter outputs/sft/final_adapter

  # Full path with llama.cpp installed at ../llama.cpp
  python scripts/export_gguf.py --adapter outputs/sft/final_adapter --llama-cpp-dir ../llama.cpp

  # Just merge (no conversion)
  python scripts/export_gguf.py --adapter outputs/sft/final_adapter --merge-only
"""
import argparse, os, sys, subprocess, shutil
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import PROJECT_ROOT, DEFAULT_MODEL

def merge_adapter(base_model_path, adapter_path, output_path):
    print(f"Loading base model from {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(base_model_path, low_cpu_mem_usage=True)

    print(f"Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_path}...")
    model.save_pretrained(str(output_path))
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(str(output_path))
    print("Merge complete.")
    return output_path

def convert_to_gguf(model_path, output_path, llama_cpp_dir, quantize="q4_k_m"):
    convert_script = Path(llama_cpp_dir) / "convert.py"
    if not convert_script.exists():
        print(f"ERROR: {convert_script} not found.")
        print(f"Clone llama.cpp first: git clone https://github.com/ggerganov/llama.cpp {llama_cpp_dir}")
        return False

    cmd = [
        sys.executable, str(convert_script), str(model_path),
        "--outfile", str(output_path),
        "--outtype", quantize,
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"GGUF model saved to {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Export model to GGUF")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model path")
    parser.add_argument("--adapter", required=True, help="LoRA adapter path")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "quantized"), help="Output directory")
    parser.add_argument("--llama-cpp-dir", default=None, help="Path to llama.cpp (skip conversion if not set)")
    parser.add_argument("--merge-only", action="store_true", help="Only merge adapter, skip GGUF conversion")
    parser.add_argument("--quantize", default="q4_k_m", help="Quantization type (q4_k_m, q5_k_m, q8_0, etc.)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_name = Path(args.adapter).parent.name
    model_name = Path(args.model).name
    merged_dir = output_dir / f"{model_name}_{adapter_name}_merged"
    gguf_path = output_dir / f"{model_name}_{adapter_name}_{args.quantize}.gguf"

    # Step 1: Merge adapter
    merge_adapter(args.model, args.adapter, merged_dir)

    if args.merge_only:
        print(f"Merged model saved to {merged_dir}")
        print("Skipping GGUF conversion (--merge-only)")
        return

    # Step 2: Convert to GGUF
    if args.llama_cpp_dir:
        convert_to_gguf(merged_dir, gguf_path, args.llama_cpp_dir, args.quantize)
    else:
        print(f"\nMerged model ready at: {merged_dir}")
        print("\nTo convert to GGUF manually:")
        print(f"  1. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp")
        print(f"  2. Convert:")
        print(f"     python llama.cpp/convert.py {merged_dir} \\")
        print(f"       --outfile {gguf_path} --outtype {args.quantize}")
        print(f"  3. Run inference:")
        print(f"     ./llama.cpp/main -m {gguf_path} -p \"Your prompt\" -n 256")
        print("\nOr use pre-quantized HF GGUF files directly.")

if __name__ == "__main__":
    main()

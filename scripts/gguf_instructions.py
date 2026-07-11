"""GGUF conversion instructions for llama.cpp inference.
This is a manual process — automated conversion is planned but blocked until
llama.cpp's convert.py is tested with SmolLM2 architecture.

Usage: python scripts/gguf_instructions.py
"""
import os, sys
from pathlib import Path
from config import PROJECT_ROOT

def main():
    model_path = str(PROJECT_ROOT / "models" / "smollm2-360m")
    output_path = str(PROJECT_ROOT / "outputs" / "quantized" / "smollm2-360m-q4_k_m.gguf")

    print("=" * 60)
    print("GGUF Conversion Guide")
    print("=" * 60)
    print()
    print("To convert the model to GGUF for faster CPU inference with llama.cpp:")
    print()
    print("  1. Clone llama.cpp:")
    print("     git clone https://github.com/ggerganov/llama.cpp")
    print()
    print(f"  2. Convert model:")
    print(f"     python llama.cpp/convert.py {model_path} \\")
    print(f"       --outfile {output_path} --outtype q4_k_m")
    print()
    print("  3. Run inference:")
    print(f"     ./llama.cpp/main -m {output_path} -p \"Your prompt here\" -n 256")
    print()
    print("  Alternative: Use pre-quantized HF GGUF files:")
    print("    https://huggingface.co/models?search=smollm2-360m-GGUF")
    print()
    print("  Note: Adapter merging not yet automated.")
    print("  After SFT/DPO, merge adapter into base model before conversion.")
    print("=" * 60)

if __name__ == "__main__":
    main()

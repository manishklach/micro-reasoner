"""Convert model to GGUF format for llama.cpp inference.
Usage: pip install llama-cpp-python  # requires build tools
       python scripts/convert_to_gguf.py
"""
import os, json, subprocess, sys

os.chdir("/home/manishkl/single-gpu-reasoner")
model_path = "models/smollm2-360m"
output_path = "outputs/quantized/smollm2-360m-q4_k_m.gguf"

print("GGUF conversion requires llama.cpp")
print("To convert manually:")
print(f"  1. git clone https://github.com/ggerganov/llama.cpp")
print(f"  2. cd llama.cpp && make")
print(f'  3. python convert.py {model_path} --outfile {output_path} --outtype q4_k_m')
print(f"\nOr use the huggingface direct GGUF files:")
print(f"  https://huggingface.co/models?search=smollm2-360m-GGUF")

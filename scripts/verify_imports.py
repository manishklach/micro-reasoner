import transformers
import datasets
import accelerate
import peft
import bitsandbytes
import safetensors
import sentencepiece
print('transformers:', transformers.__version__)
print('datasets:', datasets.__version__)
print('accelerate:', accelerate.__version__)
print('peft:', peft.__version__)
print('bitsandbytes:', bitsandbytes.__version__)
print('safetensors OK')
print('sentencepiece OK')
# Check trl
try:
    import trl
    print('trl OK')
except ImportError as e:
    print('trl import failed:', e)
print('All imports successful')

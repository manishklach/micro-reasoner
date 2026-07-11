from pathlib import Path
import os

def get_project_root():
    return Path(__file__).resolve().parent.parent

PROJECT_ROOT = get_project_root()
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
BENCHMARKS_DIR = OUTPUTS_DIR / "benchmarks"
SFT_DIR = OUTPUTS_DIR / "sft"
DPO_DIR = OUTPUTS_DIR / "dpo"
QUANTIZED_DIR = OUTPUTS_DIR / "quantized"

# ---- Model paths ----
DEFAULT_MODEL = str(MODELS_DIR / "smollm2-360m")
QWEN_MODEL = str(MODELS_DIR / "qwen2.5-1.5b")

# ---- Data paths ----
DEFAULT_SFT_DATA = str(PROCESSED_DIR / "sft_train.parquet")
DEFAULT_EVAL_DATA = str(PROCESSED_DIR / "sft_eval.parquet")
DEFAULT_EVAL_PROMPTS = str(EVAL_DIR / "eval_prompts.json")
DEFAULT_DPO_DATA = str(PROCESSED_DIR / "dpo_pairs_ultrafeedback.parquet")

# ---- Output paths ----
DEFAULT_SFT_OUTPUT = str(SFT_DIR)
DEFAULT_DPO_OUTPUT = str(DPO_DIR)

# ---- Chat templates ----
CHAT_TEMPLATES = {
    "smollm2": {
        "user_prefix": "<|user|>\n",
        "user_suffix": "\n<|assistant|>\n",
        "assistant_suffix": "</s>",
    },
    "qwen2.5": {
        "user_prefix": "<|im_start|>user\n",
        "user_suffix": "<|im_end|>\n<|im_start|>assistant\n",
        "assistant_suffix": "<|im_end|>",
    },
}

def get_chat_template(model_name_or_path):
    name = Path(model_name_or_path).name.lower()
    if "qwen" in name:
        return CHAT_TEMPLATES["qwen2.5"]
    return CHAT_TEMPLATES["smollm2"]

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

DEFAULT_MODEL = str(MODELS_DIR / "smollm2-360m")
DEFAULT_SFT_DATA = str(PROCESSED_DIR / "sft_train.parquet")
DEFAULT_EVAL_DATA = str(PROCESSED_DIR / "sft_eval.parquet")
DEFAULT_SFT_OUTPUT = str(SFT_DIR)
DEFAULT_DPO_OUTPUT = str(DPO_DIR)
DEFAULT_EVAL_PROMPTS = str(EVAL_DIR / "eval_prompts.json")

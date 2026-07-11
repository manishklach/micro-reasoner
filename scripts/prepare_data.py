import os
os.environ['HF_HUB_DISABLE_HF_TRANSFER'] = '1'
from datasets import load_dataset, Dataset, concatenate_datasets
import json

os.makedirs('/home/manishkl/single-gpu-reasoner/data/processed', exist_ok=True)
os.makedirs('/home/manishkl/single-gpu-reasoner/data/eval', exist_ok=True)

print("Loading GSM8K...")
gsm8k = load_dataset('openai/gsm8k', 'main', split='train')
print(f"GSM8K train: {len(gsm8k)} examples")

def format_gsm8k(example):
    return {
        'instruction': example['question'],
        'response': example['answer'],
        'source': 'gsm8k'
    }

sft_data = gsm8k.map(format_gsm8k, remove_columns=gsm8k.column_names)
split = sft_data.train_test_split(test_size=0.1, seed=42)
train_data = split['train']
eval_data = split['test']

print(f"SFT train: {len(train_data)}, eval: {len(eval_data)}")

train_data.to_parquet('/home/manishkl/single-gpu-reasoner/data/processed/sft_train.parquet')
eval_data.to_parquet('/home/manishkl/single-gpu-reasoner/data/processed/sft_eval.parquet')

print("Loading GSM8K test set for evaluation...")
gsm8k_test = load_dataset('openai/gsm8k', 'main', split='test')
eval_prompts = gsm8k_test.select(range(min(100, len(gsm8k_test))))

eval_formatted = []
for ex in eval_prompts:
    eval_formatted.append({
        'prompt': ex['question'],
        'reference': ex['answer'],
        'category': 'math/reasoning'
    })

with open('/home/manishkl/single-gpu-reasoner/data/eval/eval_prompts.json', 'w') as f:
    json.dump(eval_formatted, f, indent=2)

print(f"Evaluation set: {len(eval_formatted)} prompts saved")
print("Data preparation complete!")

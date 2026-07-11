# Outputs

| Directory | Contents |
|-----------|----------|
| `benchmarks/` | Baseline + eval benchmark results (JSON) |
| `sft/` | SFT adapter weights + training summary |
| `dpo/` | DPO adapter weights + training summary |
| `quantized/` | GGUF quantized models (when exported) |

## Benchmark Files
- `baseline_results.json` — raw generation results + aggregate metrics
- `eval_{label}.json` — evaluation with accuracy scoring

## Compare Stages
```bash
python scripts/benchmark_compare.py
```

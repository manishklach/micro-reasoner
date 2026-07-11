"""Compare benchmark results across model stages (baseline, SFT, DPO).
Reads eval_{label}.json files from outputs/benchmarks/ and prints a comparison table.

Usage: python scripts/benchmark_compare.py
"""
import json, sys, argparse
from pathlib import Path
from config import PROJECT_ROOT

def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument("--benchmarks-dir", default=str(PROJECT_ROOT / "outputs" / "benchmarks"))
    parser.add_argument("--labels", nargs="+", default=["baseline", "sft", "dpo"],
                        help="Labels to compare (must match eval_{label}.json files)")
    args = parser.parse_args()

    benchmarks_dir = Path(args.benchmarks_dir)
    results = []

    for label in args.labels:
        path = benchmarks_dir / f"eval_{label}.json"
        if not path.exists():
            print(f"  [SKIP] {path.name} not found")
            continue
        with open(str(path)) as f:
            data = json.load(f)
        results.append((label, data))

    if not results:
        print("No eval files found. Run scripts/evaluate.py first.")
        print(f"Expected files in: {benchmarks_dir}")
        return

    print(f"\n{'Stage':<12} {'Accuracy':<10} {'Tok/s':<10} {'Avg Len':<10} {'Repetitions':<12} {'Time (s)':<10}")
    print("-" * 64)
    for label, data in results:
        acc = f"{data.get('accuracy', 0):.2%}"
        tps = f"{data.get('avg_tokens_per_sec', 0):.1f}"
        avg_len = f"{data.get('avg_output_length', 0):.0f}"
        reps = f"{data.get('repetition_failures', 'N/A')}"
        t = f"{data.get('total_time_sec', 0):.1f}"
        print(f"{label:<12} {acc:<10} {tps:<10} {avg_len:<10} {reps:<12} {t:<10}")

    if len(results) > 1:
        base_acc = results[0][1].get("accuracy", 0)
        for label, data in results[1:]:
            acc = data.get("accuracy", 0)
            delta = acc - base_acc
            print(f"\n{label} vs {results[0][0]}: accuracy Δ = {delta:+.2%}")

if __name__ == "__main__":
    main()

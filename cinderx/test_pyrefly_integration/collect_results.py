#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Collect benchmark JSON results and print a summary table.

Each benchmark script writes a JSON file to BENCHMARK_RESULTS_DIR.
This script reads them all and prints a formatted comparison table.
"""

import json
import os
import sys


def load_results(results_dir: str) -> list[dict]:
    results = []
    for name in sorted(os.listdir(results_dir)):
        if name.endswith(".json"):
            with open(os.path.join(results_dir, name)) as f:
                data = json.load(f)
            # Clean up config name: "1_Untyped_(no_JIT)" -> "Untyped (no JIT)"
            config = data.get("config", "")
            # Strip leading number prefix used for sort order
            if len(config) > 2 and config[0].isdigit() and config[1] == "_":
                config = config[2:]
            config = config.replace("_", " ")
            data["config"] = config
            results.append(data)
    return results


def fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    return f"{seconds:.3f}s"


def fmt_speedup(baseline: float | None, current: float | None) -> str:
    if baseline is None or current is None or baseline == 0:
        return ""
    ratio = baseline / current
    if ratio > 1:
        return f" ({ratio:.1f}x faster)"
    elif ratio < 1:
        return f" ({1/ratio:.1f}x slower)"
    return " (1.0x)"


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print a markdown-style table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "-|-".join("-" * widths[i] for i in range(len(headers)))
    print(f" {header_line}")
    print(f" {sep_line}")
    for row in rows:
        line = " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        print(f" {line}")


def main() -> None:
    results_dir = os.environ.get("BENCHMARK_RESULTS_DIR", "/tmp/benchmark_results")
    if not os.path.isdir(results_dir):
        print(f"No results directory found at {results_dir}")
        sys.exit(1)

    results = load_results(results_dir)
    if not results:
        print("No result files found.")
        sys.exit(1)

    print()
    print("=" * 72)
    print("  BENCHMARK SUMMARY")
    print("=" * 72)
    print()

    # --- Group 1: Scalar math (add + multiply + fib) ---
    scalar_results = [r for r in results if "scalar_math_best_s" in r]
    if scalar_results:
        print("  Scalar Math (add + multiply + fib, 100k iterations)")
        print()
        headers = ["Config", "Best Time", "us/iter", "vs baseline"]
        rows = []
        baseline = None
        for r in scalar_results:
            best = r["scalar_math_best_s"]
            us = r.get("scalar_math_us_per_iter", best / 100_000 * 1e6)
            if baseline is None:
                baseline = best
            rows.append([
                r["config"],
                fmt_time(best),
                f"{us:.1f}",
                fmt_speedup(baseline, best),
            ])
        print_table(headers, rows)
        print()

    # --- Group 2: MLP forward pass ---
    mlp_results = [r for r in results if "mlp_best_s" in r]
    if mlp_results:
        print("  MLP Forward Pass (500 iterations)")
        print()
        headers = ["Config", "Best Time", "vs baseline"]
        rows = []
        baseline = None
        for r in mlp_results:
            best = r["mlp_best_s"]
            if baseline is None:
                baseline = best
            rows.append([
                r["config"],
                fmt_time(best),
                fmt_speedup(baseline, best),
            ])
        print_table(headers, rows)
        print()

    # --- Group 3: MicroGPT inference (small, from ml_ops) ---
    gpt_infer_results = [r for r in results if "microgpt_infer_best_s" in r]
    if gpt_infer_results:
        print("  MicroGPT Inference — small (20 tokens x 10 passes)")
        print()
        headers = ["Config", "Best Time", "vs baseline"]
        rows = []
        baseline = None
        for r in gpt_infer_results:
            best = r["microgpt_infer_best_s"]
            if baseline is None:
                baseline = best
            rows.append([
                r["config"],
                fmt_time(best),
                fmt_speedup(baseline, best),
            ])
        print_table(headers, rows)
        print()

    # --- Group 4: Karpathy MicroGPT (full training + inference) ---
    gpt_train_results = [r for r in results if "microgpt_train_s" in r]
    if gpt_train_results:
        steps = gpt_train_results[0].get("microgpt_steps", "?")
        samples = gpt_train_results[0].get("microgpt_samples", "?")
        print(f"  Karpathy MicroGPT (training={steps} steps, inference={samples} samples)")
        print()
        headers = ["Config", "Training", "ms/step", "Inference", "Total", "vs baseline"]
        rows = []
        baseline_total = None
        for r in gpt_train_results:
            train = r["microgpt_train_s"]
            ms = r["microgpt_ms_per_step"]
            infer = r["microgpt_infer_s"]
            total = r["microgpt_total_s"]
            if baseline_total is None:
                baseline_total = total
            rows.append([
                r["config"],
                fmt_time(train),
                f"{ms:.1f}",
                fmt_time(infer),
                fmt_time(total),
                fmt_speedup(baseline_total, total),
            ])
        print_table(headers, rows)
        print()

    print("=" * 72)
    print()


if __name__ == "__main__":
    main()

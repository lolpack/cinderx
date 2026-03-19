#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Typed benchmark runner — imports math_ops through the Pyrefly Static Python loader.

The Static Python compiler works best with scalar-typed code (int, float).
This benchmark imports math_ops (add, multiply, fib) through the pyrefly
loader for Static Python compilation, then runs a tight loop to measure
the speedup from type-specialized bytecode + JIT.

Three-way comparison:
  A) PYTHONJITDISABLE=1 python pure_python_benchmark.py  — untyped baseline
  B) PYTHONJITDISABLE=1 python typed_benchmark.py        — Static Python, no JIT
  C) python typed_benchmark.py                            — Static Python + JIT
"""

import json
import os
import sys
import time


def main() -> None:
    print(f"Python {sys.version}")
    print()

    jit_enabled = os.environ.get("PYTHONJITDISABLE") != "1"

    # --- Install the Pyrefly loader BEFORE importing math_ops ---
    type_dir = os.path.join(os.path.dirname(__file__), "benchmark", "pyrefly_types")
    if not os.path.isdir(type_dir):
        print(f"FAIL: Type dir not found at {type_dir}")
        sys.exit(1)

    index_path = os.path.join(type_dir, "index.json")
    with open(index_path) as f:
        index = json.load(f)
    source_modules = set()
    for entry in index.get("modules", []):
        path = entry.get("path", "")
        if path.endswith(".py"):
            source_modules.add(entry["module_name"])

    types_subdir = os.path.join(type_dir, "types")
    if not os.path.isdir(types_subdir):
        types_subdir = type_dir

    print(f"  Source modules:       {source_modules}")
    print(f"  Types dir:            {types_subdir}")

    import cinderx
    print(f"  CinderX initialized:  {cinderx.is_initialized()}")

    from cinderx.compiler.strict.pyrefly_loader import install as install_loader
    install_loader(pyrefly_type_dir=types_subdir, opt_in_list=source_modules)
    print(f"  Pyrefly loader:       installed")
    print()

    # --- Import math_ops through the Static Python loader ---
    sys.path.insert(0, os.path.dirname(__file__))
    print("Importing benchmark.math_ops through Pyrefly loader...")
    from benchmark import math_ops

    print(f"  math_ops module type: {type(math_ops).__name__}")

    # Check Static Python compilation
    try:
        import _static
        for fn_name in ("add", "multiply", "fib"):
            fn = getattr(math_ops, fn_name, None)
            if fn is not None:
                is_static = _static.is_static_callable(fn)
                print(f"  _static.is_static_callable({fn_name}): {is_static}")
    except ImportError:
        print("  _static module not available")
    print()

    # --- Enable JIT ---
    import cinderx.jit
    cinderx.jit.enable()
    cinderx.jit.auto()

    # --- Benchmark: scalar math through Static Python ---
    num_iters = 100_000
    repeat = 3

    print("=" * 60)
    print("BENCHMARK: Static Python scalar math (add + multiply + fib)")
    print("=" * 60)
    print(f"  {num_iters:,} iterations x {repeat} runs")
    print()

    # Warmup
    for i in range(5000):
        math_ops.add(i, i + 1)
        math_ops.multiply(i, i + 1)
        math_ops.fib(20)

    times = []
    for run in range(repeat):
        t0 = time.perf_counter()
        for i in range(num_iters):
            math_ops.add(i, i + 1)
            math_ops.multiply(i, i + 1)
            math_ops.fib(20)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  Run {run + 1}/{repeat}: {elapsed:.3f}s "
              f"({elapsed / num_iters * 1e6:.1f} us/iter)")

    best = min(times)
    avg = sum(times) / len(times)

    # --- Also import and benchmark ml_ops (regular import, JIT only) ---
    print()
    print("=" * 60)
    print("BENCHMARK: Pure Python MLP (JIT only, no Static Python)")
    print("=" * 60)

    from benchmark import ml_ops

    # Warmup
    ml_ops.run_mlp(50)

    mlp_times = []
    mlp_iters = 500
    for run in range(repeat):
        t0 = time.perf_counter()
        ml_ops.run_mlp(mlp_iters)
        elapsed = time.perf_counter() - t0
        mlp_times.append(elapsed)
        print(f"  Run {run + 1}/{repeat}: {elapsed:.3f}s")

    best_mlp = min(mlp_times)

    print()
    print("=" * 60)
    print("BENCHMARK: Pure Python MicroGPT (JIT only, no Static Python)")
    print("=" * 60)
    print("  Config: 1 layer, 16 embd, 4 heads, vocab=27, 20 tokens x 10 passes")
    print()

    # Warmup
    ml_ops.run_microgpt(num_tokens=5, num_passes=2)

    gpt_times = []
    for run in range(repeat):
        t0 = time.perf_counter()
        ml_ops.run_microgpt(num_tokens=20, num_passes=10)
        elapsed = time.perf_counter() - t0
        gpt_times.append(elapsed)
        print(f"  Run {run + 1}/{repeat}: {elapsed:.3f}s")

    best_gpt = min(gpt_times)

    # --- Summary ---
    num_compiled = len(cinderx.jit.get_compiled_functions())
    compile_time = cinderx.jit.get_compilation_time()
    cinderx.jit.disable()

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  JIT enabled:          {jit_enabled}")
    print(f"  JIT compiled funcs:   {num_compiled}")
    print(f"  JIT compile time:     {compile_time}ms")
    print(f"  Scalar math best:     {best:.3f}s ({best / num_iters * 1e6:.1f} us/iter)")
    print(f"  MLP best:             {best_mlp:.3f}s")
    print(f"  MicroGPT best:        {best_gpt:.3f}s")
    print()
    print("Typed benchmark: SUCCESS")
    print("=" * 60)

    # Write JSON results for collection
    results_dir = os.environ.get("BENCHMARK_RESULTS_DIR")
    config = os.environ.get("BENCHMARK_CONFIG", "Static Python" + (" + JIT" if jit_enabled else " (no JIT)"))
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        results = {
            "config": config,
            "scalar_math_best_s": best,
            "scalar_math_us_per_iter": best / num_iters * 1e6,
            "mlp_best_s": best_mlp,
            "microgpt_infer_best_s": best_gpt,
        }
        path = os.path.join(results_dir, f"scalar_mlp_{config.replace(' ', '_')}.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

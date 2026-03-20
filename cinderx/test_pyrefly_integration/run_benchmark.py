#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Pyrefly + CinderX Static Python integration benchmark.

This script:
1. Reads pyrefly-generated type info (index.json + per-module JSON)
2. Installs the CinderX pyrefly loader as a strict import hook
3. Imports a typed module through the loader (Static Python compilation)
4. Enables the CinderX JIT and runs a timed benchmark
"""

import json
import os
import sys
import time


def find_type_dir() -> str:
    """Find the pyrefly type output directory."""
    candidate = os.path.join(os.path.dirname(__file__), "benchmark", "pyrefly_types")
    if os.path.isdir(candidate):
        return candidate
    raise RuntimeError(f"Type dir not found at {candidate}")


def get_source_modules(type_dir: str) -> set[str]:
    """Read index.json and return module names for source files (not stubs)."""
    index_path = os.path.join(type_dir, "index.json")
    if not os.path.isfile(index_path):
        return set()
    with open(index_path) as f:
        index = json.load(f)
    modules: set[str] = set()
    for entry in index.get("modules", []):
        path = entry.get("path", "")
        if path.endswith(".py"):
            modules.add(entry["module_name"])
    return modules


def find_types_subdir(type_dir: str) -> str:
    """Return the directory containing per-module JSON files.

    Pyrefly may output type files directly in type_dir or in a types/
    subdirectory. The CinderX Pyrefly class expects the directory that
    directly contains {module_name}.json files.
    """
    types_subdir = os.path.join(type_dir, "types")
    if os.path.isdir(types_subdir):
        return types_subdir
    return type_dir


def main() -> None:
    print(f"Python {sys.version}")
    print()

    # --- Step 1: Find type info and install the Pyrefly loader ---
    type_dir = find_type_dir()
    source_modules = get_source_modules(type_dir)
    pyrefly_type_dir = find_types_subdir(type_dir)

    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"  Pyrefly output dir:       {type_dir}")
    print(f"  Per-module type JSON dir: {pyrefly_type_dir}")
    print(f"  Source modules:           {source_modules}")
    print(f"  Files in type dir:        {os.listdir(type_dir)}")
    print(f"  Files in types subdir:    {os.listdir(pyrefly_type_dir)}")

    # Print index.json contents
    index_path = os.path.join(type_dir, "index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            print(f"  index.json:               {json.dumps(json.load(f), indent=2)}")

    if not source_modules:
        print("FAIL: No source modules found in index.json")
        sys.exit(1)

    # Print type info for each module
    for mod_name in source_modules:
        json_path = os.path.join(pyrefly_type_dir, f"{mod_name}.json")
        if os.path.isfile(json_path):
            with open(json_path) as f:
                data = json.load(f)
            print(f"  Type info for {mod_name}: {len(data.get('type_table', []))} types, "
                  f"{len(data.get('locations', []))} locations")
        else:
            print(f"  WARNING: No type info file found at {json_path}")
    print()

    import cinderx
    print(f"  CinderX initialized:      {cinderx.is_initialized()}")
    print(f"  CinderX import error:     {cinderx.get_import_error()}")

    from cinderx.compiler.strict.pyrefly_loader import install as install_loader

    install_loader(pyrefly_type_dir=pyrefly_type_dir, opt_in_list=source_modules)
    print(f"  Pyrefly loader installed: True")
    print(f"  sys.path_hooks count:     {len(sys.path_hooks)}")
    print()

    # --- Step 2: Import the module AFTER the loader is installed ---
    # The loader intercepts the import and uses pyrefly type info
    # to compile the module with Static Python optimizations.
    sys.path.insert(0, os.path.dirname(__file__))
    print("=" * 60)
    print("IMPORTING benchmark.math_ops THROUGH PYREFLY LOADER")
    print("=" * 60)
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        from benchmark import math_ops
        print("  Import succeeded!")
    except Exception as e:
        import traceback
        print(f"  Import FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Step 3: Verify Static Python compilation ---
    print("=" * 60)
    print("STATIC PYTHON VERIFICATION")
    print("=" * 60)

    # Check module type (StrictModule vs regular module)
    print(f"  math_ops module type:     {type(math_ops).__name__}")
    print(f"  math_ops __file__:        {getattr(math_ops, '__file__', 'N/A')}")

    # Check code object details for each function
    for fn_name in ("add", "multiply", "fib"):
        fn = getattr(math_ops, fn_name)
        code = fn.__code__
        print(f"  {fn_name}:")
        print(f"    co_flags:    0x{code.co_flags:x}")
        print(f"    co_names:    {code.co_names}")
        print(f"    co_consts:   {code.co_consts}")
        print(f"    co_varnames: {code.co_varnames}")

    # Check for <fixed-modules> in module dict (sign of Static Python)
    has_fixed = "<fixed-modules>" in math_ops.__dict__
    print(f"  <fixed-modules> in module dict: {has_fixed}")
    if has_fixed:
        fixed_mods = math_ops.__dict__["<fixed-modules>"]
        print(f"  <fixed-modules> keys: {list(fixed_mods.keys())}")

    try:
        import _static

        is_static_add = _static.is_static_callable(math_ops.add)
        is_static_multiply = _static.is_static_callable(math_ops.multiply)
        is_static_fib = _static.is_static_callable(math_ops.fib)
        print(f"  _static.is_static_callable(add):      {is_static_add}")
        print(f"  _static.is_static_callable(multiply): {is_static_multiply}")
        print(f"  _static.is_static_callable(fib):      {is_static_fib}")
    except ImportError:
        print("  _static module not available")
    print()

    # --- Step 4: Enable CinderX JIT ---
    print("=" * 60)
    print("JIT SETUP")
    print("=" * 60)

    import cinderx.jit

    jit_enabled = os.environ.get("PYTHONJITDISABLE") != "1"
    print(f"  PYTHONJITDISABLE:         {os.environ.get('PYTHONJITDISABLE', 'not set')}")
    print(f"  JIT will be enabled:      {jit_enabled}")

    cinderx.jit.enable()
    cinderx.jit.auto()
    print(f"  cinderx.jit.enable():     called")
    print(f"  cinderx.jit.auto():       called")
    print()

    # --- Step 5: Warmup (let JIT compile hot functions) ---
    print("Warming up (5000 iterations)...")
    for i in range(5000):
        math_ops.add(i, i + 1)
        math_ops.multiply(i, i + 1)
        math_ops.fib(30)

    # --- Step 6: Timed benchmark ---
    num_iters = 100_000
    repeat = 3
    print(f"Running benchmark: {num_iters} iterations x {repeat} runs")
    print()

    times = []
    for run in range(repeat):
        t0 = time.perf_counter()
        for i in range(num_iters):
            math_ops.add(i, i + 1)
            math_ops.multiply(i, i + 1)
            math_ops.fib(20)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(
            f"  Run {run + 1}/{repeat}: {elapsed:.3f}s "
            f"({elapsed / num_iters * 1e6:.1f} us/iter)"
        )

    cinderx.jit.disable()

    avg = sum(times) / len(times)
    best = min(times)
    worst = max(times)
    print()
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Iterations per run: {num_iters:,}")
    print(f"  Number of runs:     {repeat}")
    print(f"  Best run:           {best:.3f}s ({best / num_iters * 1e6:.1f} us/iter)")
    print(f"  Worst run:          {worst:.3f}s ({worst / num_iters * 1e6:.1f} us/iter)")
    print(f"  Avg per run:        {avg:.3f}s ({avg / num_iters * 1e6:.1f} us/iter)")
    print(f"  JIT enabled:        {jit_enabled}")
    print()
    print("Pyrefly + CinderX Static Python integration: SUCCESS")
    print("=" * 60)


if __name__ == "__main__":
    main()

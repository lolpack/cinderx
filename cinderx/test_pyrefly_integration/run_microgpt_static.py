#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Run microgpt.py through the Pyrefly Static Python loader.

This script installs the CinderX Pyrefly import hook, then imports microgpt
so that the module is compiled with Static Python type-specialized bytecode.
All of microgpt's training and inference runs at module level on import.
"""

import json
import os
import sys
import time


def main() -> None:
    print(f"Python {sys.version}")
    print()

    jit_enabled = os.environ.get("PYTHONJITDISABLE") != "1"

    # --- Locate pyrefly type info for microgpt ---
    type_dir = os.path.join(os.path.dirname(__file__), "microgpt_pyrefly_types")
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

    # --- Enable JIT ---
    import cinderx.jit
    cinderx.jit.enable()
    cinderx.jit.auto()

    # --- Import microgpt through the Static Python loader ---
    # microgpt.py must be on sys.path. The CI copies it alongside this script.
    microgpt_dir = os.path.dirname(__file__)
    if microgpt_dir not in sys.path:
        sys.path.insert(0, microgpt_dir)

    print("=" * 60)
    print("RUNNING microgpt THROUGH STATIC PYTHON LOADER")
    print("=" * 60)
    sys.stdout.flush()

    t0 = time.perf_counter()
    import microgpt  # noqa: F401 — top-level code runs training + inference
    elapsed = time.perf_counter() - t0

    # --- Verify Static Python compilation ---
    print()
    print("=" * 60)
    print("STATIC PYTHON VERIFICATION")
    print("=" * 60)

    print(f"  microgpt module type: {type(microgpt).__name__}")

    try:
        import _static
        for fn_name in ("linear", "softmax", "rmsnorm", "gpt"):
            fn = getattr(microgpt, fn_name, None)
            if fn is not None:
                is_static = _static.is_static_callable(fn)
                print(f"  _static.is_static_callable({fn_name}): {is_static}")
    except ImportError:
        print("  _static module not available")

    # --- Summary ---
    num_compiled = len(cinderx.jit.get_compiled_functions())
    compile_time = cinderx.jit.get_compilation_time()
    cinderx.jit.disable()

    train_s = getattr(microgpt, "train_elapsed", None)
    infer_s = getattr(microgpt, "infer_elapsed", None)
    total_s = getattr(microgpt, "total_elapsed", None)
    num_steps = getattr(microgpt, "num_steps", None)
    num_samples = getattr(microgpt, "num_samples", None)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  JIT enabled:          {jit_enabled}")
    print(f"  JIT compiled funcs:   {num_compiled}")
    print(f"  JIT compile time:     {compile_time}ms")
    print(f"  Total elapsed:        {elapsed:.3f}s")
    if train_s is not None:
        print(f"  Training:             {train_s:.3f}s ({train_s/num_steps*1000:.1f} ms/step)")
        print(f"  Inference:            {infer_s:.3f}s")
    print()
    print("MicroGPT Static Python: SUCCESS")
    print("=" * 60)

    # Write JSON results for collection (microgpt.py also writes its own,
    # but this one includes the correct config label)
    results_dir = os.environ.get("BENCHMARK_RESULTS_DIR")
    config = os.environ.get("BENCHMARK_CONFIG", "Static Python + JIT")
    if results_dir and train_s is not None:
        os.makedirs(results_dir, exist_ok=True)
        results = {
            "config": config,
            "microgpt_train_s": round(train_s, 3),
            "microgpt_ms_per_step": round(train_s / num_steps * 1000, 1),
            "microgpt_infer_s": round(infer_s, 3),
            "microgpt_total_s": round(total_s, 3),
            "microgpt_steps": num_steps,
            "microgpt_samples": num_samples,
        }
        path = os.path.join(results_dir, f"microgpt_{config.replace(' ', '_')}.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

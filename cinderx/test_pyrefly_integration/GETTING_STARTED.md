# Getting Started with Static Python (CinderX + Pyrefly)

Static Python is a performance optimization that uses type information from [Pyrefly](https://github.com/facebook/pyrefly) (a fast Rust-based type checker) to generate optimized bytecode at import time. When combined with the CinderX JIT compiler, this produces native code with type-specialized operations — eliminating type checks, enabling direct field access, and unlocking optimizations that standard CPython cannot perform.

## How It Works

The integration is a two-phase pipeline with a JSON contract between Pyrefly and CinderX:

```
┌─────────────────────────────┐       JSON files        ┌──────────────────────────────────┐
│         PYREFLY             │  ──────────────────────> │           CINDERX                │
│  (Rust type checker)        │                          │  (Python runtime extension)      │
│                             │   index.json             │                                  │
│  pyrefly check              │   types/<module>.json    │  pyrefly_loader.install()        │
│    --report-cinderx <dir>   │                          │    -> PyreflyCompiler            │
│    --search-path <root>     │                          │    -> PyreflyTypeBinder          │
│    <source files>           │                          │    -> Static Python bytecode     │
│                             │                          │    -> CinderX JIT compilation    │
└─────────────────────────────┘                          └──────────────────────────────────┘
```

**Phase 1 — Pyrefly generates type info:** Running `pyrefly check --report-cinderx` analyzes your Python source files and produces per-module JSON files containing a type table and source location mappings.

**Phase 2 — CinderX consumes type info at import time:** The `pyrefly_loader.install()` function hooks into Python's import system. When your typed modules are imported, CinderX's Static Python compiler reads the type info and generates optimized bytecode with type-specialized operations.

## Why the JIT Matters

Static Python bytecode alone is faster than standard Python bytecode, but the real performance gains come when the CinderX JIT compiler kicks in. The JIT:

- Compiles hot Python functions to native machine code
- Uses the type information from Static Python to generate **type-specialized** native code (e.g., direct integer operations instead of generic `PyObject` dispatch)
- Eliminates Python frame overhead for JIT-compiled functions
- Performs inline caching, constant folding, and dead code elimination

With Static Python + JIT together, typed Python code can approach the performance of hand-written C extensions.

## Requirements

- **Python 3.14+** (CPython)
- **Linux x86_64** (CinderX native extension requirement)
- **Pyrefly 0.57.0+** (first version with `--report-cinderx` support)
- `pip install cinderx "pyrefly>=0.57.0"`

## Quick Start

### 1. Install

```bash
pip install cinderx "pyrefly>=0.57.0"
```

### 2. Write typed Python code

```python
# math_ops.py
def add(x: int, y: int) -> int:
    return x + y

def multiply(x: int, y: int) -> int:
    return x * y

def fib(n: int) -> int:
    if n <= 1:
        return n
    a: int = 0
    b: int = 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
```

### 3. Generate type info with Pyrefly

```bash
pyrefly check \
    --search-path . \
    --report-cinderx pyrefly_types \
    math_ops.py
```

This produces:
- `pyrefly_types/index.json` — lists analyzed modules
- `pyrefly_types/types/math_ops.json` — per-module type table and source locations

### 4. Run with the Pyrefly loader and CinderX JIT

```python
# run.py
import json
import os
import sys

# --- Install the Pyrefly loader BEFORE importing your typed modules ---
type_dir = "pyrefly_types"

# Read index.json to find source modules
with open(os.path.join(type_dir, "index.json")) as f:
    index = json.load(f)
source_modules = {
    e["module_name"]
    for e in index.get("modules", [])
    if e.get("path", "").endswith(".py")
}

# Point at the types/ subdirectory containing per-module JSON
types_subdir = os.path.join(type_dir, "types")
if not os.path.isdir(types_subdir):
    types_subdir = type_dir

from cinderx.compiler.strict.pyrefly_loader import install as install_loader
install_loader(pyrefly_type_dir=types_subdir, opt_in_list=source_modules)

# --- Now import your typed module (goes through the Static Python compiler) ---
import math_ops

# --- Enable CinderX JIT for native code compilation ---
import cinderx.jit
cinderx.jit.enable()
cinderx.jit.auto()  # Auto-compile hot functions

# Your code runs with Static Python + JIT optimizations
result = math_ops.fib(30)
print(f"fib(30) = {result}")
```

> **Important:** The pyrefly loader must be installed *before* importing any modules you want statically compiled. If you import `math_ops` before calling `install_loader()`, the default Python loader handles it and Static Python is bypassed.

### 5. Run and compare

```bash
# With CinderX JIT (Static Python + native compilation)
python run.py

# Without JIT (for comparison)
PYTHONJITDISABLE=1 python run.py
```

## Environment Variables

| Variable | Description |
|---|---|
| `PYTHONJITDISABLE=1` | Disable CinderX JIT (for A/B comparison) |
| `PYTHONJITDUMPASM=1` | Dump JIT assembly to stderr |
| `PYTHONSTRICTMODULESTUBSPATH=<dir>` | Override strict module stubs directory |

## Benchmark Results

All benchmarks run on GitHub Actions `ubuntu-latest`, Python 3.14.3, CinderX from PyPI.

### Static Python scalar math (100,000 iterations x 3 runs)

Each iteration calls `add(i, i+1)` + `multiply(i, i+1)` + `fib(20)`.
Functions are compiled through the Pyrefly loader (`is_static_callable: True`).

| Configuration | Best time | us/iter | Speedup |
|---|---|---|---|
| Static Python + CinderX JIT | 0.070s | 0.7 | **6.4x** |
| Static Python, no JIT | 0.447s | 4.5 | baseline |

### Pure Python MLP forward pass (500 iterations x 3 runs)

64-input, 128-hidden, 10-output MLP. Nested loops with dot products, ReLU, softmax. No external dependencies.

| Configuration | Best time | Speedup |
|---|---|---|
| CinderX JIT | 0.182s | **1.7x** |
| No JIT (CPython 3.14 baseline) | 0.317s | baseline |

### Pure Python MicroGPT inference (20 tokens x 10 passes x 3 runs)

1-layer transformer with 4-head attention, 16-dim embeddings, vocab=27. All Python, no NumPy/PyTorch.

| Configuration | Best time | Speedup |
|---|---|---|
| CinderX JIT | 0.051s | **7.3x** |
| No JIT (CPython 3.14 baseline) | 0.373s | baseline |

> Run the benchmarks yourself with the scripts in this directory:
> ```bash
> # A) Untyped baseline (no types, no JIT)
> PYTHONJITDISABLE=1 python pure_python_benchmark.py
>
> # B) Static Python types, no JIT
> PYTHONJITDISABLE=1 python typed_benchmark.py
>
> # C) Static Python types + JIT
> python typed_benchmark.py
> ```

## Current Limitations

- **List literals**: The Static Python compiler does not yet support `list[float]` generic annotations or list literal expressions in statically-compiled modules. Use scalar type annotations (`int`, `float`) for best results.
- **PYTHONJITDUMPASM**: Dumps assembly for *all* JIT-compiled functions globally, which can produce very large output and timeouts in CI.

## Further Reading

- [CinderX GitHub](https://github.com/facebookincubator/cinderx)
- [Pyrefly GitHub](https://github.com/facebook/pyrefly)

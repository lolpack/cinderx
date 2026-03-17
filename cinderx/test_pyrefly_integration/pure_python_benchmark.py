#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Pure Python ML benchmarks for CinderX JIT — no external dependencies.

Two benchmarks that exercise Python-level math where the JIT can shine:

1. MLP forward pass: nested loops doing dot products, ReLU, softmax
2. MicroGPT inference: simplified transformer (attention + MLP) in pure Python
"""

import math
import os
import random
import sys
import time


# ---------------------------------------------------------------------------
# Benchmark 1: Pure Python MLP forward pass
# ---------------------------------------------------------------------------

def benchmark_mlp(iterations: int = 200) -> float:
    INPUT_SIZE = 64
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 10

    random.seed(42)
    weights_ih = [[random.uniform(-1, 1) for _ in range(INPUT_SIZE)] for _ in range(HIDDEN_SIZE)]
    weights_ho = [[random.uniform(-1, 1) for _ in range(HIDDEN_SIZE)] for _ in range(OUTPUT_SIZE)]
    inputs = [random.random() for _ in range(INPUT_SIZE)]

    t0 = time.perf_counter()
    for _ in range(iterations):
        # Forward: input -> hidden (dot product + ReLU)
        hidden = []
        for row in weights_ih:
            dot = sum(x * w for x, w in zip(inputs, row))
            hidden.append(max(0, dot))

        # Forward: hidden -> output (dot product + softmax)
        outputs = []
        for row in weights_ho:
            dot = sum(h * w for h, w in zip(hidden, row))
            outputs.append(math.exp(dot))

        total = sum(outputs)
        outputs = [o / total for o in outputs]

    elapsed = time.perf_counter() - t0
    ops_per_pass = (INPUT_SIZE * HIDDEN_SIZE) + (HIDDEN_SIZE * OUTPUT_SIZE)
    total_ops = ops_per_pass * iterations
    return elapsed, total_ops


# ---------------------------------------------------------------------------
# Benchmark 2: MicroGPT inference (Karpathy-style, pure Python)
# https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
# ---------------------------------------------------------------------------

class Value:
    """Minimal autograd scalar — forward only for inference benchmark."""
    __slots__ = ('data',)

    def __init__(self, data: float):
        self.data = data

    def __add__(self, other):
        return Value(self.data + (other.data if isinstance(other, Value) else other))

    def __mul__(self, other):
        return Value(self.data * (other.data if isinstance(other, Value) else other))

    def __sub__(self, other):
        return Value(self.data - (other.data if isinstance(other, Value) else other))

    def __truediv__(self, other):
        return Value(self.data / (other.data if isinstance(other, Value) else other))

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        return Value(self.data ** other)

    def exp(self):
        return Value(math.exp(self.data))

    def relu(self):
        return Value(max(0.0, self.data))


def linear(x: list, w: list) -> list:
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits: list) -> list:
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(e.data for e in exps)
    return [Value(e.data / total) for e in exps]


def rmsnorm(x: list) -> list:
    ms = sum(xi.data * xi.data for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [Value(xi.data * scale) for xi in x]


def benchmark_microgpt(num_tokens: int = 20, num_passes: int = 10) -> float:
    n_layer = 1
    n_embd = 16
    n_head = 4
    head_dim = n_embd // n_head
    vocab_size = 27  # a-z + BOS

    random.seed(42)
    matrix = lambda nout, nin: [[Value(random.gauss(0, 0.08)) for _ in range(nin)] for _ in range(nout)]

    state_dict = {
        'wte': matrix(vocab_size, n_embd),
        'wpe': matrix(num_tokens, n_embd),
        'lm_head': matrix(vocab_size, n_embd),
    }
    for i in range(n_layer):
        state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
        state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
        state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

    def gpt_forward(token_id, pos_id, keys, values):
        tok_emb = state_dict['wte'][token_id]
        pos_emb = state_dict['wpe'][pos_id]
        x = [Value(t.data + p.data) for t, p in zip(tok_emb, pos_emb)]
        x = rmsnorm(x)

        for li in range(n_layer):
            x_residual = x
            x = rmsnorm(x)
            q = linear(x, state_dict[f'layer{li}.attn_wq'])
            k = linear(x, state_dict[f'layer{li}.attn_wk'])
            v = linear(x, state_dict[f'layer{li}.attn_wv'])
            keys[li].append(k)
            values[li].append(v)
            x_attn = []
            for h in range(n_head):
                hs = h * head_dim
                q_h = q[hs:hs + head_dim]
                k_h = [ki[hs:hs + head_dim] for ki in keys[li]]
                v_h = [vi[hs:hs + head_dim] for vi in values[li]]
                attn_logits = [
                    sum(q_h[j].data * k_h[t][j].data for j in range(head_dim)) / head_dim ** 0.5
                    for t in range(len(k_h))
                ]
                max_logit = max(attn_logits)
                attn_exp = [math.exp(a - max_logit) for a in attn_logits]
                attn_sum = sum(attn_exp)
                attn_weights = [a / attn_sum for a in attn_exp]
                head_out = [
                    Value(sum(attn_weights[t] * v_h[t][j].data for t in range(len(v_h))))
                    for j in range(head_dim)
                ]
                x_attn.extend(head_out)
            x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
            x = [Value(a.data + b.data) for a, b in zip(x, x_residual)]
            x_residual = x
            x = rmsnorm(x)
            x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
            x = [xi.relu() for xi in x]
            x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
            x = [Value(a.data + b.data) for a, b in zip(x, x_residual)]

        return linear(x, state_dict['lm_head'])

    t0 = time.perf_counter()
    for _ in range(num_passes):
        keys = [[] for _ in range(n_layer)]
        values = [[] for _ in range(n_layer)]
        token_id = vocab_size - 1  # BOS
        for pos_id in range(num_tokens):
            logits = gpt_forward(token_id, pos_id, keys, values)
            probs = softmax(logits)
            token_id = max(range(len(probs)), key=lambda i: probs[i].data)
    elapsed = time.perf_counter() - t0
    return elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Python {sys.version}")
    print()

    jit_enabled = os.environ.get("PYTHONJITDISABLE") != "1"

    # Enable CinderX JIT
    import cinderx.jit
    cinderx.jit.enable()
    cinderx.jit.auto()

    # --- MLP Benchmark ---
    print("=" * 60)
    print("BENCHMARK 1: Pure Python MLP Forward Pass")
    print("=" * 60)

    # Warmup
    benchmark_mlp(iterations=50)

    mlp_times = []
    mlp_iters = 500
    for run in range(3):
        elapsed, total_ops = benchmark_mlp(iterations=mlp_iters)
        mlp_times.append(elapsed)
        print(f"  Run {run + 1}/3: {elapsed:.3f}s "
              f"({total_ops / elapsed:,.0f} float ops/sec)")

    avg_mlp = sum(mlp_times) / len(mlp_times)
    best_mlp = min(mlp_times)
    print(f"\n  Best:  {best_mlp:.3f}s")
    print(f"  Avg:   {avg_mlp:.3f}s")
    print()

    # --- MicroGPT Benchmark ---
    print("=" * 60)
    print("BENCHMARK 2: MicroGPT Inference (Pure Python Transformer)")
    print("=" * 60)
    print("  Config: 1 layer, 16 embd, 4 heads, vocab=27")
    print("  20 tokens x 10 passes")
    print()

    # Warmup
    benchmark_microgpt(num_tokens=5, num_passes=2)

    gpt_times = []
    for run in range(3):
        elapsed = benchmark_microgpt(num_tokens=20, num_passes=10)
        gpt_times.append(elapsed)
        print(f"  Run {run + 1}/3: {elapsed:.3f}s")

    avg_gpt = sum(gpt_times) / len(gpt_times)
    best_gpt = min(gpt_times)
    print(f"\n  Best:  {best_gpt:.3f}s")
    print(f"  Avg:   {avg_gpt:.3f}s")
    print()

    # --- Summary ---
    num_compiled = len(cinderx.jit.get_compiled_functions())
    compile_time = cinderx.jit.get_compilation_time()
    cinderx.jit.disable()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  JIT enabled:          {jit_enabled}")
    print(f"  JIT compiled funcs:   {num_compiled}")
    print(f"  JIT compile time:     {compile_time}ms")
    print(f"  MLP best:             {best_mlp:.3f}s")
    print(f"  MicroGPT best:        {best_gpt:.3f}s")
    print()
    print("Pure Python ML benchmark: SUCCESS")
    print("=" * 60)


if __name__ == "__main__":
    main()

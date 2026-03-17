"""Typed pure Python ML operations for Static Python benchmarking.

Functions use scalar type annotations (int, float) that Static Python can
optimize. Container variables are left unannotated to avoid the Static Python
compiler's checked_list assertion.
"""

import math
import random


# ---------------------------------------------------------------------------
# MLP Forward Pass
# ---------------------------------------------------------------------------

def dot_product(a, b) -> float:
    result: float = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result


def relu(x: float) -> float:
    if x > 0.0:
        return x
    return 0.0


def mlp_forward(inputs, weights_ih, weights_ho):
    # Input -> Hidden (dot product + ReLU)
    hidden = []
    for row in weights_ih:
        val: float = dot_product(inputs, row)
        hidden.append(relu(val))

    # Hidden -> Output (dot product + softmax)
    raw_outputs = []
    for row in weights_ho:
        val2: float = dot_product(hidden, row)
        raw_outputs.append(math.exp(val2))

    total: float = sum(raw_outputs)
    outputs = []
    for o in raw_outputs:
        outputs.append(o / total)
    return outputs


def run_mlp(iterations: int) -> None:
    input_size: int = 64
    hidden_size: int = 128
    output_size: int = 10

    random.seed(42)
    weights_ih = [
        [random.uniform(-1.0, 1.0) for _ in range(input_size)]
        for _ in range(hidden_size)
    ]
    weights_ho = [
        [random.uniform(-1.0, 1.0) for _ in range(hidden_size)]
        for _ in range(output_size)
    ]
    inputs = [random.random() for _ in range(input_size)]

    for _ in range(iterations):
        mlp_forward(inputs, weights_ih, weights_ho)


# ---------------------------------------------------------------------------
# MicroGPT Inference (pure Python transformer)
# ---------------------------------------------------------------------------

def mat_vec(w, x):
    out = []
    for row in w:
        out.append(dot_product(row, x))
    return out


def vec_add(a, b):
    out = []
    for i in range(len(a)):
        out.append(a[i] + b[i])
    return out


def vec_scale(x, s: float):
    out = []
    for v in x:
        out.append(v * s)
    return out


def rmsnorm(x):
    n: int = len(x)
    ms: float = 0.0
    for v in x:
        ms += v * v
    ms = ms / n
    scale: float = (ms + 1e-5) ** -0.5
    return vec_scale(x, scale)


def softmax_floats(logits):
    max_val: float = logits[0]
    for v in logits:
        if v > max_val:
            max_val = v
    exps = []
    total: float = 0.0
    for v in logits:
        e: float = math.exp(v - max_val)
        exps.append(e)
        total += e
    out = []
    for e2 in exps:
        out.append(e2 / total)
    return out


def relu_vec(x):
    out = []
    for v in x:
        out.append(relu(v))
    return out


def gpt_forward(
    token_id: int,
    pos_id: int,
    keys,
    values,
    state_dict,
    n_layer: int,
    n_head: int,
    head_dim: int,
):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = vec_add(tok_emb, pos_emb)
    x = rmsnorm(x)

    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = mat_vec(state_dict[f'layer{li}.attn_wq'], x)
        k = mat_vec(state_dict[f'layer{li}.attn_wk'], x)
        v = mat_vec(state_dict[f'layer{li}.attn_wv'], x)
        keys[li].append(k)
        values[li].append(v)

        x_attn = []
        for h in range(n_head):
            hs: int = h * head_dim
            he: int = hs + head_dim
            q_h = q[hs:he]

            attn_logits = []
            inv_scale: float = 1.0 / (head_dim ** 0.5)
            for t in range(len(keys[li])):
                k_t = keys[li][t][hs:he]
                score: float = dot_product(q_h, k_t) * inv_scale
                attn_logits.append(score)

            attn_weights = softmax_floats(attn_logits)

            for j in range(head_dim):
                val: float = 0.0
                for t2 in range(len(values[li])):
                    val += attn_weights[t2] * values[li][t2][hs + j]
                x_attn.append(val)

        x = mat_vec(state_dict[f'layer{li}.attn_wo'], x_attn)
        x = vec_add(x, x_residual)

        x_residual = x
        x = rmsnorm(x)
        x = mat_vec(state_dict[f'layer{li}.mlp_fc1'], x)
        x = relu_vec(x)
        x = mat_vec(state_dict[f'layer{li}.mlp_fc2'], x)
        x = vec_add(x, x_residual)

    return mat_vec(state_dict['lm_head'], x)


def run_microgpt(num_tokens: int, num_passes: int) -> None:
    n_layer: int = 1
    n_embd: int = 16
    n_head: int = 4
    head_dim: int = n_embd // n_head
    vocab_size: int = 27

    random.seed(42)

    def make_matrix(nout: int, nin: int):
        return [
            [random.gauss(0.0, 0.08) for _ in range(nin)]
            for _ in range(nout)
        ]

    state_dict = {
        'wte': make_matrix(vocab_size, n_embd),
        'wpe': make_matrix(num_tokens, n_embd),
        'lm_head': make_matrix(vocab_size, n_embd),
    }
    for i in range(n_layer):
        state_dict[f'layer{i}.attn_wq'] = make_matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wk'] = make_matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wv'] = make_matrix(n_embd, n_embd)
        state_dict[f'layer{i}.attn_wo'] = make_matrix(n_embd, n_embd)
        state_dict[f'layer{i}.mlp_fc1'] = make_matrix(4 * n_embd, n_embd)
        state_dict[f'layer{i}.mlp_fc2'] = make_matrix(n_embd, 4 * n_embd)

    for _ in range(num_passes):
        keys = [[] for _ in range(n_layer)]
        vals = [[] for _ in range(n_layer)]
        token_id: int = vocab_size - 1
        for pos_id in range(num_tokens):
            logits = gpt_forward(
                token_id, pos_id, keys, vals,
                state_dict, n_layer, n_head, head_dim,
            )
            probs = softmax_floats(logits)
            best: int = 0
            for idx in range(len(probs)):
                if probs[idx] > probs[best]:
                    best = idx
            token_id = best

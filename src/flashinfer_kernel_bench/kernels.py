from __future__ import annotations

import math
from typing import Any

import numpy as np


def attention_numpy(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Reference scaled dot-product attention.

    Shapes:
    - query: `[batch, heads, q_len, dim]`
    - key: `[batch, heads, kv_len, dim]`
    - value: `[batch, heads, kv_len, dim]`
    """

    dim = query.shape[-1]
    scores = np.matmul(query, np.swapaxes(key, -1, -2)) / math.sqrt(dim)
    weights = softmax(scores, axis=-1)
    return np.matmul(weights, value)


def attention_torch(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    import torch

    q = torch.from_numpy(query)
    k = torch.from_numpy(key)
    v = torch.from_numpy(value)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    return out.detach().cpu().numpy()


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def sample_top_p_numpy(logits: np.ndarray, top_p: float = 0.9) -> np.ndarray:
    """Deterministic top-p representative token for benchmark repeatability."""

    probabilities = softmax(logits, axis=-1)
    sorted_indices = np.argsort(-probabilities, axis=-1)
    sorted_probs = np.take_along_axis(probabilities, sorted_indices, axis=-1)
    cumulative = np.cumsum(sorted_probs, axis=-1)
    mask = cumulative <= top_p
    mask[..., 0] = True
    masked_probs = np.where(mask, sorted_probs, 0)
    chosen_in_sorted = np.argmax(masked_probs, axis=-1)
    return np.take_along_axis(sorted_indices, np.expand_dims(chosen_in_sorted, -1), axis=-1).squeeze(-1)


def flashinfer_available() -> bool:
    try:
        import flashinfer  # noqa: F401
    except Exception:
        return False
    return True


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def max_abs_error(left: Any, right: Any) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


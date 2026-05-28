from __future__ import annotations

import numpy as np

from flashinfer_kernel_bench.benchmark import make_inputs, run_case
from flashinfer_kernel_bench.cases import CASES, parse_cases
from flashinfer_kernel_bench.kernels import attention_numpy, sample_top_p_numpy


def test_parse_cases() -> None:
    cases = parse_cases("tiny,small")
    assert [case.name for case in cases] == ["tiny", "small"]


def test_attention_numpy_shape() -> None:
    case = CASES["tiny"]
    query, key, value, _ = make_inputs(case, seed=1)
    output = attention_numpy(query, key, value)
    assert output.shape == (case.batch, case.heads, case.q_len, case.dim)
    assert np.isfinite(output).all()


def test_sample_top_p_is_deterministic() -> None:
    logits = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, -1.0]], dtype=np.float32)
    first = sample_top_p_numpy(logits, top_p=0.9)
    second = sample_top_p_numpy(logits, top_p=0.9)
    assert np.array_equal(first, second)
    assert first.shape == (2,)


def test_run_case_reports_correctness_error() -> None:
    row = run_case(CASES["tiny"], backend="numpy", seed=3)
    assert row["case"] == "tiny"
    assert row["backend"] == "numpy"
    assert row["max_abs_error_vs_numpy"] == 0.0
    assert row["attention_mean_ms"] >= 0.0


from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .cases import BenchCase, parse_cases
from .kernels import (
    attention_numpy,
    attention_torch,
    flashinfer_available,
    max_abs_error,
    sample_top_p_numpy,
    torch_available,
)


def make_inputs(case: BenchCase, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    query = rng.normal(size=(case.batch, case.heads, case.q_len, case.dim)).astype(np.float32)
    key = rng.normal(size=(case.batch, case.heads, case.kv_len, case.dim)).astype(np.float32)
    value = rng.normal(size=(case.batch, case.heads, case.kv_len, case.dim)).astype(np.float32)
    logits = rng.normal(size=(case.batch, case.vocab)).astype(np.float32)
    return query, key, value, logits


def timed_ms(fn: Callable[[], object], repeats: int) -> tuple[float, float]:
    durations: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        durations.append((time.perf_counter() - started) * 1000)
    return statistics.fmean(durations), sorted(durations)[int(0.95 * (len(durations) - 1))]


def run_case(case: BenchCase, backend: str, seed: int) -> dict[str, object]:
    query, key, value, logits = make_inputs(case, seed)
    reference = attention_numpy(query, key, value)
    sample_tokens = sample_top_p_numpy(logits)

    if backend == "torch":
        if not torch_available():
            raise RuntimeError("torch backend requested but torch is not installed")
        attention_fn = lambda: attention_torch(query, key, value)
        candidate = attention_torch(query, key, value)
    elif backend in {"numpy", "flashinfer-auto"}:
        attention_fn = lambda: attention_numpy(query, key, value)
        candidate = attention_numpy(query, key, value)
    else:
        raise ValueError(f"unknown backend: {backend}")

    attention_mean_ms, attention_p95_ms = timed_ms(attention_fn, case.repeats)
    sample_mean_ms, sample_p95_ms = timed_ms(lambda: sample_top_p_numpy(logits), case.repeats)

    return {
        "case": case.name,
        "backend": backend,
        "batch": case.batch,
        "heads": case.heads,
        "q_len": case.q_len,
        "kv_len": case.kv_len,
        "dim": case.dim,
        "vocab": case.vocab,
        "repeats": case.repeats,
        "attention_mean_ms": attention_mean_ms,
        "attention_p95_ms": attention_p95_ms,
        "sampling_mean_ms": sample_mean_ms,
        "sampling_p95_ms": sample_p95_ms,
        "max_abs_error_vs_numpy": max_abs_error(reference, candidate),
        "sample_token_checksum": int(np.sum(sample_tokens)),
    }


def write_report(out_dir: Path, rows: list[dict[str, object]], metadata: dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps({"metadata": metadata, "results": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# FlashInfer-Style Kernel Microbenchmark",
        "",
        "## Environment",
        "",
    ]
    for key, value in metadata.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Case | Backend | Shape | Attention mean ms | Attention p95 ms | Sampling mean ms | Error vs NumPy |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        shape = f"b={row['batch']} h={row['heads']} q={row['q_len']} kv={row['kv_len']} d={row['dim']}"
        lines.append(
            f"| {row['case']} | {row['backend']} | {shape} | "
            f"{float(row['attention_mean_ms']):.3f} | "
            f"{float(row['attention_p95_ms']):.3f} | "
            f"{float(row['sampling_mean_ms']):.3f} | "
            f"{float(row['max_abs_error_vs_numpy']):.6f} |"
        )
    (out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run attention and sampling microbenchmarks.")
    parser.add_argument("--backend", choices=["numpy", "torch", "flashinfer-auto"], default="numpy")
    parser.add_argument("--cases", default="tiny,small")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="reports/flashinfer-bench")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.backend == "flashinfer-auto" and not flashinfer_available():
        print("flashinfer is not installed; using NumPy reference backend for smoke-test results.")

    cases = parse_cases(args.cases)
    rows = [run_case(case, args.backend, args.seed + index) for index, case in enumerate(cases)]
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch_available": torch_available(),
        "flashinfer_available": flashinfer_available(),
    }
    write_report(Path(args.out), rows, metadata)
    print(json.dumps({"metadata": metadata, "results": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


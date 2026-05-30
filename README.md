# FlashInfer Kernel Bench

[![CI](https://github.com/Gloria72/flashinfer-kernel-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/Gloria72/flashinfer-kernel-bench/actions/workflows/ci.yml)

Small benchmark harness for attention and sampling code paths used in LLM decoding.

The first version is deliberately plain: NumPy reference kernels, optional Torch comparison, and a report format that can be reused on a CUDA box. I wrote it this way so correctness and shapes are pinned down before swapping in faster kernels.

## What is here

- Implements reference scaled dot-product attention in NumPy.
- Implements deterministic top-p representative sampling for repeatable runs.
- Provides tiny/small/medium benchmark cases across batch size, heads, sequence length, head dimension, and vocabulary size.
- Optionally compares against Torch `scaled_dot_product_attention`.
- Reports correctness error versus the NumPy reference.
- Writes JSON, CSV, and Markdown reports.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Run the local NumPy benchmark:

```bash
python -m flashinfer_kernel_bench.benchmark \
  --backend numpy \
  --cases tiny,small \
  --out reports/flashinfer-bench
```

If Torch is installed:

```bash
python -m flashinfer_kernel_bench.benchmark \
  --backend torch \
  --cases tiny,small \
  --out reports/torch-bench
```

## FlashInfer Path

On a CUDA machine, install FlashInfer following the upstream instructions, then run:

```bash
python -m flashinfer_kernel_bench.benchmark \
  --backend flashinfer-auto \
  --cases tiny,small,medium \
  --out reports/flashinfer-gpu
```

Right now `flashinfer-auto` records whether FlashInfer is installed and falls back to the NumPy path. I kept it conservative because FlashInfer APIs move across versions; the reference path makes it easier to add a specific kernel call without changing the report contract.

## Notes

- The benchmark is not trying to replace Nsight or a full serving profiler.
- It is useful for quick shape sweeps and sanity checks.
- On GPU, the important follow-up is to add hardware info, CUDA version, dtype, and warmup runs to the report.
- The correctness check stays in the loop even when a faster backend is added.

## Example Result

See [examples/results.md](examples/results.md).

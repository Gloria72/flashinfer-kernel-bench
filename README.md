# FlashInfer Kernel Bench

A correctness-first microbenchmark suite for LLM decode attention and sampling kernels.

This project is designed to connect service-level LLM serving metrics to lower-level GPU kernel behavior. It runs locally with a NumPy reference backend, can compare with Torch when installed, and records whether FlashInfer is available for later GPU-backed runs.

## What It Does

- Implements reference scaled dot-product attention in NumPy.
- Implements deterministic top-p representative sampling for repeatable benchmark output.
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

Run the local fallback benchmark:

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

The current `flashinfer-auto` mode records FlashInfer availability and keeps the NumPy reference path as the safe fallback. This gives a reliable baseline before wiring in specific FlashInfer APIs for the target kernel/version.

## Interview Narrative

- Service-level decode latency depends on attention, sampling, memory movement, and scheduling.
- Correctness comes before speed: every optimized backend should be compared against a reference.
- Shape matters: batch size, context length, head dimension, dtype, and vocabulary size change bottlenecks.
- A benchmark report should include hardware, Python/package versions, sequence shapes, latency distribution, and correctness error.

## Example Result

See [examples/results.md](examples/results.md).


# FlashInfer-Style Kernel Microbenchmark

This example was produced on a local CPU fallback path with NumPy. It validates the benchmark harness and correctness check before moving to a CUDA/FlashInfer machine.

## Environment

- `platform`: macOS arm64
- `backend`: numpy
- `torch_available`: false
- `flashinfer_available`: false

## Results

| Case | Backend | Shape | Attention mean ms | Attention p95 ms | Sampling mean ms | Error vs NumPy |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| tiny | numpy | b=1 h=2 q=1 kv=64 d=32 | 0.010 | 0.010 | 0.097 | 0.000000 |


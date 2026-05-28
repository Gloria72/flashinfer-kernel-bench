from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchCase:
    name: str
    batch: int
    heads: int
    q_len: int
    kv_len: int
    dim: int
    vocab: int
    repeats: int


CASES: dict[str, BenchCase] = {
    "tiny": BenchCase("tiny", batch=1, heads=2, q_len=1, kv_len=64, dim=32, vocab=1024, repeats=8),
    "small": BenchCase("small", batch=2, heads=4, q_len=1, kv_len=256, dim=64, vocab=4096, repeats=8),
    "medium": BenchCase("medium", batch=4, heads=8, q_len=1, kv_len=512, dim=64, vocab=8192, repeats=5),
}


def parse_cases(raw: str) -> list[BenchCase]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [name for name in names if name not in CASES]
    if unknown:
        raise ValueError(f"unknown cases: {', '.join(unknown)}")
    return [CASES[name] for name in names]


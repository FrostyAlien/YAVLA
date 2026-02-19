"""Benchmark dataloader throughput and per-transform latency."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pyarrow
import torch
from torch.utils.data import DataLoader, Dataset


def _hardware_info() -> dict[str, Any]:
    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3) if hasattr(os, "sysconf") else None
    return {
        "torch_version": torch.__version__,
        "pyarrow_version": pyarrow.__version__,
        "python_version": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "gpu": gpu,
        "ram_gb": round(ram_gb, 1) if ram_gb else None,
    }


class SyntheticDataset(Dataset[dict[str, Any]]):
    """Random tensors for benchmarking transforms + collation."""

    def __init__(self, size: int = 10_000, action_dim: int = 7, state_dim: int = 14) -> None:
        self.size = size
        self.action_dim = action_dim
        self.state_dim = state_dim

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "action": torch.randn(self.action_dim),
            "observation.state": torch.randn(self.state_dim),
            "index": index,
            "timestamp": float(index) / 30.0,
        }


class _TimingWrapper:
    """Wrap a transform to record per-call latency."""

    def __init__(self, transform: Any, name: str) -> None:
        self._transform = transform
        self.name = name
        self.latencies_ns: list[int] = []

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter_ns()
        result = self._transform(sample)
        self.latencies_ns.append(time.perf_counter_ns() - t0)
        return result


def _wrap_transforms(transform: Any) -> list[_TimingWrapper]:
    """Wrap composed transforms with timing. Returns list of wrappers."""
    wrappers: list[_TimingWrapper] = []
    if hasattr(transform, "transforms"):
        for t in transform.transforms:
            w = _TimingWrapper(t, type(t).__name__)
            wrappers.append(w)
        transform.transforms = tuple(wrappers)
    return wrappers


def _measure_throughput(loader: DataLoader[Any], *, warmup: int, batch_size: int) -> dict[str, float]:
    """Iterate loader, skip warmup, return samples/sec median + IQR."""
    batch_times: list[float] = []
    for i, _ in enumerate(loader):
        if i < warmup:
            continue
        if i == warmup:
            t0 = time.perf_counter()
            continue
        t1 = time.perf_counter()
        batch_times.append(t1 - t0)
        t0 = t1
        if i >= warmup + 100:
            break

    if not batch_times:
        return {"samples_per_sec_median": 0.0, "samples_per_sec_iqr": 0.0}

    rates = np.array([batch_size / t for t in batch_times])
    q25, median, q75 = np.percentile(rates, [25, 50, 75])
    return {"samples_per_sec_median": float(median), "samples_per_sec_iqr": float(q75 - q25)}


def _transform_stats(wrappers: list[_TimingWrapper]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for w in wrappers:
        if not w.latencies_ns:
            continue
        arr = np.array(w.latencies_ns)
        result[w.name] = {"median": int(np.median(arr)), "p95": int(np.percentile(arr, 95))}
    return result


def _run_synthetic(args: argparse.Namespace) -> dict[str, Any]:
    from yavla.data.transforms import NormalizeTransform, compose

    stats = {
        "action": {"mean": torch.zeros(7).tolist(), "std": torch.ones(7).tolist()},
        "observation.state": {"mean": torch.zeros(14).tolist(), "std": torch.ones(14).tolist()},
    }
    transform = compose(NormalizeTransform(stats=stats, mode="z-score"))
    wrappers = _wrap_transforms(transform)

    dataset = SyntheticDataset()
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers, collate_fn=_synthetic_collate(transform)
    )

    throughput = _measure_throughput(loader, warmup=args.warmup_batches, batch_size=args.batch_size)
    return {
        "config": {
            "repo_id": None,
            "backend": "synthetic",
            "num_workers": args.workers,
            "batch_size": args.batch_size,
            "synthetic": True,
        },
        "throughput": throughput,
        "per_transform_latency_ns": _transform_stats(wrappers),
        "hardware": _hardware_info(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _synthetic_collate(transform: Any) -> Any:
    """Collate that applies transform per-sample (simulates real pipeline)."""

    def collate(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [transform(s) for s in batch]

    return collate


def _run_real(args: argparse.Namespace) -> list[dict[str, Any]]:
    from yavla.data.factory import DataConfig, create_dataloader

    results = []
    for backend in args.backends.split(","):
        backend = backend.strip()
        config = DataConfig(
            repo_id=args.repo_id,
            backend=backend,  # type: ignore[arg-type]
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
        loader = create_dataloader(config)

        # Wrap transforms if num_workers=0
        wrappers: list[_TimingWrapper] = []
        if args.workers == 0:
            ds = getattr(loader, "dataset", None)
            t = getattr(ds, "transforms", None) or getattr(ds, "transform", None)
            if t is not None:
                wrappers = _wrap_transforms(t)

        throughput = _measure_throughput(loader, warmup=args.warmup_batches, batch_size=args.batch_size)
        results.append(
            {
                "config": {
                    "repo_id": args.repo_id,
                    "backend": backend,
                    "num_workers": args.workers,
                    "batch_size": args.batch_size,
                    "synthetic": False,
                },
                "throughput": throughput,
                "per_transform_latency_ns": _transform_stats(wrappers),
                "hardware": _hardware_info(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark YAVLA dataloader throughput")
    parser.add_argument("--repo-id", help="Dataset repo ID")
    parser.add_argument("--backends", default="lazy,streaming", help="Comma-separated backends")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", help="JSON output path (default: stdout)")
    args = parser.parse_args()

    if not args.synthetic and not args.repo_id:
        parser.error("--repo-id is required unless --synthetic is specified")

    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            results = [_run_synthetic(args)] if args.synthetic else _run_real(args)
        prof.export_chrome_trace("bench_trace.json")
        print("Profiler trace written to bench_trace.json", file=sys.stderr)
    else:
        results = [_run_synthetic(args)] if args.synthetic else _run_real(args)

    output = json.dumps(results if len(results) > 1 else results[0], indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output + "\n")
    else:
        print(output)


if __name__ == "__main__":
    main()

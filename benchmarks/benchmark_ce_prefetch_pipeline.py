# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive integers"
        )
    return values


def _event_pair() -> tuple[torch.cuda.Event, torch.cuda.Event]:
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _measure_copy(
    src: torch.Tensor,
    dst: torch.Tensor,
    stream: torch.cuda.Stream,
    repeat: int,
) -> float:
    times = []
    for _ in range(repeat):
        start, end = _event_pair()
        with torch.cuda.device(dst.device), torch.cuda.stream(stream):
            start.record()
            dst.copy_(src, non_blocking=True)
            end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def _measure_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    stream: torch.cuda.Stream,
    iters: int,
    repeat: int,
) -> float:
    times = []
    for _ in range(repeat):
        start, end = _event_pair()
        with torch.cuda.device(a.device), torch.cuda.stream(stream):
            start.record()
            for _ in range(iters):
                torch.mm(a, b, out=out)
            end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def _measure_overlap(
    src: torch.Tensor,
    dst: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    copy_stream: torch.cuda.Stream,
    compute_stream: torch.cuda.Stream,
    iters: int,
    repeat: int,
) -> tuple[float, float, float]:
    wall_times = []
    copy_times = []
    compute_times = []
    device = dst.device

    for _ in range(repeat):
        torch.cuda.synchronize(device)
        default_stream = torch.cuda.default_stream(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        copy_start, copy_end = _event_pair()
        compute_start, compute_end = _event_pair()

        start.record(default_stream)
        copy_stream.wait_event(start)
        compute_stream.wait_event(start)

        with torch.cuda.device(device), torch.cuda.stream(copy_stream):
            copy_start.record()
            dst.copy_(src, non_blocking=True)
            copy_end.record()

        with torch.cuda.device(device), torch.cuda.stream(compute_stream):
            compute_start.record()
            for _ in range(iters):
                torch.mm(a, b, out=out)
            compute_end.record()

        default_stream.wait_event(copy_end)
        default_stream.wait_event(compute_end)
        end.record(default_stream)
        end.synchronize()

        wall_times.append(start.elapsed_time(end))
        copy_times.append(copy_start.elapsed_time(copy_end))
        compute_times.append(compute_start.elapsed_time(compute_end))

    return (
        statistics.median(wall_times),
        statistics.median(copy_times),
        statistics.median(compute_times),
    )


def _record_environment(src_device: int, dst_device: int) -> dict[str, Any]:
    return {
        "timestamp": _now(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "src_device": src_device,
        "dst_device": dst_device,
        "src_name": torch.cuda.get_device_name(src_device),
        "dst_name": torch.cuda.get_device_name(dst_device),
        "p2p_enabled": torch.cuda.can_device_access_peer(dst_device, src_device),
    }


def _blocked(reason: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "timestamp": _now(),
        "status": "blocked",
        "reason": reason,
        "src_device": args.src_device,
        "dst_device": args.dst_device,
    }


def _write_row(row: dict[str, Any], output: Path | None) -> None:
    line = json.dumps(row, sort_keys=True)
    print(line, flush=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        return [_blocked("requires CUDA", args)]
    if torch.cuda.device_count() < 2:
        return [_blocked("requires at least two CUDA devices", args)]
    if args.src_device == args.dst_device:
        return [_blocked("source and destination devices must differ", args)]
    if not torch.cuda.can_device_access_peer(args.dst_device, args.src_device):
        return [
            _blocked("destination device cannot access source device with P2P", args)
        ]

    dtype = DTYPES[args.dtype]
    elem_size = torch.empty((), dtype=dtype).element_size()
    env = _record_environment(args.src_device, args.dst_device)
    rows = []

    with torch.cuda.device(args.src_device):
        source_buffers = {
            copy_mib: torch.empty(
                (copy_mib * 1024 * 1024) // elem_size,
                device=args.src_device,
                dtype=dtype,
            )
            for copy_mib in args.copy_mib
        }
        for tensor in source_buffers.values():
            tensor.fill_(1)

    with torch.cuda.device(args.dst_device):
        dest_buffers = {
            copy_mib: torch.empty_like(source_buffers[copy_mib], device=args.dst_device)
            for copy_mib in args.copy_mib
        }
        gemm_inputs = {
            size: (
                torch.randn(size, size, device=args.dst_device, dtype=dtype),
                torch.randn(size, size, device=args.dst_device, dtype=dtype),
                torch.empty(size, size, device=args.dst_device, dtype=dtype),
            )
            for size in args.gemm_size
        }
        copy_stream = torch.cuda.Stream(device=args.dst_device)
        compute_stream = torch.cuda.Stream(device=args.dst_device)

    for copy_mib, src in source_buffers.items():
        dst = dest_buffers[copy_mib]
        for gemm_size, (a, b, out) in gemm_inputs.items():
            for gemm_iters in args.gemm_iters:
                for _ in range(args.warmup):
                    with (
                        torch.cuda.device(args.dst_device),
                        torch.cuda.stream(copy_stream),
                    ):
                        dst.copy_(src, non_blocking=True)
                    with (
                        torch.cuda.device(args.dst_device),
                        torch.cuda.stream(compute_stream),
                    ):
                        for _ in range(gemm_iters):
                            torch.mm(a, b, out=out)
                torch.cuda.synchronize(args.dst_device)

                copy_ms = _measure_copy(src, dst, copy_stream, args.repeat)
                compute_ms = _measure_gemm(
                    a, b, out, compute_stream, gemm_iters, args.repeat
                )
                overlap_ms, overlap_copy_ms, overlap_compute_ms = _measure_overlap(
                    src,
                    dst,
                    a,
                    b,
                    out,
                    copy_stream,
                    compute_stream,
                    gemm_iters,
                    args.repeat,
                )

                serial_ms = copy_ms + compute_ms
                hidden = (serial_ms - overlap_ms) / copy_ms if copy_ms > 0 else 0.0
                compute_slowdown = (
                    (overlap_compute_ms - compute_ms) / compute_ms * 100
                    if compute_ms > 0
                    else 0.0
                )
                copy_gib = copy_mib / 1024
                row = {
                    **env,
                    "status": "pass",
                    "dtype": args.dtype,
                    "copy_mib": copy_mib,
                    "gemm_size": gemm_size,
                    "gemm_iters": gemm_iters,
                    "copy_ms": copy_ms,
                    "copy_gib_s": copy_gib / (copy_ms / 1000),
                    "compute_ms": compute_ms,
                    "overlap_wall_ms": overlap_ms,
                    "overlap_copy_ms": overlap_copy_ms,
                    "overlap_compute_ms": overlap_compute_ms,
                    "serial_ms": serial_ms,
                    "speedup_vs_serial": serial_ms / overlap_ms,
                    "copy_hidden_fraction": max(0.0, min(1.0, hidden)),
                    "compute_slowdown_pct": compute_slowdown,
                    "warmup": args.warmup,
                    "repeat": args.repeat,
                }
                rows.append(row)
    return rows


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure copy-engine prefetch overlap with local GPU compute.",
    )
    parser.add_argument("--src-device", type=int, default=0)
    parser.add_argument("--dst-device", type=int, default=1)
    parser.add_argument("--copy-mib", type=_parse_int_list, default=[64, 256, 512])
    parser.add_argument("--gemm-size", type=_parse_int_list, default=[4096, 8192])
    parser.add_argument("--gemm-iters", type=_parse_int_list, default=[1, 8])
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    for row in _benchmark(args):
        _write_row(row, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

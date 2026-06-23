# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_int_list(value: str) -> list[int]:
    items = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not items or any(item <= 0 for item in items):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return items


def _token_groups(
    batch_size: int, completion_len: int, vocab_size: int
) -> list[list[int]]:
    return [
        [(row + col) % vocab_size for col in range((row % completion_len) + 1)]
        for row in range(batch_size)
    ]


def _current_gpu_row_staging(
    groups: list[list[int]],
    *,
    completion_len: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids = torch.zeros(
        (len(groups), completion_len), device=device, dtype=torch.long
    )
    mask = torch.zeros((len(groups), completion_len), device=device, dtype=torch.bool)
    for row, group in enumerate(groups):
        clipped = [int(token) % vocab_size for token in group[:completion_len]]
        if not clipped:
            continue
        values = torch.tensor(clipped, device=device, dtype=torch.long)
        token_ids[row, : values.numel()] = values
        mask[row, : values.numel()] = True
    return token_ids, mask


def _packed_cpu_staging(
    groups: list[list[int]],
    *,
    completion_len: int,
    vocab_size: int,
    device: torch.device,
    pinned: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_cpu = torch.zeros((len(groups), completion_len), dtype=torch.long)
    mask_cpu = torch.zeros((len(groups), completion_len), dtype=torch.bool)
    for row, group in enumerate(groups):
        clipped = [int(token) % vocab_size for token in group[:completion_len]]
        if not clipped:
            continue
        count = len(clipped)
        token_cpu[row, :count] = torch.as_tensor(clipped, dtype=torch.long)
        mask_cpu[row, :count] = True
    use_pinned = pinned and device.type == "cuda"
    if use_pinned:
        token_cpu = token_cpu.pin_memory()
        mask_cpu = mask_cpu.pin_memory()
    return token_cpu.to(device, non_blocking=use_pinned), mask_cpu.to(
        device, non_blocking=use_pinned
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor]], device: torch.device, args
) -> float:
    for _ in range(args.warmup):
        fn()
        _sync(device)
    times = []
    for _ in range(args.repeat):
        started = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - started) * 1000)
    return statistics.median(times)


def _write(row: dict, output: Path | None) -> None:
    line = json.dumps(row, sort_keys=True)
    print(line, flush=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _blocked(reason: str, args) -> dict:
    return {
        "timestamp": _now(),
        "status": "blocked",
        "reason": reason,
        "device": args.device,
    }


def _run(args) -> list[dict]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        return [_blocked("CUDA is not available", args)]

    rows = []
    for batch_size in args.batch_size:
        groups = _token_groups(batch_size, args.completion_len, args.vocab_size)
        current_ms = _measure(
            lambda groups=groups: _current_gpu_row_staging(
                groups,
                completion_len=args.completion_len,
                vocab_size=args.vocab_size,
                device=device,
            ),
            device,
            args,
        )
        packed_ms = _measure(
            lambda groups=groups: _packed_cpu_staging(
                groups,
                completion_len=args.completion_len,
                vocab_size=args.vocab_size,
                device=device,
                pinned=False,
            ),
            device,
            args,
        )
        pinned_ms = _measure(
            lambda groups=groups: _packed_cpu_staging(
                groups,
                completion_len=args.completion_len,
                vocab_size=args.vocab_size,
                device=device,
                pinned=True,
            ),
            device,
            args,
        )
        rows.append(
            {
                "timestamp": _now(),
                "status": "pass",
                "device": str(device),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "batch_size": batch_size,
                "completion_len": args.completion_len,
                "vocab_size": args.vocab_size,
                "current_gpu_row_ms": current_ms,
                "packed_cpu_ms": packed_ms,
                "packed_pinned_ms": pinned_ms,
                "packed_cpu_speedup": current_ms / packed_ms,
                "packed_pinned_speedup": current_ms / pinned_ms,
                "warmup": args.warmup,
                "repeat": args.repeat,
            }
        )
    return rows


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark rollout token payload staging into training tensors.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--batch-size", type=_parse_int_list, default=[1024, 4096, 8192]
    )
    parser.add_argument("--completion-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=128000)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.completion_len <= 0:
        parser.error("--completion-len must be positive")
    if args.vocab_size <= 0:
        parser.error("--vocab-size must be positive")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    for row in _run(args):
        _write(row, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

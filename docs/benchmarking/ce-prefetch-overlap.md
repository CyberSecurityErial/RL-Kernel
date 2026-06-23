# Copy-Engine Prefetch Overlap Experiment Log

Date: 2026-06-23 UTC
Branch: `cse/p2p-copy-overlap-benchmark`

## Question

Can peer-to-peer GPU copies be issued on the copy engine while the destination GPU keeps its SMs busy with compute?

The useful production pattern would be a double-buffered pipeline:

1. SMs compute the current weight, KV, or activation shard.
2. A separate CUDA stream prefetches the next shard from another GPU.
3. The next compute step waits only on the staged shard it actually consumes.

This is not an all-reduce experiment. Reductions still need NCCL/NVLS or SM-side arithmetic. This probe is for data staging work where a byte copy can be overlapped with local compute.

## Environment

Observed by the benchmark process:

- PyTorch: `2.11.0+cu128`
- CUDA runtime reported by PyTorch: `12.8`
- Source GPU: `0`
- Destination GPU: `1`
- Reported device name: `NVIDIA L20X`
- P2P access: `true`
- dtype: `bfloat16`

The machine was identified by the operator as an H200 node, but the container reports the device name above. The experiment log records what the runtime reported.

## Benchmark Method

The benchmark compares three timings for each shape:

- `copy_ms`: device-to-device P2P copy from source GPU to destination GPU.
- `compute_ms`: GEMM on the destination GPU.
- `overlap_wall_ms`: wall time when copy and GEMM are launched together on separate streams.

Derived metrics:

- `copy_hidden_fraction = (copy_ms + compute_ms - overlap_wall_ms) / copy_ms`
- `compute_slowdown_pct = (overlap_compute_ms - compute_ms) / compute_ms * 100`
- `speedup_vs_serial = (copy_ms + compute_ms) / overlap_wall_ms`

A useful case has high `copy_hidden_fraction` and low `compute_slowdown_pct`.

## Failed Or Noisy Steps

These are recorded so the setup mistakes are not repeated.

| Step | Command | Result | Action |
| --- | --- | --- | --- |
| Patch application | `apply_patch` | Failed with `bwrap: Creating new namespace failed ... ENOSPC` | Used a controlled `python3` write inside the repo instead. |
| Python launcher | `python - <<'PY' ...` | Failed with `python: command not found` | Used `/usr/bin/python3` for file writes and `.codex-nightly/envs/issue112-py312/bin/python` for benchmark runs. |
| Formatting check | `.codex-nightly/envs/issue112-py312/bin/python -m black --check benchmarks/benchmark_ce_prefetch_pipeline.py` | Failed because Black tried a newer target syntax than Python 3.12 could safety-check, and the file needed formatting | Ran Black with `--target-version py312`. |
| Tiny smoke shape | `--copy-mib 1 --gemm-size 1024 --gemm-iters 1 --repeat 2` | Passed, but produced unstable performance ratios because startup overhead dominates | Kept it only as a CLI/CUDA smoke test, not as performance evidence. |

## Validation Commands

```bash
.codex-nightly/envs/issue112-py312/bin/python -m py_compile \
  benchmarks/benchmark_ce_prefetch_pipeline.py

.codex-nightly/envs/issue112-py312/bin/python -m ruff check \
  benchmarks/benchmark_ce_prefetch_pipeline.py

.codex-nightly/envs/issue112-py312/bin/python -m black --check --target-version py312 \
  benchmarks/benchmark_ce_prefetch_pipeline.py
```

Result: all passed after formatting.

Smoke command:

```bash
.codex-nightly/envs/issue112-py312/bin/python \
  benchmarks/benchmark_ce_prefetch_pipeline.py \
  --src-device 0 \
  --dst-device 1 \
  --copy-mib 1 \
  --gemm-size 1024 \
  --gemm-iters 1 \
  --warmup 1 \
  --repeat 2
```

Result: passed. The row reported `p2p_enabled=true` and `status=pass`.

## Main Sweep

Command:

```bash
.codex-nightly/envs/issue112-py312/bin/python \
  benchmarks/benchmark_ce_prefetch_pipeline.py \
  --src-device 0 \
  --dst-device 1 \
  --copy-mib 64,256,512 \
  --gemm-size 4096,8192 \
  --gemm-iters 1,8 \
  --warmup 3 \
  --repeat 10 \
  --output .codex-nightly/artifacts/ce_prefetch_probe.jsonl
```

| Copy MiB | GEMM | Iters | Copy ms | Compute ms | Overlap ms | Hidden % | Compute slowdown % | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 4096 | 1 | 0.190 | 0.189 | 0.261 | 61.8 | 2.5 | 1.45 |
| 64 | 4096 | 8 | 0.189 | 1.426 | 1.499 | 61.1 | 0.5 | 1.08 |
| 64 | 8192 | 1 | 0.190 | 1.443 | 1.513 | 62.9 | 0.3 | 1.08 |
| 64 | 8192 | 8 | 0.188 | 13.377 | 13.417 | 78.8 | -0.2 | 1.01 |
| 256 | 4096 | 1 | 0.697 | 0.234 | 0.720 | 30.3 | 2.3 | 1.29 |
| 256 | 4096 | 8 | 0.696 | 1.509 | 1.538 | 95.8 | -2.6 | 1.43 |
| 256 | 8192 | 1 | 0.696 | 1.438 | 1.518 | 88.5 | 0.9 | 1.41 |
| 256 | 8192 | 8 | 0.695 | 12.951 | 13.390 | 36.9 | 2.8 | 1.02 |
| 512 | 4096 | 1 | 1.369 | 0.236 | 1.395 | 15.4 | 2.1 | 1.15 |
| 512 | 4096 | 8 | 1.370 | 1.456 | 1.533 | 94.4 | 0.7 | 1.84 |
| 512 | 8192 | 1 | 1.369 | 1.450 | 1.539 | 93.5 | 1.6 | 1.83 |
| 512 | 8192 | 8 | 1.369 | 12.266 | 13.653 | 0.0 | 10.7 | 1.00 |

## Boundary Repeats

The first sweep showed that very large copies can sometimes slow a large GEMM enough to remove most of the gain. These cases were repeated with more warmup and repeats.

Command:

```bash
.codex-nightly/envs/issue112-py312/bin/python \
  benchmarks/benchmark_ce_prefetch_pipeline.py \
  --src-device 0 \
  --dst-device 1 \
  --copy-mib 256,512 \
  --gemm-size 8192 \
  --gemm-iters 8 \
  --warmup 5 \
  --repeat 20 \
  --output .codex-nightly/artifacts/ce_prefetch_boundary_repeat.jsonl
```

| Copy MiB | GEMM | Iters | Copy ms | Compute ms | Overlap ms | Hidden % | Compute slowdown % | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 8192 | 8 | 0.699 | 13.092 | 13.462 | 47.0 | 2.3 | 1.02 |
| 512 | 8192 | 8 | 1.373 | 12.906 | 13.835 | 32.3 | 6.7 | 1.03 |

## Promising-Case Repeats

Command:

```bash
.codex-nightly/envs/issue112-py312/bin/python \
  benchmarks/benchmark_ce_prefetch_pipeline.py \
  --src-device 0 \
  --dst-device 1 \
  --copy-mib 512 \
  --gemm-size 4096,8192 \
  --gemm-iters 8,1 \
  --warmup 5 \
  --repeat 20 \
  --output .codex-nightly/artifacts/ce_prefetch_good_cases_repeat.jsonl
```

| Copy MiB | GEMM | Iters | Copy ms | Compute ms | Overlap ms | Hidden % | Compute slowdown % | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 4096 | 8 | 1.369 | 1.427 | 1.633 | 84.9 | 9.8 | 1.71 |
| 512 | 4096 | 1 | 1.368 | 0.219 | 1.393 | 14.2 | -5.5 | 1.14 |
| 512 | 8192 | 8 | 1.370 | 12.956 | 13.352 | 71.1 | 2.4 | 1.07 |
| 512 | 8192 | 1 | 1.369 | 1.441 | 1.736 | 78.4 | 16.1 | 1.62 |

## Takeaways

1. P2P copy bandwidth is high enough to matter: the measured 256-512 MiB copies were roughly 358-365 GiB/s.
2. Copy-engine overlap is real: when compute is near the copy time, the overlapped wall time is much closer to `max(copy, compute)` than to `copy + compute`.
3. CE does not consume SMs, but the copy is not free. It still competes for destination-side memory-system resources. The repeated 512 MiB cases showed compute slowdown from about 2% to 16% depending on GEMM shape.
4. Small compute windows cannot hide large copies. The 512 MiB + 4096x4096x1 case hid only about 14-15% of the copy.
5. Huge compute windows do not automatically make large copies ideal. Some 8192x8192x8 repeats had only 2-3% net speedup because compute slowed down.

## Optimization Direction

The next production-facing experiment should not issue one huge copy and hope the SM work hides it. The better target is a chunked, double-buffered staging path:

- Stage the next shard with P2P copy on a side stream.
- Keep chunk sizes tunable, likely starting around 128-256 MiB on this node.
- Use events so compute waits only for the chunk it consumes.
- Measure both wall-time speedup and compute slowdown.

Good candidate integration points in this repo:

- Weight or rollout weight staging around the existing bridge benchmark.
- TP-aware reward/critic forward paths when a shard can be localized before compute.
- KV or activation staging only if the access pattern is contiguous enough to make copy-only prefetch meaningful.

This is not yet evidence for accelerating all-reduce. The evidence supports copy-style data movement that can be prefetched ahead of local compute.

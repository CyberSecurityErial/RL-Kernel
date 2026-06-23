# Benchmarking

RL-Kernel benchmarks track operator latency, memory behavior, and dispatch overhead.

Current benchmark entry points:

```bash
python scripts/run_profile_suite.py --smoke
python scripts/run_profile_suite.py --output reports/profile.csv
python benchmarks/profiler.py --format json --output reports/profile.json
python benchmarks/benchmark_sampling.py
python benchmarks/benchmark_grpo_op.py
python benchmarks/benchmark_ce_prefetch_pipeline.py --copy-mib 64,256,512 --gemm-size 4096,8192
python scripts/run_perf.py
```

The automated profiler records one row per workload shape with:

- `tokens_per_sec`: active tokens divided by median latency.
- `tflops`: estimated operator FLOPs divided by median latency.
- `peak_vram_gb`: CUDA peak allocated memory during the measured run.
- `gpu_*`: detected device name, architecture, backend, driver, and memory.
- `status`: `pass`, `blocked`, or `oom`.

Useful presets:

```bash
# CPU-friendly validation for CI or local development.
python scripts/run_profile_suite.py --smoke --workloads logp-native

# CUDA logprob profiling with native and fused candidates.
python scripts/run_profile_suite.py \
  --device cuda \
  --dtype float16 \
  --batch-sizes 8,16,32 \
  --seq-lens 128,512 \
  --vocab-sizes 4096,128256 \
  --workloads logp-native,logp-fused \
  --output reports/logp_profile.csv

# Sampling baseline profiling.
python scripts/run_profile_suite.py \
  --workloads sampling-native \
  --batch-sizes 64,128,256 \
  --vocab-sizes 128256 \
  --top-k 50 \
  --top-p 0.9
```

When adding a new operator, document the benchmark command on the operator page and keep
the tested shapes close to the target RL workload.

## Copy-Engine Prefetch Overlap

`benchmarks/benchmark_ce_prefetch_pipeline.py` measures whether peer-to-peer GPU copies can be
hidden behind local GEMM work on another CUDA stream. It is meant for weight, KV, or activation
staging experiments where the next shard can be prefetched while the current shard is computed.

Example:

```bash
python benchmarks/benchmark_ce_prefetch_pipeline.py \
  --src-device 0 \
  --dst-device 1 \
  --copy-mib 64,256,512 \
  --gemm-size 4096,8192 \
  --gemm-iters 1,8 \
  --output reports/ce_prefetch.jsonl
```

Each JSONL row reports copy-only latency, compute-only latency, overlapped wall time,
`copy_hidden_fraction`, `compute_slowdown_pct`, and `speedup_vs_serial`. The useful case is a
high hidden fraction with low compute slowdown; small compute windows are expected to expose most
of the copy time.

Detailed local experiment notes are in [ce-prefetch-overlap.md](ce-prefetch-overlap.md).

## Adding Workloads

Profiler workloads are registered in `benchmarks/profiler.py` through
`WORKLOAD_REGISTRY`. To add an operator benchmark:

1. Add a small workload runner that builds deterministic inputs and calls the relevant
   `PerformanceProfiler.profile_*` method.
2. Register the runner under a stable CLI name in `WORKLOAD_REGISTRY`.
3. Add a focused test in `tests/test_profiler.py` that verifies the workload can be
   selected through `--workloads`.
4. Document a representative command and shape preset for the operator.

Workloads that require CUDA should report `status=blocked` with a clear note when the
requested device cannot run them, so CPU smoke validation still produces useful reports.

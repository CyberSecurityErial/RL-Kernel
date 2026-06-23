# Hardware Benchmark Dashboard

This page defines the reporting format for reproducible RL-Kernel hardware
benchmarks. It is a dashboard scaffold, not a performance claim. Rows marked
`pending` have not been measured yet.

## Reporting Rules

Every published result must include:

- command line;
- RL-Kernel commit;
- Python and PyTorch versions;
- GPU model, memory, backend, driver or runtime, and compute capability;
- workload name, dtype, shape, and selected backend;
- latency, throughput, peak VRAM, and status;
- notes for blocked, fallback, or out-of-memory runs.

A fallback backend result must not be reported as a fused-kernel result. If the
runtime selects PyTorch, native CUDA, or another fallback, record that backend
explicitly in the `Backend` column.

## Status Definitions

| Status | Meaning |
| --- | --- |
| `pass` | The workload completed. Verify the selected backend before reporting an optimized result. |
| `blocked` | The workload could not run because hardware, compiled extensions, or optional dependencies were unavailable. |
| `oom` | The workload exceeded available memory. |
| `pending` | No measurement has been collected yet. |

## Profiler Field Mapping

`benchmarks/profiler.py` writes JSON and CSV reports with the same fields used by
this dashboard.

| Dashboard column | Profiler field |
| --- | --- |
| Environment | `gpu_target.name`, `gpu_target.architecture`, `gpu_target.backend` |
| Backend | `benchmark_name` and backend notes from the workload |
| Batch | `batch_size` |
| Sequence length | `seq_len` |
| Vocabulary | `vocab_size` |
| Latency | `latency_ms` |
| Tokens/s | `tokens_per_sec` |
| TFLOPS | `tflops` |
| Peak VRAM | `peak_vram_gb` |
| Status | `status` |
| Notes | `notes` and `extra` |

## Environment Matrix

| Environment ID | GPU | Architecture | Backend | Driver / Runtime | PyTorch | Compute Capability | Memory | RL-Kernel Commit | Date | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `a100-80gb-baseline` | A100 80GB | Ampere | CUDA | pending | pending | SM80 | 80GB | pending | pending | pending |
| `h100-sxm5-template` | H100 SXM5 | Hopper | CUDA | pending | pending | SM90 | pending | pending | pending | pending |
| `h200-template` | H200 | Hopper | CUDA | pending | pending | SM90 | pending | pending | pending | pending |
| `mi300x-template` | MI300X | CDNA 3 | ROCm | pending | pending | N/A | pending | pending | pending | pending |

## Selected LogP Results

Use `logp-native` for the PyTorch/native baseline and `logp-fused` for the
registered RL-Kernel fused path. If `logp-fused` reports `blocked`, keep the row
and copy the reason into `Notes`.

| Environment | Workload | Backend | Batch | Sequence Length | Vocabulary | Dtype | Latency (ms) | Tokens/s | TFLOPS | Peak VRAM (GB) | Status | Command |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| `a100-80gb-baseline` | `logp-native` | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| `a100-80gb-baseline` | `logp-fused` | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| `h100-sxm5-template` | `logp-native` | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| `h100-sxm5-template` | `logp-fused` | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| `h200-template` | `logp-native` | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| `h200-template` | `logp-fused` | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| `mi300x-template` | `logp-native` | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| `mi300x-template` | `logp-fused` | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

## Sampling Results

The current profiler workload is `sampling-native`. Add fused sampling rows only
when a benchmark workload records the selected fused backend explicitly.

| Environment | Workload | Backend | Batch | Vocabulary | Top-k | Top-p | Temperature | Dtype | Latency (ms) | Tokens/s | Peak VRAM (GB) | Status | Command |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| `a100-80gb-baseline` | `sampling-native` | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| `h100-sxm5-template` | `sampling-native` | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| `h200-template` | `sampling-native` | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| `mi300x-template` | `sampling-native` | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

## Reproduction

Run a small smoke report first:

```bash
python scripts/run_profile_suite.py \
  --smoke \
  --workloads logp-native,sampling-native \
  --no-summary \
  --output-dir reports/hardware-dashboard-smoke
```

Run CUDA logprob profiling:

```bash
python scripts/run_profile_suite.py \
  --device cuda \
  --dtype float16 \
  --batch-sizes 8,16,32 \
  --seq-lens 128,512 \
  --vocab-sizes 4096,128256 \
  --workloads logp-native,logp-fused \
  --output-dir reports/hardware-dashboard-logp
```

Run CUDA sampling profiling:

```bash
python scripts/run_profile_suite.py \
  --device cuda \
  --dtype float16 \
  --batch-sizes 64,128,256 \
  --seq-lens 1 \
  --vocab-sizes 128256 \
  --workloads sampling-native \
  --top-k 50 \
  --top-p 0.9 \
  --output-dir reports/hardware-dashboard-sampling
```

Copy results from the emitted CSV or JSON report into the tables above. Keep the
raw report in the PR notes or an attached artifact when the benchmark output is
large.

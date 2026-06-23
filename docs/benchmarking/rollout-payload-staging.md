# Rollout Payload Staging

Date: 2026-06-23 UTC
Branch: `cse/rollout-payload-staging`

## Background

`RolloutBatchMixin._batch_from_token_groups()` converts generated token ids from
rollout payloads into dense training tensors. The payload is ragged Python data,
while the training step expects dense `token_ids` and `completion_mask` tensors.

The old CUDA path allocated the dense tensors on the training device and then
created one small CUDA tensor per sequence:

```python
values = torch.tensor(clipped, device=self.device, dtype=torch.long)
token_ids[row, : values.numel()] = values
```

For large rollout batches, that turns batch construction into thousands of small
host-to-device copies and row assignments.

## Change

For CUDA training devices, the payload is packed on CPU first:

1. Allocate dense `token_ids` and `completion_mask` on CPU.
2. Fill the ragged rows on CPU.
3. Move each dense tensor to the training device once.

CPU-only behavior still builds the tensors directly on CPU.

## Benchmark

```bash
.codex-nightly/envs/issue112-py312/bin/python \
  benchmarks/benchmark_rollout_payload_staging.py \
  --device cuda \
  --batch-size 1024,4096,8192 \
  --completion-len 128 \
  --warmup 3 \
  --repeat 10 \
  --output .codex-nightly/artifacts/rollout_payload_staging.jsonl
```

Local CUDA 12.8 result:

| Batch | Completion | Row-by-row CUDA staging | Packed CPU staging | Packed pinned staging | Packed CPU speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 128 | 36.957 ms | 20.308 ms | 19.023 ms | 1.82x |
| 4096 | 128 | 154.828 ms | 77.892 ms | 76.337 ms | 1.99x |
| 8192 | 128 | 325.124 ms | 150.326 ms | 148.927 ms | 2.16x |

The production path uses regular CPU tensors. The pinned column is included only
to show the benchmark comparison; it is not used by this PR.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .codex-nightly/envs/issue112-py312/bin/python -m pytest \
  tests/test_training_contract.py -q
```

The focused tests cover CPU staging and CUDA staging behavior.

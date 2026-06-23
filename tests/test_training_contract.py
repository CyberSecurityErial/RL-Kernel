# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.executors.training_contract import (
    RolloutBatchMixin,
    TorchRLTrainingConfig,
    make_rollout_result,
)


class _BatchBuilder(RolloutBatchMixin):
    def __init__(self, config: TorchRLTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)


def _payload():
    return {
        "normalized_outputs": [
            [{"token_ids": [3, 9, 5]}],
            [{"token_ids": [6]}],
        ]
    }


def test_rollout_payload_batch_packs_token_groups_on_cpu():
    worker = _BatchBuilder(
        TorchRLTrainingConfig(
            prompt_len=2,
            completion_len=4,
            vocab_size=8,
            device="cpu",
            seed=7,
        )
    )
    rollout = make_rollout_result(iteration=0, weight_version=1, payload=_payload())

    batch, metrics = worker._batch_from_rollout_or_synthetic(rollout)

    assert metrics["training_data_source"] == "rollout_payload"
    assert batch.token_ids.device.type == "cpu"
    assert batch.token_ids.tolist() == [[3, 1, 5], [6, 0, 0]]
    assert batch.completion_mask.tolist() == [[True, True, True], [True, False, False]]
    assert batch.valid_indices.tolist() == [0, 1, 2, 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_rollout_payload_batch_stages_token_groups_to_cuda():
    worker = _BatchBuilder(
        TorchRLTrainingConfig(
            prompt_len=2,
            completion_len=4,
            vocab_size=8,
            device="cuda",
            seed=7,
        )
    )
    rollout = make_rollout_result(iteration=0, weight_version=1, payload=_payload())

    batch, _ = worker._batch_from_rollout_or_synthetic(rollout)

    assert batch.token_ids.device.type == "cuda"
    assert batch.completion_mask.device.type == "cuda"
    assert batch.token_ids.cpu().tolist() == [[3, 1, 5], [6, 0, 0]]
    assert batch.completion_mask.cpu().tolist() == [
        [True, True, True],
        [True, False, False],
    ]

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""heads.py fit trains each head on the objective the joint trainer gives it."""

from functools import partial

import pytest
import torch
from forespark.models.heads import fore_d_loss
from forespark.train import frozen_heads as fh
from torch.nn.functional import binary_cross_entropy_with_logits

K = 4


def width_of() -> torch.Tensor:
    return torch.cat([torch.full((w,), w - 1) for w in range(1, K + 1)])


def test_confidence_loss_is_the_trainers():
    from speculators.losses import dflash_loss_decay, loss_function

    torch.manual_seed(0)
    logits, labels = torch.randn(6, K), torch.rand(6, K)
    valid = torch.rand(6, K) > 0.2
    gamma = 7.0
    ours = fh.confidence_loss(
        logits, labels, valid.float(), torch.exp(-torch.arange(K) / gamma)
    )

    # The trainer sees blocks of K + 1 slots, the anchor first (masked and weighted 0).
    def blocks(t, first):
        return torch.cat([torch.full((6, 1), first, dtype=t.dtype), t], 1).reshape(
            1, -1
        )

    position = torch.arange(K + 1).repeat(6).unsqueeze(0)
    theirs = loss_function(
        blocks(logits, 0.0), blocks(labels, 0.0), blocks(valid, False), position,
        loss_fn=partial(binary_cross_entropy_with_logits, reduction="none"),
        decay_fn=partial(dflash_loss_decay, gamma=gamma, sample_from_anchor=False),
    )  # fmt: skip
    torch.testing.assert_close(ours, theirs, rtol=1e-4, atol=1e-6)


def test_fore_d_loss_averages_the_trainers_per_width_steps():
    torch.manual_seed(0)
    columns = K * (K + 1) // 2
    logits, labels = torch.randn(5, columns), torch.rand(5, columns)
    valid = torch.rand(5, columns) > 0.3
    valid[:, 0] = True  # every width labeled somewhere
    ours = fh.fore_d_loss_per_width(logits, labels, valid.float(), width_of())
    # Training fits width w's row alone in a step (the other columns masked), every width in turn.
    steps = []
    for row in fh._width_rows(K):
        if valid[:, row].any():
            steps.append(fore_d_loss(logits[:, row], labels[:, row], valid[:, row])[0])
    torch.testing.assert_close(ours, torch.stack(steps).mean())


def test_perfect_predictions_score_perfectly():
    torch.manual_seed(0)
    columns = K * (K + 1) // 2
    labels = torch.rand(8, columns).clamp(0.05, 0.95)
    valid = torch.ones(8, columns)
    exact = fh.fore_d_metrics(torch.logit(labels), labels, valid, width_of())
    assert exact["accepted_mae_mean"] < 1e-5
    assert all(exact[f"regret@{c:g}"] < 1e-6 for c in fh.REGRET_COSTS)
    slots = torch.rand(8, K).clamp(0.05, 0.95)
    decay = torch.ones(K)
    conf = fh.confidence_metrics(torch.logit(slots), slots, torch.ones(8, K), decay)
    assert conf["accept_len_mae"] < 1e-5
    assert all(conf[f"ranking@{b:g}"] > 0.9999 for b in fh.RANKING_BUDGETS)
    # A ranking reversed to the worst order verifies less than the best share.
    flipped = fh.confidence_metrics(-torch.logit(slots), slots, torch.ones(8, K), decay)
    assert flipped["ranking@0.25"] < conf["ranking@0.25"]


def test_spans_and_shards_parse():
    assert fh._span("100:300") == (100, 300)
    assert fh._span(":50") == (0, 50)
    assert fh._shard("1/3") == (1, 3)
    assert fh._shard("0") == (0, 1)


def test_merge_swaps_in_the_fitted_heads(tmp_path):
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    drafter, out = tmp_path / "drafter", tmp_path / "out"
    drafter.mkdir()
    state = {
        "layers.0.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        "confidence_head.proj.weight": torch.zeros(1, 4, dtype=torch.bfloat16),
    }
    save_file(state, str(drafter / "model.safetensors"), metadata={"format": "pt"})
    for name in ("config.json", "draft_model.py", "training_state.json"):
        (drafter / name).write_text("{}")
    heads = tmp_path / "heads.safetensors"
    save_file({"confidence_head.proj.weight": torch.ones(1, 4)}, str(heads))

    assert fh.main(["merge", str(drafter), str(heads), "--out", str(out)]) == 0
    merged = load_file(str(out / "model.safetensors"))
    torch.testing.assert_close(merged["layers.0.weight"], state["layers.0.weight"])
    assert merged["confidence_head.proj.weight"].dtype == torch.bfloat16
    assert merged["confidence_head.proj.weight"].eq(1).all()
    with safe_open(str(out / "model.safetensors"), "pt") as f:
        assert f.metadata() == {"format": "pt"}
    assert sorted(p.name for p in out.iterdir()) == [
        "config.json",
        "draft_model.py",
        "model.safetensors",
    ]
    # A tensor the drafter lacks, or of another shape, is not one of its heads.
    for bad in (
        {"fore_d_predictor.fusion_bias": torch.ones(2)},
        {"confidence_head.proj.weight": torch.ones(1, 5)},
    ):
        save_file(bad, str(heads))
        with pytest.raises(SystemExit):
            fh.main(["merge", str(drafter), str(heads), "--out", str(out)])

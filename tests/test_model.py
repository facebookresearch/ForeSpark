# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""ForeSpark's dflash2_conf training model: the non-causal contract, the heads, and serving translation."""

import pytest
import torch

pytest.importorskip("speculators")

from forespark.config import ForeDConfig  # noqa: E402
from forespark.models.draft_model import (  # noqa: E402
    DFlash2ConfDraftModel,
    DFlash2ConfSpeculatorConfig,
)
from forespark.models.registry import _translate  # noqa: E402
from speculators import (  # noqa: E402
    SpeculatorModelConfig,
    SpeculatorsConfig,
    VerifierConfig,
)
from speculators.losses import resolve_loss_config  # noqa: E402
from speculators.proposals import GreedyTokenProposalConfig  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config  # noqa: E402

BLOCK = 4


def tiny_config(**overrides) -> DFlash2ConfSpeculatorConfig:
    layers = Qwen3Config(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, vocab_size=64, layer_types=["sliding_attention"], use_sliding_window=True, sliding_window=16,
        _attn_implementation="eager",
    )  # fmt: skip
    values = {
        "transformer_layer_config": layers, "draft_vocab_size": 64, "block_size": BLOCK,
        "aux_hidden_state_layer_ids": [0, 1], "mask_token_id": 0, "conv_kernel_size": 2, "conv_group_size": 4,
        "selector_rank": 8, "selector_top_k": 4, "sliding_window_non_causal": True,
        "enable_fore_d_predictor": True, "fore_d_rank": 8,
    }  # fmt: skip
    return DFlash2ConfSpeculatorConfig(**(values | overrides))


def test_heads_and_clip_groups():
    model = DFlash2ConfDraftModel(tiny_config())
    k = BLOCK - 1
    assert (
        model.fore_d_predictor.output_projection.out_features == k * (k + 1) // 2
    )  # one row per width
    groups = model.gradient_clip_groups()
    ids = [id(p) for group in groups for p in group]
    assert len(groups) == 2 and sorted(ids) == sorted(id(p) for p in model.parameters())
    assert {id(p) for p in groups[1]} == {
        id(p) for p in model.fore_d_predictor.parameters()
    }


def test_config_round_trip_and_serving_translation(tmp_path, monkeypatch):
    config = tiny_config(
        speculators_config=SpeculatorsConfig(
            algorithm="dflash2_conf",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=BLOCK - 1)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path="Qwen/Qwen3-4B", architectures=["Qwen3ForCausalLM"]
            ),
        )
    )
    config.save_pretrained(tmp_path)
    loaded = SpeculatorModelConfig.from_pretrained(tmp_path)
    assert isinstance(loaded, DFlash2ConfSpeculatorConfig)
    assert loaded.enable_fore_d_predictor and loaded.sliding_window_non_causal
    assert loaded.fore_d_output_semantics == "width_conditioned_survival"
    config_dict = loaded.to_dict()
    config_dict.setdefault("aux_hidden_state_layer_ids", [1, 2])
    translated = {}
    # not Fore-D: the head stays out (any serving width fits)
    _translate(config_dict, translated)
    assert translated["architectures"] == ["DFlash2ConfDraftModel"]
    assert not translated["is_causal"] and not translated["dflash_config"]["causal"]
    assert not translated["dflash_config"]["enable_fore_d_predictor"]
    monkeypatch.setenv(ForeDConfig.ENV["enabled"], "1")
    translated = {}
    _translate(config_dict, translated)
    assert translated["dflash_config"]["enable_fore_d_predictor"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_training_step_reaches_both_heads():
    torch.manual_seed(0)
    model = DFlash2ConfDraftModel(tiny_config()).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.isnan().any():
                torch.nn.init.normal_(parameter, std=0.02)
    seq, hidden = 32, 16
    inputs = {
        "hidden_states": torch.randn(
            1, seq, 2 * hidden, device="cuda", dtype=torch.bfloat16
        ),
        "input_ids": torch.randint(
            0, model.verifier_vocab_size, (1, seq), device="cuda"
        ),
        "loss_mask": torch.ones(1, seq, device="cuda"),
        "verifier_last_hidden_states": torch.randn(
            1, seq, hidden, device="cuda", dtype=torch.bfloat16
        ),
        "document_ids": torch.zeros(1, seq, device="cuda", dtype=torch.long),
    }
    _, loss, metrics = model(
        **inputs, max_anchors=4, loss_config=resolve_loss_config("kl_div", "eager"),
        tv_loss_fn=resolve_loss_config("tv", "eager")["tv"][0],
    )  # fmt: skip
    assert torch.isfinite(loss)
    assert {
        "confidence_loss_sum",
        "fore_d_loss_sum",
        "fore_d_pred_accept_len_sum",
    } <= metrics.keys()
    loss.backward()
    for head in (model.confidence_head, model.fore_d_predictor):
        grads = [p.grad for p in head.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g) for g in grads)


def test_width_conditioned_head_is_the_non_causal_contract():
    with pytest.raises(ValueError, match="non-causal"):
        DFlash2ConfDraftModel(tiny_config(sliding_window_non_causal=False))
    DFlash2ConfDraftModel(
        tiny_config(sliding_window_non_causal=False, enable_fore_d_predictor=False)
    )  # a causal drafter without the head
    model = DFlash2ConfDraftModel(
        tiny_config(
            sliding_window_non_causal=True,
            fore_d_output_semantics="width_conditioned_survival",
        )
    )
    k = BLOCK - 1
    assert (
        model.fore_d_predictor.output_projection.out_features == k * (k + 1) // 2
    )  # one row per width


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_width_conditioned_training_cycles_widths_with_fresh_narrow_drafts():
    torch.manual_seed(0)
    config = tiny_config(
        sliding_window_non_causal=True,
        fore_d_output_semantics="width_conditioned_survival",
    )
    model = DFlash2ConfDraftModel(config).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.isnan().any():
                torch.nn.init.normal_(parameter, std=0.02)
    seq, hidden = 32, 16
    inputs = {
        "hidden_states": torch.randn(
            1, seq, 2 * hidden, device="cuda", dtype=torch.bfloat16
        ),
        "input_ids": torch.randint(
            0, model.verifier_vocab_size, (1, seq), device="cuda"
        ),
        "loss_mask": torch.ones(1, seq, device="cuda"),
        "verifier_last_hidden_states": torch.randn(
            1, seq, hidden, device="cuda", dtype=torch.bfloat16
        ),
        "document_ids": torch.zeros(1, seq, device="cuda", dtype=torch.long),
    }
    losses = {
        "loss_config": resolve_loss_config("kl_div", "eager"),
        "tv_loss_fn": resolve_loss_config("tv", "eager")["tv"][0],
    }
    model.train()
    k = BLOCK - 1
    for step in range(k):  # widths K, 1, ..., K - 1 in turn
        model.zero_grad()
        _, loss, metrics = model(**inputs, max_anchors=4, **losses)
        assert torch.isfinite(loss) and torch.isfinite(metrics["fore_d_loss_sum"])
        loss.backward()
        grads = [p.grad for p in model.fore_d_predictor.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        # Only this step's width's row of the triangular output learned.
        width = (step - 1) % k + 1
        rows = model.fore_d_predictor.output_projection.weight.grad
        start = width * (width - 1) // 2
        learned = torch.count_nonzero(rows, dim=1).nonzero().flatten().tolist()
        assert learned and set(learned) <= set(range(start, start + width))
    assert model._train_steps == BLOCK - 1
    model.eval()
    with torch.no_grad():
        _, loss, _ = model(
            **inputs, max_anchors=4, **losses
        )  # evaluation: the full-width row
    assert torch.isfinite(loss) and model._train_steps == BLOCK - 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_narrow_labels_match_scoring_every_row_of_a_fresh_narrow_draft():
    """The narrow re-draft scores only rows 1..w against the full blocks' targets and
    normalizer; the reference drafts the (w + 1)-row blocks and scores every row.
    Re-drafting every other anchor gives those anchors the same labels."""
    from forespark.models.heads import log_normalizer, overlap_label

    torch.manual_seed(0)
    config = tiny_config(
        sliding_window_non_causal=True,
        fore_d_output_semantics="width_conditioned_survival",
    )
    model = DFlash2ConfDraftModel(config).to(device="cuda", dtype=torch.float32)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.isnan().any():
                torch.nn.init.normal_(parameter, std=0.02)
        seq, hidden, anchors_max = 32, 16, 6
        batch = (
            torch.randn(1, seq, 2 * hidden, device="cuda"),
            torch.randint(0, model.verifier_vocab_size, (1, seq), device="cuda"),
            torch.ones(1, seq, device="cuda"),
            torch.randn(1, seq, hidden, device="cuda"),
            torch.zeros(1, seq, device="cuda", dtype=torch.long),
            None,
        )
        anchors = model._select_anchors(batch[2], anchors_max)
        targets, normalizer = model._block_targets_and_normalizer(batch[3], anchors[0])
        for width in range(1, BLOCK - 1):
            block = width + 1
            masks = model._build_attention_mask(
                batch[2], anchors_max, batch[4], "cuda", block, anchors
            )[:2]
            label, mask = model._narrow_labels(
                width, 1, targets, normalizer, anchors, masks, *batch
            )
            halves = tuple(a[::2] for a in anchors)
            masks = model._build_attention_mask(
                batch[2], anchors_max // 2, batch[4], "cuda", block, halves
            )[:2]
            half_label, half_mask = model._narrow_labels(
                width, 2, targets, normalizer, halves, masks, *batch
            )
            assert torch.allclose(half_label[::2], label[::2], atol=1e-6)
            assert torch.equal(half_mask[::2], mask[::2])
            assert not half_mask[1::2].any() and not half_label[1::2].any()
            narrow = targets.view(-1, BLOCK, targets.shape[-1])[:, :block].reshape(
                1, -1, targets.shape[-1]
            )
            out = model._backbone_forward(
                *batch,
                max_anchors=anchors_max,
                block_size=block,
                anchors=anchors,
                target_logits=narrow,
            )
            hidden_rows, logits, _, ref_mask, indices = out[:5]
            candidates = logits.topk(model.candidate_selector.top_k, dim=-1).indices
            scores = model.candidate_selector.score_candidates(
                logits,
                hidden_rows,
                model._predecessor_ids(batch[1], indices, block),
                candidates,
            )
            ref = overlap_label(
                narrow, candidates, scores, log_normalizer(narrow)
            ).view(-1, block)[:, 1:]
            assert torch.allclose(label[:, :width], ref, atol=1e-6)
            assert torch.equal(mask[:, :width], ref_mask.bool().view(-1, block)[:, 1:])
            assert not mask[:, width:].any() and not label[:, width:].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_topk_indices_is_exact():
    from forespark.models.topk import topk_indices

    torch.manual_seed(0)
    x = torch.randn(1024, 154880, device="cuda")
    x[::5] = 0.0  # every entry tied: the in-program fallback
    x[1::5] = torch.round(x[1::5] * 4) / 4  # ties at the threshold
    x[2::5, 1000:1064] += 6.0  # every winner inside one tile
    x[3::5, -40:] += 6.0  # winners in the last, partial tile
    for rows in (x, x.to(torch.bfloat16), x[:, :151936].contiguous()):
        got = topk_indices(rows, 16)
        assert got.shape == (*rows.shape[:-1], 16)
        assert (torch.sort(got, -1).values.diff(dim=-1) > 0).all()
        values = torch.sort(rows.gather(-1, got), -1, descending=True).values
        assert torch.equal(values, rows.topk(16, dim=-1).values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("non_causal", [False, True])
def test_causal_drafts_are_prefixes_of_wider_drafts(non_causal):
    """Drafted from the same anchors, a causal drafter's narrower drafts are the
    prefixes of its wider ones -- draft hidden states and logits, every width
    pair -- and a non-causal drafter's are not: why ForeSpark's Fore-D head has
    a row per width, labelled by a narrower re-draft."""
    torch.manual_seed(0)
    model = DFlash2ConfDraftModel(
        tiny_config(sliding_window_non_causal=non_causal, enable_fore_d_predictor=False)
    )
    model = model.to(device="cuda", dtype=torch.float32)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.isnan().any():
                torch.nn.init.normal_(parameter, std=0.02)
        seq, hidden = 32, 16
        inputs = {
            "hidden_states": torch.randn(1, seq, 2 * hidden, device="cuda"),
            "input_ids": torch.randint(
                0, model.verifier_vocab_size, (1, seq), device="cuda"
            ),
            "loss_mask": torch.ones(1, seq, device="cuda"),
            "verifier_last_hidden_states": torch.randn(1, seq, hidden, device="cuda"),
            "document_ids": torch.zeros(1, seq, device="cuda", dtype=torch.long),
        }
        full = model._backbone_forward(
            **inputs, max_anchors=4, return_pre_draft_features=True
        )
        _, _, valid, positions = full[5]
        k, targets = BLOCK - 1, full[2]
        drafts = {k: [t.view(-1, BLOCK, t.shape[-1]) for t in full[:2]]}
        for d in range(1, k):
            narrow = targets.view(-1, BLOCK, targets.shape[-1])[:, : d + 1].reshape(
                1, -1, targets.shape[-1]
            )
            out = model._backbone_forward(
                **inputs,
                max_anchors=4,
                block_size=d + 1,
                anchors=(positions, valid),
                target_logits=narrow,
            )
            drafts[d] = [t.view(-1, d + 1, t.shape[-1]) for t in out[:2]]
    for d1 in range(1, k + 1):
        for d2 in range(d1 + 1, k + 1):
            prefix = all(
                torch.allclose(a[:, : d1 + 1], b[:, : d1 + 1], atol=1e-5)
                for a, b in zip(drafts[d1], drafts[d2], strict=True)
            )
            assert prefix != non_causal, (d1, d2)

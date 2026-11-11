# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Fore-D's packed kernels against upstream's dense ones (GPU).

At uniform width K the packed layout *is* upstream's layout, so every output
must match bit for bit; at mixed widths each request must match its own
dense computation.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("vllm")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from forespark.fore_d.kernels.packed_prep import (  # noqa: E402
    prepare_context,
    prepare_queries,
)
from forespark.fore_d.kernels.selector import (  # noqa: E402
    packed_edge_scores,
    packed_walk,
)
from vllm.model_executor.models.qwen3_dflash2 import (  # noqa: E402
    _grouped_conv,
    _score_edges,
)
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (  # noqa: E402
    prepare_dflash_inputs,
)

DEV = torch.device("cuda")
K, R, BLOCK, MASK_ID, MAX_LEN = 4, 6, 16, 999, 4096
CONTEXT_ROWS = 64  # a context graph's size: >= the batch's target tokens


def batch(num_reqs: int, rejected: list[int]):
    """A decode batch: request i verifies 1+K tokens at positions 100*i + ..."""
    lens = np.full(num_reqs, 1 + K)
    qsl = np.concatenate(([0], np.cumsum(lens)))
    positions = np.concatenate(
        [100 * (i + 1) + np.arange(n) for i, n in enumerate(lens)]
    )
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_tokens=int(qsl[-1]),
        num_scheduled_tokens=lens,
        positions=torch.tensor(positions, device=DEV),
        query_start_loc=torch.tensor(qsl, dtype=torch.int32, device=DEV),
        idx_mapping=torch.arange(num_reqs, dtype=torch.int32, device=DEV)
        .flip(0)
        .contiguous(),
    ), torch.tensor(rejected, dtype=torch.int32, device=DEV)


def buffers(widths: list[int]):
    rows, samples = R * (K + 1), R * K
    b = SimpleNamespace(max_width=K, max_rows=rows, max_num_reqs=R, max_samples=samples)
    for name, size, dtype in [
        ("input_ids", 256, torch.int32), ("positions", 256, torch.int64), ("query_start_loc", R + 1, torch.int32),
        ("seq_lens", R, torch.int32), ("context_positions", 256, torch.int64), ("row_position", 256, torch.int32),
        ("sample_rows", samples, torch.int64), ("sample_pos", samples, torch.int64), ("sample_slot", samples, torch.int32),
        ("sample_step", samples, torch.int32), ("sample_flat", samples, torch.int64),
        ("sample_pred", samples, torch.int64), ("sample_anchor", samples, torch.int64), ("temperature", R, torch.float32),
        ("seeds", R, torch.int64), ("row_start", R + 1, torch.int32), ("sample_start", R + 1, torch.int32),
    ]:  # fmt: skip
        setattr(b, name, torch.zeros(size, dtype=dtype, device=DEV))
    b.tokens = torch.zeros(R + 1, K + 1, dtype=torch.int64, device=DEV)
    b.widths = torch.zeros(R, dtype=torch.int32, device=DEV)
    b.widths[: len(widths)] = torch.tensor(widths, dtype=torch.int32)
    w = b.widths[: len(widths)]
    b.row_start[1 : len(widths) + 1] = torch.cumsum(w + 1, 0)
    b.sample_start[1 : len(widths) + 1] = torch.cumsum(w, 0)
    return b


def run_packed(widths: list[int], rejected: list[int]):
    ib, num_rejected = batch(len(widths), rejected)
    b = buffers(widths)
    q_slots, c_slots = (
        torch.zeros(256, dtype=torch.int64, device=DEV),
        torch.full((256,), 7777, dtype=torch.int64, device=DEV),
    )
    block_table = torch.arange(1, 1 + R * 32, dtype=torch.int32, device=DEV).view(R, 32)
    tables = SimpleNamespace(
        kernel_block_sizes=[BLOCK], cp_rank=0, cp_size=1, cp_interleave=1
    )
    last = torch.arange(10, 10 + R, device=DEV)
    ones = torch.ones(R, dtype=torch.int32, device=DEV)
    n = torch.tensor(ib.num_reqs, dtype=torch.int32, device=DEV)
    total = torch.tensor(sum(widths), dtype=torch.int32, device=DEV)
    prepare_queries(
        b, n, total, R, ib.positions, ib.query_start_loc, ib.idx_mapping, last, last, ones[: ib.num_reqs],
        num_rejected, torch.zeros(R, device=DEV), torch.zeros(R, dtype=torch.int64, device=DEV), q_slots,
        block_table, tables, 0, MASK_ID, MAX_LEN,
    )  # fmt: skip
    prepare_context(n, CONTEXT_ROWS, ib.positions, ib.query_start_loc, num_rejected, b.context_positions, c_slots,
                    block_table, tables, 0, R)  # fmt: skip
    return b, q_slots, c_slots, ib, num_rejected, block_table, last


def test_uniform_width_matches_upstream_layout():
    widths, rejected = [K] * 3, [0, 2, 1]
    b, q_slots, c_slots, ib, num_rejected, block_table, last = run_packed(
        widths, rejected
    )
    up = SimpleNamespace(
        input_ids=torch.zeros(256, dtype=torch.int32, device=DEV),
        positions=torch.zeros(256, dtype=torch.int64, device=DEV),
        query_start_loc=torch.zeros(R + 1, dtype=torch.int32, device=DEV),
        seq_lens=torch.zeros(R, dtype=torch.int32, device=DEV),
    )
    uq, uc = (
        torch.zeros(256, dtype=torch.int64, device=DEV),
        torch.zeros(256, dtype=torch.int64, device=DEV),
    )
    ctx_pos = torch.zeros(256, dtype=torch.int64, device=DEV)
    s_idx, s_pos = (
        torch.zeros(R * K, dtype=torch.int64, device=DEV),
        torch.zeros(R * K, dtype=torch.int64, device=DEV),
    )
    s_map = torch.zeros(R * K, dtype=torch.int32, device=DEV)
    ib.num_scheduled_tokens = np.asarray(ib.num_scheduled_tokens)
    prepare_dflash_inputs(
        up, uq, ctx_pos, uc, s_idx, s_pos, s_map, torch.zeros(R, device=DEV), torch.zeros(R, dtype=torch.int64, device=DEV),
        ib, torch.ones(3, dtype=torch.int32, device=DEV), num_rejected, last, last,
        torch.zeros(R, device=DEV), torch.zeros(R, dtype=torch.int64, device=DEV),
        block_table, BLOCK, 0, 1, 1, MASK_ID, 1 + K, K, R, 256, MAX_LEN,
    )  # fmt: skip
    rows = 3 * (1 + K)
    assert torch.equal(b.input_ids[:rows], up.input_ids[:rows])
    assert torch.equal(b.positions[:rows], up.positions[:rows])
    assert torch.equal(q_slots[:rows], uq[:rows])
    assert torch.equal(c_slots[: ib.num_tokens], uc[: ib.num_tokens])
    assert torch.equal(b.context_positions[: ib.num_tokens], ctx_pos[: ib.num_tokens])
    assert torch.equal(b.query_start_loc[:4], up.query_start_loc[:4])
    assert torch.equal(b.seq_lens[:3], up.seq_lens[:3])
    assert torch.equal(b.sample_rows[: 3 * K], s_idx[: 3 * K])
    assert torch.equal(b.sample_pos[: 3 * K], s_pos[: 3 * K])
    assert torch.equal(b.sample_slot[: 3 * K], s_map[: 3 * K])


def test_mixed_widths_are_contiguous_prefixes_and_padding_is_inert():
    widths = [2, 0, 4]
    b, q_slots, c_slots, ib, *_ = run_packed(widths, [0, 0, 0])
    assert b.query_start_loc[:4].tolist() == [0, 3, 4, 9]
    assert b.row_position[:9].tolist() == [0, 1, 2, 0, 0, 1, 2, 3, 4]
    assert b.input_ids[:9].tolist() == [
        10 + 2,
        MASK_ID,
        MASK_ID,
        10 + 1,
        10 + 0,
        MASK_ID,
        MASK_ID,
        MASK_ID,
        MASK_ID,
    ]
    assert b.sample_rows[:6].tolist() == [1, 2, 5, 6, 7, 8]
    assert b.sample_step[:6].tolist() == [0, 1, 0, 1, 2, 3]
    assert b.sample_flat[:6].tolist() == [0, 1, 2 * K, 2 * K + 1, 2 * K + 2, 2 * K + 3]
    assert b.sample_pred[:6].tolist() == [
        0,
        1,
        2 * (K + 1),
        2 * (K + 1) + 1,
        2 * (K + 1) + 2,
        2 * (K + 1) + 3,
    ]
    assert b.sample_anchor[:6].tolist() == [12, 12, 10, 10, 10, 10]
    assert b.tokens[:3, 0].tolist() == [12, 11, 10]
    assert bool((b.sample_flat[6:] == R * K).all()) and bool(
        (b.sample_pred[6:] == R * (K + 1)).all()
    )
    assert bool((q_slots[9 : R * (K + 1)] == -1).all())  # PAD_SLOT_ID
    assert bool((b.sample_slot[6:] == -1).all())
    assert b.query_start_loc[3:].tolist() == [9] * (R + 1 - 3)
    assert bool(
        (c_slots[ib.num_tokens : CONTEXT_ROWS] == -1).all()
    )  # context rows past T: pad
    assert bool((c_slots[CONTEXT_ROWS:] == 7777).all())  # nothing past the graph's size


def test_context_rows_of_an_uneven_batch():
    # A prefill chunk between decodes: each token must find its own request.
    lens, rejected = [3, 40, 1 + K], [0, 0, 2]
    qsl = np.concatenate(([0], np.cumsum(lens)))
    positions = np.concatenate(
        [1000 * (i + 1) + np.arange(n) for i, n in enumerate(lens)]
    )
    block_table = torch.arange(1, 1 + R * 256, dtype=torch.int32, device=DEV).view(
        R, 256
    )
    tables = SimpleNamespace(
        kernel_block_sizes=[BLOCK], cp_rank=0, cp_size=1, cp_interleave=1
    )
    ctx_pos = torch.full((128,), -5, dtype=torch.int64, device=DEV)
    slots = torch.full((128,), 7777, dtype=torch.int64, device=DEV)
    prepare_context(torch.tensor(3, dtype=torch.int32, device=DEV), 96, torch.tensor(positions, device=DEV),
                    torch.tensor(qsl, dtype=torch.int32, device=DEV), torch.tensor(rejected, dtype=torch.int32, device=DEV),
                    ctx_pos, slots, block_table, tables, 0, R)  # fmt: skip
    expect_pos, expect_slot = np.zeros(96, dtype=np.int64), np.full(96, -1)
    for req, (start, end) in enumerate(zip(qsl[:-1], qsl[1:], strict=True)):
        for token in range(start, end - rejected[req]):
            pos = positions[token]
            expect_pos[token] = pos
            expect_slot[token] = (
                int(block_table[req, pos // BLOCK]) * BLOCK + pos % BLOCK
            )
    assert ctx_pos[:96].tolist() == expect_pos.tolist()
    assert slots[:96].tolist() == expect_slot.tolist()
    assert bool((slots[96:] == 7777).all()) and bool((ctx_pos[96:] == -5).all())


def packed_conv(base_kernel: torch.Tensor, groups: int, row_position: torch.Tensor):
    from forespark.models.vllm_model import PackedGroupedConv

    conv = PackedGroupedConv.__new__(PackedGroupedConv)
    torch.nn.Module.__init__(conv)
    taps, hidden = base_kernel.shape[1:]
    conv.num_groups, conv.group_size, conv.taps, conv.block_size = (
        groups,
        hidden // groups,
        taps,
        1 + K,
    )
    conv.base_kernel = torch.nn.Parameter(base_kernel, requires_grad=False)
    conv.row_position = row_position
    return conv


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def test_packed_conv_matches_upstream_on_uniform_rows_and_per_request_dense_on_mixed():
    torch.manual_seed(0)
    hidden, groups, taps = 64, 4, 2
    x = torch.randn(3 * (1 + K), hidden, device=DEV, dtype=torch.bfloat16)
    delta = torch.randn(x.shape[0], taps, groups, device=DEV, dtype=torch.bfloat16)
    base = torch.randn(taps, hidden, device=DEV, dtype=torch.bfloat16)
    conv = packed_conv(
        torch.stack([base, base]),
        groups,
        (torch.arange(x.shape[0], device=DEV) % (1 + K)).to(torch.int32),
    )
    assert torch.equal(
        conv._convolve(x, delta, 0),
        _grouped_conv(x, delta, base, 1 + K, groups, hidden // groups, taps),
    )
    # Mixed widths [2, 0, 4]: each request equals its own dense conv.
    starts, counts = [0, 3, 4], [3, 1, 5]
    conv.row_position = torch.tensor(
        [0, 1, 2, 0, 0, 1, 2, 3, 4], dtype=torch.int32, device=DEV
    )
    packed = conv._convolve(x[:9], delta[:9], 0)
    for s, c in zip(starts, counts, strict=True):
        dense = _grouped_conv(
            x[s : s + c], delta[s : s + c], base, c, groups, hidden // groups, taps
        )
        assert torch.equal(packed[s : s + c], dense)


@pytest.mark.parametrize("taps", [2, 3])
def test_fused_conv_is_bit_identical_to_the_eager_form(monkeypatch, taps):
    torch.manual_seed(taps)
    rows, hidden, group = 1000, 6144, 16  # GLM-5.3's drafter: 384 groups
    x = torch.randn(rows, hidden, device=DEV, dtype=torch.bfloat16)
    # kernel_projection's output viewed per side, as DFlashGroupedConv.prepare passes it.
    coefficients = torch.randn(
        rows, 2, taps, hidden // group, device=DEV, dtype=torch.bfloat16
    )
    base = torch.randn(2, taps, hidden, device=DEV, dtype=torch.bfloat16)
    uniform = (torch.arange(rows, device=DEV) % (1 + K)).to(torch.int32)
    packed = torch.randint(0, 1 + K, (rows + 8,), dtype=torch.int32, device=DEV)
    for side in (0, 1):
        delta = coefficients[:, side]
        # Whole blocks: upstream's own eager conv; packed rows: the eager branch.
        dense = _grouped_conv(x, delta, base[side], 1 + K, hidden // group, group, taps)
        assert same_bits(
            packed_conv(base, hidden // group, uniform)._convolve(x, delta, side), dense
        )
        conv = packed_conv(base, hidden // group, packed)
        fused = conv._convolve(x, delta, side)
        with monkeypatch.context() as m:
            m.setattr(torch.compiler, "is_compiling", lambda: True)
            assert same_bits(fused, conv._convolve(x, delta, side))


def test_packed_selector_and_walk_match_upstream_dense():
    torch.manual_seed(0)
    n, top_k, rank, vocab, hidden = 3, 8, 16, 128, 32
    selector = SimpleNamespace(
        hidden_projection=torch.nn.Linear(hidden, rank, bias=False, device=DEV),
        predecessor_codebook=torch.randn(vocab, rank, device=DEV),
        successor_codebook=torch.randn(vocab, rank, device=DEV),
    )
    candidate = torch.randint(0, vocab, (n, K, top_k), device=DEV)
    unary = torch.randn(n, K, top_k, device=DEV)
    h = torch.randn(n, K, hidden, device=DEV)
    anchors = torch.randint(0, vocab, (n,), device=DEV)
    dense = _score_edges(
        selector.predecessor_codebook, selector.successor_codebook, candidate, unary,
        selector.hidden_projection(h), anchors, top_k,
    )  # fmt: skip
    first = (torch.arange(n * K, device=DEV) % K) == 0
    packed = packed_edge_scores(
        selector, candidate.view(-1, top_k), unary.view(-1, top_k), h.view(-1, hidden), first,
        anchors.repeat_interleave(K),
    )  # fmt: skip
    torch.testing.assert_close(packed.view(n, K, top_k, top_k), dense)

    # Greedy walk: widths [K, 2, 0] take prefixes of the full-width walk.
    tokens = torch.zeros(R + 1, K, dtype=torch.int64, device=DEV)
    realized = torch.zeros(n * K, top_k, device=DEV)
    zeros = torch.zeros(R, device=DEV)

    def walk(widths):
        w = torch.tensor(widths + [0] * (R - n), dtype=torch.int32, device=DEV)
        start = torch.zeros(R + 1, dtype=torch.int32, device=DEV)
        start[1:] = torch.cumsum(torch.tensor([K] * n + [0] * (R - n), device=DEV), 0)
        tokens.zero_()
        packed_walk(
            packed, candidate.view(-1, top_k), torch.zeros(n * K, dtype=torch.int64, device=DEV),
            torch.arange(n, dtype=torch.int32, device=DEV).repeat_interleave(K), w, start,
            torch.tensor(n, dtype=torch.int32, device=DEV), zeros, torch.zeros(R, dtype=torch.int64, device=DEV),
            tokens, realized, R, probabilistic=False, use_fp64=False,
        )  # fmt: skip
        return tokens[:n].clone()

    full, mixed = walk([K] * n), walk([K, 2, 0])
    assert torch.equal(mixed[0], full[0]) and torch.equal(mixed[1, :2], full[1, :2])


@pytest.mark.parametrize("seed", range(8))
def test_prefix_select_matches_stable_sort(seed):
    from forespark.fore_d.kernels.select import PrefixSelect
    from forespark.fore_d.scheduler.allocate import prefix_counts

    g = torch.Generator().manual_seed(seed)
    rows, width, max_rows = [1, 5, 37, 200][seed % 4], 7, 256
    logits = torch.randn(max_rows, width, generator=g)
    logits[: rows // 2, 3:] = 40.0  # saturated sigmoid: exact ties inside rows
    logits, temps = logits.to(DEV), torch.rand(width, generator=g).add(0.5).to(DEV)
    probs = torch.sigmoid(logits / temps)
    legal = torch.randint(0, width + 1, (max_rows,), generator=g, dtype=torch.int32).to(
        DEV
    )
    n = torch.tensor(rows, dtype=torch.int32, device=DEV)
    sel = PrefixSelect(max_rows, width, DEV)
    sel.survival(probs, n, legal)
    scores = sel.scores.view(-1, sel.k_pad)[:rows, :width]
    torch.testing.assert_close(scores, torch.cumprod(probs[:rows], dim=1))
    legal_total = int(legal[:rows].sum())
    for total in sorted({0, 1, 3, legal_total // 2, legal_total, legal_total + 5}):
        counts = torch.full((max_rows,), -7, dtype=torch.int32, device=DEV)
        rows_a = torch.full((max_rows + 1,), -7, dtype=torch.int32, device=DEV)
        rows_b = torch.full((max_rows + 1,), -7, dtype=torch.int32, device=DEV)
        sel.select(
            n,
            torch.tensor(total, dtype=torch.int32, device=DEV),
            counts,
            rows_a,
            1,
            rows_b,
            0,
        )
        ref = prefix_counts(scores, legal[:rows], total)
        assert torch.equal(counts[:rows], ref) and bool((counts[rows:] == 0).all())
        for offsets, extra in ((rows_a, 1), (rows_b, 0)):
            expect = torch.cumsum(ref + extra, 0).to(torch.int32)
            assert offsets[0] == 0 and torch.equal(offsets[1 : rows + 1], expect)
            assert bool((offsets[rows + 1 :] == expect[-1]).all())


def test_prefix_select_verify_path_gathers_slots_and_caps():
    from forespark.fore_d.kernels.select import PrefixSelect
    from forespark.fore_d.scheduler.allocate import allocate_verify

    g = torch.Generator().manual_seed(0)
    slots, width, rows = 16, 7, 6
    confidence = torch.rand(slots, width, generator=g).to(DEV)
    idx = torch.tensor([9, 2, 14, 0, 5, 7], dtype=torch.int32, device=DEV)
    scheduled = torch.tensor([7, 7, 3, 7, 0, 7], dtype=torch.int32, device=DEV)
    widths_by_slot = torch.randint(
        0, width + 1, (slots,), generator=g, dtype=torch.int32
    ).to(DEV)
    non_draft = torch.tensor([1, 1, 1, 30, 1, 1], dtype=torch.int32, device=DEV)
    sel = PrefixSelect(8, width, DEV)
    n = torch.tensor(rows, dtype=torch.int32, device=DEV)
    sel.survival(
        confidence, n, scheduled, gather=idx, cap_b=widths_by_slot, cap_b_index=idx
    )
    capacity = torch.minimum(scheduled, widths_by_slot[idx.long()])
    counts, logits_at = (
        torch.zeros(8, dtype=torch.int32, device=DEV),
        torch.zeros(9, dtype=torch.int32, device=DEV),
    )
    qsl = torch.zeros(9, dtype=torch.int32, device=DEV)
    for quota in (0, 4, int(capacity.sum())):
        sel.select(
            n,
            torch.tensor(quota, dtype=torch.int32, device=DEV),
            counts,
            logits_at,
            1,
            qsl,
            non_draft,
        )
        ref = allocate_verify(confidence[idx.long()], capacity, quota)
        assert torch.equal(counts[:rows], ref)
        assert torch.equal(
            qsl[1 : rows + 1], torch.cumsum(ref + non_draft, 0).to(torch.int32)
        )


@pytest.mark.parametrize(
    "slice_rows", [200, 333]
)  # five even slices; a last slice of one row, under k
def test_sliced_top_k_matches_one_pass(monkeypatch, slice_rows):
    from forespark.fore_d.kernels import selector
    from vllm.model_executor.layers.logits_processor import _topk

    torch.manual_seed(0)
    rows, hidden_size, vocab, pad, k = 37, 64, 1000, 24, 8
    hidden = torch.randn(rows, hidden_size, device=DEV).to(torch.bfloat16)
    weight = torch.randn(vocab + pad, hidden_size, device=DEV).to(torch.bfloat16)
    weight[[10, 420, 990]] = (
        weight[5] * 4
    )  # identical top rows in different slices: ties must resolve as one pass
    weight[vocab:] = weight[5] * 8  # padding rows would win if they were not excluded
    head = SimpleNamespace(weight=weight, tp_size=1,
                           shard_indices=SimpleNamespace(num_org_vocab_padding=pad, org_vocab_start_index=0))  # fmt: skip
    processor = SimpleNamespace(scale=1.0, soft_cap=None)
    logits = torch.nn.functional.linear(hidden, weight)
    logits[:, -pad:] = -float("inf")
    values, ids = _topk(logits, k)
    monkeypatch.setattr(selector, "LOGITS_BLOCK_BYTES", rows * 2 * slice_rows)
    got_ids, got_values = selector.sliced_top_k(processor, head, hidden, k)
    assert torch.equal(got_ids, ids.to(torch.int64))
    assert torch.equal(got_values, values.float())


def test_width_gains_kernel_matches_the_hull_reference():
    from forespark.fore_d.kernels.select import PrefixSelect
    from forespark.fore_d.scheduler.allocate import (
        hull_gains,
        prefix_counts,
        width_expected,
    )

    torch.manual_seed(0)
    k, n, slots = 7, 37, 64
    t = k * (k + 1) // 2
    logits = (torch.randn(n, t, device=DEV) * 2).to(torch.bfloat16)
    scale, bias = torch.rand(t, device=DEV) + 0.5, torch.randn(t, device=DEV) * 0.3
    legal = torch.randint(0, k + 1, (n,), device=DEV, dtype=torch.int32)
    slot_of = torch.randperm(slots, device=DEV)[:n].to(torch.int32)
    rows_out = torch.full((slots + 1, k), -7.0, device=DEV)
    select = PrefixSelect(slots, k, DEV)
    n_dev = torch.tensor(n, dtype=torch.int32, device=DEV)
    select.width_gains(
        logits, n_dev, legal, (scale, bias), rows_out=rows_out, rows_index=slot_of
    )
    expect = hull_gains(
        width_expected(logits.float().cpu(), scale.cpu(), bias.cpu()),
        legal.cpu().long(),
    )
    got = select.scores.view(-1, select.k_pad)[:n, :k].cpu()
    valid = torch.arange(k)[None, :] < legal.cpu()[:, None]
    assert torch.allclose(got[valid], expect[valid], atol=1e-5)
    assert torch.allclose(
        rows_out[slot_of.long()].cpu()[valid], expect[valid], atol=1e-5
    )
    for draft in (0, 5, int(legal.sum()) // 2, int(legal.sum())):
        counts = torch.zeros(slots, dtype=torch.int32, device=DEV)
        offsets = torch.zeros(slots + 1, dtype=torch.int32, device=DEV)
        select.select(
            n_dev,
            torch.tensor(draft, dtype=torch.int32, device=DEV),
            counts,
            offsets,
            1,
        )
        widths = prefix_counts(expect, legal.cpu().long(), torch.tensor(draft))
        assert counts[:n].cpu().tolist() == widths.tolist()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Fore-D, ForeSpark's scheduler, on DFlash2-conf: plan (D, B), draft packed, verify top-B.

One step (all enqueue-only on the host; nothing waits for this step's GPU work):

  host    plan = planner.plan(stale rows, legal widths)  -> (D, B)
  device  pre-draft graph: the Fore-D head on the current target state -> one
          nonincreasing row per request (the increments of its
          expected-acceptance-by-width hull); widths d_i = per-request counts
          of the top D; packed query rows (request i gets 1 + d_i)
          copy the rows to the host for the next plan
          context graph: target-token positions and slots, the draft's K/V
          backbone graph (rows n + D) -> tail graph (D samples):
          LM head, selector, walk, confidence
  next    JointVerificationManager spends B on the current confidences

Every host decision is a function of data identical on all TP ranks (the
previous step's rows, host-side widths), so ranks agree without collectives.
"""

import itertools
import json
import logging
import os

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

from ...config import CPU_RANKING_ENV, ForeDConfig
from ...serving.cpu_ranking import prefix_widths
from ...serving.speculator import DFlash2ConfSpeculator
from ...serving.sts import _is_rank_zero
from ...serving.tp_consistency import rank0_broadcaster
from ..kernels.fore_d import fore_d_inputs
from ..kernels.packed_prep import prepare_context, prepare_queries
from ..kernels.select import PrefixSelect
from ..kernels.selector import (
    logits_fit,
    packed_cache_draft_logits,
    packed_edge_scores,
    packed_walk,
    sliced_top_k,
)
from ..scheduler.cost import (
    bucket_index,
    CostModel,
    extend_sizes,
    isotonic,
    MeasuredCosts,
)
from ..scheduler.planner import JointPlanner, Plan, safe_verify_quota
from .graphs import BackboneGraphs, bound_draft_splits, SizedGraphs
from .stale import PinnedStaging, StaleRows
from .steplog import StepLog
from .verification import HostJointVerificationManager, JointVerificationManager

logger = logging.getLogger(f"vllm.{__name__}")  # under vLLM's handler
PROFILE_REPLAYS = 5


class ForeDSpeculator(DFlash2ConfSpeculator):
    _speculator_name = "Fore-D"
    verification_manager_cls = JointVerificationManager
    profiling_capture = False  # set around capture() by the model runner (vllm.patch)

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        if not self.enable_adaptive_verification:
            raise ValueError(
                "Fore-D verifies through adaptive verification; enable it in the speculative config"
            )
        if (
            vllm_config.parallel_config.data_parallel_size > 1
            or vllm_config.parallel_config.pipeline_parallel_size > 1
        ):
            raise ValueError("Fore-D supports tensor parallelism only (DP = PP = 1)")
        # The packed conv changes the traced draft graph; keep its compile cache apart.
        vllm_config.additional_config = {
            **(vllm_config.additional_config or {}),
            "forespark_packed_rows": 1,
        }
        self.config = ForeDConfig.from_env()
        steps, reqs = self.num_speculative_steps, self.max_num_reqs
        self.max_width, self.max_rows, self.max_samples = (
            steps,
            reqs * (steps + 1),
            reqs * steps,
        )
        if self.max_rows > self.max_num_tokens:
            raise ValueError(
                f"max_num_batched_tokens must be >= max_num_seqs * (k+1) = {self.max_rows}"
            )
        # Host-lagged allocation (FORESPARK_CPU_RANKING): widths and the verify spread
        # come from the previous step's rows on the host, so the CPU partition is
        # exact for backends that plan from it (GLM-5.3-Flash's sparse MLA).
        self.host_allocation = bool(os.environ.get(CPU_RANKING_ENV))
        if self.host_allocation:
            self.verification_manager_cls = HostJointVerificationManager

        def buffer(size, dtype):
            return torch.zeros(size, dtype=dtype, device=device)

        # Per-step host inputs, one H2D copy: [n, D, legal widths, request slots,
        # host widths].
        self.step_inputs = buffer(2 + 3 * reqs, torch.int32)
        self.staging = PinnedStaging(2 + 3 * reqs)
        self.num_reqs_device, self.draft_total_device = (
            self.step_inputs[0],
            self.step_inputs[1],
        )
        self.legal = self.step_inputs[2 : 2 + reqs]
        self.slots = self.step_inputs[2 + reqs : 2 + 2 * reqs]
        self.host_widths = self.step_inputs[2 + 2 * reqs :]
        # Per-step device inputs the pre-draft graph reads from fixed addresses.
        self.num_sampled_buf = buffer(reqs, torch.int32)
        self.num_rejected_buf = buffer(reqs, torch.int32)
        # The sampler's temperatures and seeds live in a UVA buffer pool that hands
        # out another buffer on every staged write, so a graph must not hold its
        # tensors: it would read a stale buffer while the pool refills it, rank by
        # rank. The graphs read these copies, refreshed every step (propose).
        self.step_temperature = buffer(reqs, torch.float32)
        self.step_seeds = buffer(reqs, torch.int64)
        # Packed layout (select.py and packed_prep.py write it; the graphs read it).
        self.select = PrefixSelect(reqs, steps, device)
        self.widths = buffer(reqs, torch.int32)  # batch order
        self.widths_by_slot = buffer(
            reqs, torch.int32
        )  # read by the verification manager
        self.sample_start = buffer(reqs + 1, torch.int32)
        self.row_position = buffer(self.max_num_tokens, torch.int32)
        self.sample_step = buffer(self.max_samples, torch.int32)
        self.sample_flat = buffer(
            self.max_samples, torch.int64
        )  # into [requests + 1, K]
        self.sample_pred = buffer(
            self.max_samples, torch.int64
        )  # predecessor, into tokens
        self.sample_anchor = buffer(self.max_samples, torch.int64)
        self.fore_d_rows = buffer(reqs, torch.int64)
        self.fore_d_anchors = buffer(reqs, torch.int64)
        self.draft_hidden = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )
        self.realized_scores = torch.zeros(
            self.max_samples, self.selector_top_k, dtype=torch.float32, device=device
        )
        # [anchor, draft_1..draft_K] per request, plus a spare row that padding
        # samples read and write.
        self.tokens = buffer((reqs + 1, steps + 1), torch.int64)
        self.draft_tokens = self.tokens[:reqs, 1:]
        self._confidence = buffer((reqs + 1, steps), torch.float32)
        self.draft_token_confidence_probs = self._confidence[:reqs]
        # packed_prep.py's names for upstream buffers. A request's first row is
        # its query start, so the draft query_start_loc doubles as row_start.
        self.input_ids, self.positions = (
            self.input_buffers.input_ids,
            self.input_buffers.positions,
        )
        self.query_start_loc, self.seq_lens = (
            self.input_buffers.query_start_loc,
            self.input_buffers.seq_lens,
        )
        self.row_start = self.query_start_loc
        self.sample_rows, self.sample_slot = (
            self.sample_indices,
            self.sample_idx_mapping,
        )

        self.stale = StaleRows(reqs, steps, device)
        self._arange = np.arange(reqs + 1, dtype=np.int32)
        self.planner: JointPlanner | None = None
        # The last proposal: its plan, legal width per slot, the slots that
        # drafted, and the reward curve it was planned from.
        self.pending: tuple | None = None
        # Slots a new request took since the last proposal (on_new_request):
        # none of the proposal's drafts there are verifiable.
        self.reset_slots: set[int] = set()
        self._planned: tuple[np.ndarray, int, float] | None = None
        self.num_drafted_tokens: int | None = (
            None  # read by the model runner for metrics
        )
        # Per-output affine calibration of the Fore-D logits (calibrate_fore_d.py);
        # sized by the head in load_draft_model.
        self.fore_d_calibration: tuple[torch.Tensor, torch.Tensor] | None = None
        self._tp_broadcast = (
            rank0_broadcaster()
        )  # rank 0's Fore-D logits on every rank (serving/tp_consistency.py)
        self.tail_graphs: SizedGraphs | None = None  # init_cudagraph_manager
        self.pre_graphs: SizedGraphs | None = None
        self.context_graphs: SizedGraphs | None = None
        self.backbone_ms: dict[int, float] = {}
        self.tail_ms: dict[int, float] = {}
        self.steplog = (
            StepLog(self.config.log) if self.config.log and _is_rank_zero() else None
        )
        self._sliceable_head = False  # load_draft_model

    # --- setup ---------------------------------------------------------------

    def load_draft_model(self, target_model, target_attn_layer_names):
        model = super().load_draft_model(target_model, target_attn_layer_names)
        head = model.model.fore_d_predictor
        if head is None:
            raise ValueError(
                "Fore-D needs a dflash2_conf checkpoint with its Fore-D head"
            )
        outputs = head.output_projection.output_size  # K(K+1)/2: widths 1..K
        head.use_bf16_projections()
        model.model.use_packed_rows(self.row_position)
        head, processor = model.lm_head, model.candidate_logits_processor
        self._sliceable_head = type(head.quant_method).__name__ in (
            "UnquantizedEmbeddingMethod",
            "UnquantizedLinearMethod",
        ) and processor.head_dtype in (None, head.weight.dtype)
        scale = torch.ones(outputs, dtype=torch.float32, device=self.device)
        bias = torch.zeros(outputs, dtype=torch.float32, device=self.device)
        if self.config.fore_d_calibration:
            fit = json.loads(self.config.fore_d_calibration.read_text())
            if len(fit["scale"]) != outputs:
                raise ValueError(
                    f"{self.config.fore_d_calibration} fits {len(fit['scale'])} Fore-D outputs, "
                    f"the head has {outputs}"
                )
            scale.copy_(torch.tensor(fit["scale"], dtype=torch.float32))
            bias.copy_(torch.tensor(fit["bias"], dtype=torch.float32))
        self.fore_d_calibration = (scale, bias)
        return model

    def bind_runner(self, runner) -> None:
        """The model runner, after building the verification manager (vllm.patch)."""
        self.verification = runner.adaptive_verification
        self.verification.speculator = self
        # Persistent per-slot state the pre-draft graph reads directly (the
        # sampler's temperatures and seeds through this speculator's copies).
        self.runner_state = (
            runner.req_states.last_sampled_tokens, runner.req_states.next_prefill_tokens,
            self.step_temperature, self.step_seeds,
        )  # fmt: skip

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        if self.attn_cg_support.min_cg_support != AttentionCGSupport.ALWAYS:
            raise ValueError(
                f"Fore-D needs a draft attention backend with varlen CUDA graphs (FLASH_ATTN on FA3); "
                f"{self.attn_cg_support.min_cg_attn_backend} reports {self.attn_cg_support.min_cg_support}"
            )
        full = cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
        mode = CUDAGraphMode.FULL_DECODE_ONLY if full else CUDAGraphMode.NONE
        # A full-width draft step outgrows vLLM's graphs: n + D rows reach
        # max_num_seqs * (K+1). Past the largest graph it would run eagerly
        # and cap D at that size.
        sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes or []
        rows = extend_sizes(sizes, self.max_rows)
        self.backbone_graphs = BackboneGraphs(
            self.vllm_config, self.device, mode, self.max_width + 1, rows
        )
        self.query_cudagraph_manager = (
            self.backbone_graphs
        )  # upstream helpers look for this name
        samples = extend_sizes(sizes, self.max_samples)
        self.tail_graphs = SizedGraphs(samples, self.device) if full else None
        self.pre_graphs = (
            SizedGraphs([s for s in sizes if s <= self.max_num_reqs], self.device)
            if full
            else None
        )
        self.context_graphs = SizedGraphs(rows, self.device) if full else None

    def capture(self) -> None:
        # The drafter has its own builders (init_attn_backend); the target's stay.
        for group in itertools.chain.from_iterable(self.attn_groups):
            for builder in group.metadata_builders:
                if isinstance(builder, FlashAttentionMetadataBuilder):
                    bound_draft_splits(builder, self.max_rows)
        self.row_position.zero_()
        self.sample_rows.zero_()
        self.sample_slot.fill_(-1)
        self.sample_flat.fill_(self.max_num_reqs * self.max_width)
        self.sample_pred.fill_(self.max_num_reqs * (self.max_width + 1))
        self.widths.zero_()
        self.step_inputs.zero_()
        # Backbone first, then the rest into the blocks it leaves free in the
        # shared pool (measured: tail first grows the pool by its logits).
        self.backbone_graphs.capture(
            self._generate_backbone, self.input_buffers, self.block_tables, self.attn_groups,
            self.kv_cache_config, self.max_model_len, self._group_causal,
        )  # fmt: skip
        if self.tail_graphs is not None:
            self.tail_graphs.capture(self._generate_tail)
        if self.pre_graphs is not None:
            target = self.target_input_buffers
            self.pre_graphs.capture(
                lambda reqs: self._pre_draft(
                    reqs, target.query_start_loc, target.positions, *self.runner_state
                )
            )
            self.context_graphs.capture(
                lambda size: self._context_kv(
                    size, target.query_start_loc, target.positions
                )
            )
        # vLLM's memory profile captures once more first, on a minimal KV cache
        # it throws away: replaying graphs recorded there faults on GLM-5.3's
        # layout, and the real capture times them anyway.
        if not self.profiling_capture:
            self._profile_graphs()

    def _profile_graphs(self) -> None:
        """Replay each captured graph a few times: the planner's draft cost tables."""

        def timed(replay) -> float:
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            replay()
            start.record()
            for _ in range(PROFILE_REPLAYS):
                replay()
            end.record()
            end.synchronize()
            return start.elapsed_time(end) / PROFILE_REPLAYS

        for desc, graph in self.backbone_graphs.graphs.items():
            self.backbone_ms[desc.num_tokens] = timed(graph.replay)
        if self.tail_graphs is not None:
            for size, graph in self.tail_graphs.graphs.items():
                self.tail_ms[size] = timed(graph.replay)
        # Ranks must plan from identical tables.
        from vllm.distributed.parallel_state import get_tp_group

        self.backbone_ms, self.tail_ms = get_tp_group().broadcast_object(
            (self.backbone_ms, self.tail_ms), src=0
        )

    def on_step_costs(
        self,
        tokens,
        bare_ms,
        context_ms,
        drafted: dict[int, float],
        profile_context: int,
    ) -> None:
        """The startup profile arrived (verification.py): build the planner.

        `drafted[n]` is a whole full-width draft step for n requests; what the
        backbone and tail graphs do not explain is the step's fixed cost.
        """
        if not self.backbone_ms or not self.tail_ms:
            return  # eager (no graphs): no draft cost tables, run full width
        rows = np.array(sorted(self.backbone_ms))
        samples = np.array([0, *sorted(self.tail_ms)])
        backbone = np.array([self.backbone_ms[r] for r in rows])
        tail = np.array([0.0, *(self.tail_ms[s] for s in samples[1:])])
        fixed_requests = np.array(sorted(drafted), dtype=np.float64)
        graphs = [
            backbone[bucket_index(rows, n * (self.max_width + 1))] + tail[bucket_index(samples, n * self.max_width)]
            for n in sorted(drafted)
        ]  # fmt: skip
        # A noisy residual; the cost of the rest of a step grows with n.
        fixed = isotonic(
            np.maximum(
                np.array([drafted[n] for n in sorted(drafted)]) - np.array(graphs), 0.0
            )
        )
        config = self.vllm_config.compilation_config
        target_graphs = sorted(config.cudagraph_capture_sizes or [])
        cost = CostModel(
            row_buckets=rows, backbone_ms=backbone, sample_buckets=samples, tail_ms=tail,
            verify_tokens=np.array(tokens, dtype=np.float64),
            verify_bare_ms=np.array(bare_ms, dtype=np.float64),
            verify_context_ms=np.array(context_ms, dtype=np.float64),
            fixed_requests=fixed_requests if len(fixed_requests) else np.array([1.0]),
            fixed_ms=fixed if len(fixed) else np.array([0.0]), profile_context=float(profile_context),
            graph_limit=float(target_graphs[-1]) if target_graphs else 0.0,
        )  # fmt: skip
        logger.info(
            "Fore-D step costs: fixed %s ms at n=%s; verify at %s tokens: %s ms bare, %s ms with context",
            np.round(fixed, 2).tolist(),
            fixed_requests.astype(int).tolist(),
            tokens[::8],
            np.round(bare_ms[::8], 2).tolist(),
            np.round(context_ms[::8], 2).tolist(),
        )
        if self.config.costs:
            cost.measured = MeasuredCosts.load(self.config.costs)
            levels = cost.measured.requests.astype(int).tolist()
            logger.info("Fore-D step costs: measured at n=%s", levels)
        self.planner = JointPlanner(cost)
        # The target's graph sizes, then a grid past them: a step whose prefill
        # already exceeds the largest graph still verifies.
        buckets = extend_sizes(target_graphs, self.max_num_tokens)
        self.planner.verify_buckets = np.array(buckets)

    def on_new_request(self, slot: int) -> None:
        self.stale.reset(slot)
        # Nothing of the previous occupant is verifiable. fill_ on a view, not
        # `[slot] = 0`: that copies from pageable memory and blocks the host
        # until the GPU drains.
        self.widths_by_slot[slot].fill_(0)
        self.reset_slots.add(slot)

    # --- planning ------------------------------------------------------------

    def _legal_widths(self, input_batch) -> np.ndarray:
        """Widest legal draft per request, host-side and conservative: 0 for a
        request still mid-prefill after this step, else min(K, room left)."""
        n = input_batch.num_reqs
        room = self.max_model_len - input_batch.seq_lens_cpu_upper_bound[:n].numpy()
        legal = np.clip(room, 0, self.max_width)
        samples = (
            input_batch.num_computed_tokens_np[:n]
            + input_batch.num_scheduled_tokens[:n]
            >= input_batch.prefill_len_np[:n]
        )
        return np.where(samples, legal, 0).astype(np.int32)

    def _stale_rows(self, input_batch) -> np.ndarray:
        slots = input_batch.idx_mapping_np[: input_batch.num_reqs]
        rows = self.stale.read()[slots]  # fancy indexing: already a copy
        missing = self.stale.missing[slots]
        if missing.any():
            known = rows[~missing]
            rows[missing] = known.mean(axis=0) if len(known) else 1.0
        return rows

    def _plan(self, input_batch, legal: np.ndarray, dummy_run: bool) -> Plan:
        total = int(legal.sum())
        full = Plan(total, total, 0.0)
        if (
            dummy_run
            or self.planner is None
            or self.config.log
            or input_batch.has_structured_output_reqs
        ):
            return full  # full width, verify everything
        n = input_batch.num_reqs
        reward = self.planner.reward_curve(self._stale_rows(input_batch), legal)
        context = float(input_batch.seq_lens_cpu_upper_bound[:n].numpy().sum())
        self._planned = (reward, n, context)
        if self.config.fixed_width is not None:  # legal is already clamped to it
            return Plan(
                total, self.planner.replan_quota(total, reward, n, n, context), 0.0
            )
        return self.planner.plan(reward, n, non_draft=n, context_tokens=context)

    def verify_quota(
        self, slots: np.ndarray, scheduled: np.ndarray, non_draft: int
    ) -> int:
        """B for verifying the last proposal. D is spent; B is re-planned with
        the verify step's actual non-draft tokens (new prefill chunks change
        the step's cost), then bounded by what surely survives."""
        if self.pending is None:
            return 0
        plan, legal_by_slot, drafted_slots, planned = self.pending
        if planned is not None:
            reward, n, context = planned
            plan = Plan(
                plan.draft_total,
                self.planner.replan_quota(
                    plan.draft_total, reward, n, non_draft, context
                ),
                plan.rate,
            )
        scheduled_by_slot = np.zeros(self.max_num_reqs, dtype=np.int64)
        scheduled_by_slot[slots] = scheduled
        return safe_verify_quota(
            plan, legal_by_slot, drafted_slots, scheduled_by_slot, self.reset_slots
        )

    # --- the draft step ------------------------------------------------------

    def _pre_draft(
        self,
        reqs,
        query_start_loc,
        positions,
        last_sampled,
        next_prefill,
        temperature,
        seeds,
        prep=True,
    ):
        """Fore-D head -> width gains -> widths -> packed query rows for up to `reqs`
        requests (n and D read from device): the pre-draft graph's body. `prep`
        off only for memory profiling, which has no block tables yet."""
        n, slots = self.num_reqs_device, self.slots
        # The head reads the target state of the last accepted token and the anchor.
        fore_d_inputs(
            query_start_loc, self.num_rejected_buf, self.num_sampled_buf, slots, last_sampled, next_prefill, n,
            self.fore_d_rows, self.fore_d_anchors,
        )  # fmt: skip
        logits = self.model.compute_fore_d_logits(
            self.hidden_states[self.fore_d_rows[:reqs]], self.fore_d_anchors[:reqs]
        )
        if (
            self._tp_broadcast is not None
        ):  # every decision below must be the same on every rank
            self._tp_broadcast(logits)
        # One nonincreasing row per request: the gains of the
        # expected-acceptance-by-width hull.
        self.select.width_gains(logits, n, self.legal, calibration=self.fore_d_calibration,
                                rows_out=self.stale.device_rows, rows_index=slots)  # fmt: skip
        if self.host_allocation:  # the host chose the widths: select exactly those
            self.select.limits[:reqs].copy_(self.host_widths[:reqs])
        self.select.select(
            n, self.draft_total_device, self.widths, self.row_start, 1, self.sample_start, 0,
            slot_counts=self.widths_by_slot, slots=slots,
        )  # fmt: skip
        for gid in self.draft_kv_cache_group_ids if prep else ():
            prepare_queries(
                self, n, self.draft_total_device, reqs, positions, query_start_loc, slots, last_sampled,
                next_prefill, self.num_sampled_buf, self.num_rejected_buf, temperature, seeds,
                self.block_tables.slot_mappings[gid], self.block_tables.input_block_tables[gid], self.block_tables,
                gid, self.parallel_drafting_token_id, self.max_model_len,
            )  # fmt: skip
        return logits

    def _context_slots(self, num_tokens: int):
        """Each draft layer's context slots (its KV group's row), or one shared row."""
        if self._layer_group_idx is not None:
            return [
                self._context_slot_mappings[g][:num_tokens]
                for g in self._layer_group_idx
            ]
        return self._context_slot_mappings[0][:num_tokens]

    def _context_kv(
        self, size: int, query_start_loc, positions, store: bool = True
    ) -> None:
        """The draft's K/V for the step's target tokens: the context graph's
        body. Of `size` rows the first T (read from device) are real; the rest
        get pad slots, which the cache write skips (stale slots from an earlier
        step would overwrite live cache entries). `store` off: compute only."""
        for i, gid in enumerate(self.draft_kv_cache_group_ids):
            prepare_context(
                self.num_reqs_device, size, positions, query_start_loc, self.num_rejected_buf, self.context_positions,
                self._context_slot_mappings[i], self.block_tables.input_block_tables[gid], self.block_tables, gid,
                self.max_num_reqs,
            )  # fmt: skip
        self.model.precompute_and_store_context_kv(
            self.hidden_states[:size],
            self.context_positions[:size],
            self._context_slots(size) if store else None,
        )

    def _packed_attn_metadata(self, n: int, desc, draft_total: int, seq_upper):
        # A conservative CPU partition: the right total (n + D) and a max query
        # length that bounds every device layout (the first requests take K
        # drafts each). FLASH_ATTN takes the real per-request boundaries from
        # the device buffers.
        index = self._arange[: n + 1]
        qsl = index + np.minimum(draft_total, self.max_width * index)
        return DraftModelSpeculator._build_attn_metadata(
            self, n, desc, qsl, seq_upper, step=1 + self.max_width, causal=self._group_causal,
        )  # fmt: skip

    def _generate_backbone(self, num_reqs, num_tokens_padded, attn_metadata, slot_mappings, num_tokens_across_dp,
                           cudagraph_runtime_mode=CUDAGraphMode.NONE) -> None:  # fmt: skip
        hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self.draft_hidden[:num_tokens_padded].copy_(hidden)

    def _generate_tail(self, samples: int) -> None:
        step = self.sample_step[:samples]
        hidden = self.draft_hidden[self.sample_rows[:samples]]
        candidate_ids, unary_logits = self._candidates(hidden)
        scores = packed_edge_scores(
            self.model.model.candidate_selector, candidate_ids, unary_logits, hidden, step == 0,
            self.sample_anchor[:samples],
        )  # fmt: skip
        packed_walk(
            scores, candidate_ids, self.sample_pos, self.sample_slot, self.widths, self.sample_start,
            self.num_reqs_device, self.temperature, self.seeds, self.draft_tokens, self.realized_scores,
            self.max_num_reqs, probabilistic=self.draft_logits is not None, use_fp64=self.use_fp64_gumbel,
        )  # fmt: skip
        if self.draft_logits is not None:
            packed_cache_draft_logits(
                self.draft_logits,
                self._cached_candidate_ids,
                candidate_ids,
                self.realized_scores,
                self.sample_slot,
                step,
            )
        # The confidence head conditions on the token the walk actually chose.
        previous = self.tokens.view(-1)[self.sample_pred[:samples]]
        self._confidence.view(-1)[self.sample_flat[:samples]] = (
            self.model.compute_confidence(hidden, previous, step)
        )

    def _candidates(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k candidates per sample row. A full [samples, vocab] logits block
        would be the draft graph pool's largest allocation (1.1 GiB at 3.5k
        samples, qwen14 TP1), carved out of the KV cache; past 256 MiB it is
        computed in vocab slices instead."""
        model = self.model
        if logits_fit(model.lm_head, hidden) or not self._sliceable_head:
            return model.compute_candidates(hidden)
        return sliced_top_k(
            model.candidate_logits_processor, model.lm_head, hidden, self.selector_top_k
        )

    @torch.inference_mode()
    def propose(
        self, input_batch, attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states,
        num_sampled, num_rejected, last_sampled, next_prefill_tokens, temperature, seeds,
        dp_sync=None, dummy_run=False, skip_attn_for_dummy_run=False, mm_inputs=None, is_profile=False,
    ) -> torch.Tensor:  # fmt: skip
        n, num_target = input_batch.num_reqs, input_batch.num_tokens
        hidden = (
            self.model.combine_hidden_states(torch.cat(aux_hidden_states, dim=-1))
            if aux_hidden_states
            else last_hidden_states
        )
        self.hidden_states[:num_target].copy_(hidden[:num_target])
        seq_upper = input_batch.seq_lens_cpu_upper_bound
        self.draft_max_seq_len = min(
            int(seq_upper[:n].max()) + 1 + self.max_width, self.max_model_len
        )

        legal = (
            self._legal_widths(input_batch)
            if not dummy_run
            else np.full(n, self.max_width, dtype=np.int32)
        )
        if self.config.fixed_width is not None and not dummy_run:
            legal = np.minimum(
                legal, self.config.fixed_width
            )  # D is their sum: all selected
        plan = self._plan(input_batch, legal, dummy_run)
        draft_total, widths = plan.draft_total, legal
        if self.host_allocation:
            # The planner's own stale rows, split on the host.
            if draft_total < int(legal.sum()):
                valid = np.arange(self.max_width)[None, :] < legal[:, None]
                rows = self._stale_rows(input_batch)
                widths = prefix_widths(rows, valid, draft_total)
            draft_total = int(widths.sum())

        # The current Fore-D rows -> widths -> packed layout on the device (the planner
        # only fixed the total). One H2D copy of the host inputs, four small D2D
        # copies of the sampler's, then one graph replay on decode-only steps.
        host = self.staging.next()
        host[:2] = n, draft_total
        host[2 : 2 + n] = legal
        host[2 + self.max_num_reqs : 2 + self.max_num_reqs + n] = (
            input_batch.idx_mapping_np[:n]
        )
        staged = 2 + self.max_num_reqs + n
        if self.host_allocation:
            host[2 + 2 * self.max_num_reqs : 2 + 2 * self.max_num_reqs + n] = widths
            staged = 2 + 2 * self.max_num_reqs + n
        self.staging.copy_to(self.step_inputs, staged)
        self.num_sampled_buf[:n].copy_(num_sampled[:n])
        self.num_rejected_buf[:n].copy_(num_rejected[:n])
        self.step_temperature.copy_(temperature[: self.step_temperature.shape[0]])
        self.step_seeds.copy_(seeds[: self.step_seeds.shape[0]])
        qsl, target_positions = input_batch.query_start_loc, input_batch.positions
        logits = None
        captured = (
            self.pre_graphs is not None
            and self.pre_graphs.graphs
            and n <= self.pre_graphs.sizes[-1]
        )
        # Log mode runs eager on every rank (the step log lives on rank 0 only): the
        # pre-draft stage broadcasts rank 0's Fore-D logits, and an eager [n] broadcast
        # cannot pair with a graph's padded one.
        if not captured or skip_attn_for_dummy_run or is_profile or self.config.log:
            logits = self._pre_draft(
                n, qsl, target_positions, last_sampled, next_prefill_tokens, temperature, seeds,
                prep=not (dummy_run and skip_attn_for_dummy_run),
            )  # fmt: skip
        else:
            self.pre_graphs.run(n)
        if not dummy_run:
            # The rows are final once the pre-draft stage has run: publish now,
            # so the next step's planner waits on this point, not the draft's end.
            self.stale.publish(input_batch.idx_mapping_np[:n][legal > 0])

        if (
            dummy_run and skip_attn_for_dummy_run
        ):  # memory profiling: no block tables yet
            self.model.precompute_and_store_context_kv(
                self.hidden_states[:num_target], self.context_positions[:num_target]
            )
            self._generate_backbone(n, n + draft_total, None, None, None)
            self._generate_tail(max(draft_total, 1))
            return self.draft_tokens[:n]

        # Context K/V: a graph keyed by target tokens. Dummy runs write no
        # cache: their slots are placeholders.
        graphs = self.context_graphs
        if (
            graphs is not None
            and graphs.graphs
            and num_target <= graphs.sizes[-1]
            and not (dummy_run or is_profile)
        ):
            graphs.run(num_target)
        else:
            self._context_kv(num_target, qsl, target_positions, store=not dummy_run)

        rows = n + draft_total
        desc, sync = dispatch_cg_and_sync_dp(
            self.backbone_graphs, n, rows, uniform_token_count=None, dp_size=self.dp_size, dp_rank=self.dp_rank,
            max_query_len=1 + self.max_width, need_eager=is_profile,
        )  # fmt: skip
        tokens_padded = desc.num_tokens
        attn = self._packed_attn_metadata(n, desc, draft_total, seq_upper)
        layers = build_slot_mappings_by_layer(
            self.block_tables.slot_mappings[:, :tokens_padded], self.kv_cache_config
        )
        self._prepare_eplb_forward(rows)
        if desc.cg_mode == CUDAGraphMode.FULL:
            self.backbone_graphs.run_fullgraph(desc)
        else:
            self._generate_backbone(
                n,
                tokens_padded,
                attn,
                layers,
                sync.num_tokens_across_dp if sync else None,
                desc.cg_mode,
            )
        if draft_total > 0:
            tail = (
                self.tail_graphs
            )  # captured after vLLM's warmup runs, which also draft
            if (
                tail is not None
                and tail.graphs
                and not is_profile
                and draft_total <= tail.sizes[-1]
            ):
                tail.run(draft_total)
            else:
                self._generate_tail(draft_total)

        if self.steplog is not None and not dummy_run:
            self.steplog.record(
                input_batch.req_ids, logits.float().cpu().numpy(), self._confidence[:n].cpu().numpy(),
                legal.copy(), num_sampled[:n].cpu().numpy(),
            )  # fmt: skip
        if not dummy_run:
            drafted = input_batch.idx_mapping_np[:n]
            legal_by_slot = np.zeros(self.max_num_reqs, dtype=np.int64)
            # Exact under host allocation, else an upper bound.
            legal_by_slot[drafted] = widths
            # Re-plan B at verify time only for planner-chosen D (not forced widths).
            planned = (
                self._planned
                if self.config.fixed_width is None and plan.rate > 0
                else None
            )
            self.pending = (plan, legal_by_slot, drafted.copy(), planned)
            self.reset_slots.clear()
            self._planned = None
            self.num_drafted_tokens = draft_total
        return self.draft_tokens[:n]

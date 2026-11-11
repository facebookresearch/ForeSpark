# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Portions adapted from vLLM v0.30.0's DFlash CUDA graphs
# (vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py).

"""CUDA graphs for the packed draft step.

  pre-draft  the Fore-D head, selection, packed query rows: plain graphs keyed by
             request count (every step, prefill chunks included)
  context    target-token positions and slots, then the draft's K/V: plain
             graphs keyed by target tokens, padding rows given pad slots
  backbone   attention + MLP over n + D packed rows: upstream's varlen decode
             graphs, keyed by total rows and replayable by any mix of 1..K+1
             rows per request (the device query_start_loc/seq_lens drive the
             attention)
  tail       LM head, selector, walk, confidence over the D sample rows: plain
             graphs keyed by sample count

Every key is host-known (the planner chose D), so dispatch never waits on the GPU.
"""

import copy
from collections.abc import Callable, Mapping

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import graph_capture
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.input_batch import InputBatch


class BackboneGraphs(CudaGraphManager):
    def __init__(self, vllm_config, device: torch.device, cudagraph_mode: CUDAGraphMode, max_query_len: int,
                 capture_sizes: list[int]):  # fmt: skip
        self._capture_sizes = capture_sizes
        super().__init__(
            vllm_config,
            device,
            cudagraph_mode,
            decode_query_len=max_query_len,
            varlen_decode=True,
        )

    def _init_candidates(self) -> None:
        # Our own sizes, not vLLM's: the manager reads its compilation config
        # only here, so a private copy changes nothing else.
        sizes = self._capture_sizes
        self.compilation_config = copy.copy(self.compilation_config)
        self.compilation_config.cudagraph_capture_sizes = sizes
        self.compilation_config.max_cudagraph_capture_size = sizes[-1] if sizes else 0
        super()._init_candidates()

    def capture(
        self,
        forward_fn: Callable,
        input_buffers,
        block_tables,
        attn_groups,
        kv_cache_config,
        max_model_len: int,
        causal: bool | Mapping[int, bool],
    ) -> None:
        def create_forward_fn(desc: BatchExecutionDescriptor, warmup: bool):
            num_tokens, num_reqs = desc.num_tokens, desc.num_reqs
            batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers)
            slot_mappings = block_tables.get_dummy_slot_mappings(num_tokens)
            # Built for the promised max query length (K+1), not the dummy
            # split, so the captured attention serves any packed layout.
            attn_metadata = build_attn_metadata(
                attn_groups=attn_groups,
                num_reqs=num_reqs,
                num_tokens=num_tokens,
                query_start_loc_gpu=batch.query_start_loc,
                query_start_loc_cpu=torch.from_numpy(batch.query_start_loc_np),
                max_query_len=desc.max_query_len,
                seq_lens=batch.seq_lens,
                max_seq_len=max_model_len,
                block_tables=block_tables.get_dummy_block_tables(num_reqs),
                slot_mappings=slot_mappings,
                kv_cache_config=kv_cache_config,
                for_cudagraph_capture=True,
                causal=causal,
            )
            layers = build_slot_mappings_by_layer(slot_mappings, kv_cache_config)
            return lambda cg_mode: forward_fn(
                num_reqs, num_tokens, attn_metadata, layers, None, cg_mode
            )

        super().capture(create_forward_fn, "Capturing Fore-D backbone graphs")


def bound_draft_splits(builder, max_rows: int) -> None:
    """Serve the draft graphs (up to `max_rows`, past vLLM's largest) from a
    FlashAttention builder whose split-KV bound holds for all of them.

    FA3 sizes the split buffers a graph captures by that bound, and leaves it
    unbounded past max_cudagraph_size, where replayed schedules could outgrow
    them. Past vLLM's own graphs the query tiles fill the GPU without splits,
    and 32-way buffers at 4k rows would cost GiBs of KV cache: one split.
    """
    if hasattr(builder, "_fore_d_split_rows"):
        return
    # No FA3 AOT schedule, as upstream builds proposer metadata (fast_build): it
    # lives in one buffer each build overwrites, so a graph replayed without a
    # rebuild reads another size's schedule. Upstream switches it off only for
    # sliding-window cache specs, which some targets' layouts (GLM-5.3) replace.
    builder.aot_schedule = False
    builder._fore_d_split_rows = rows = builder.max_cudagraph_size or 0
    splits, build = builder.max_num_splits, builder.build
    builder.max_cudagraph_size = max(rows, max_rows)

    def bounded(common_prefix_len, common_attn_metadata, *args, **kwargs):
        small = common_attn_metadata.num_actual_tokens <= rows
        builder.max_num_splits = splits if small else 1
        return build(common_prefix_len, common_attn_metadata, *args, **kwargs)

    builder.build = bounded


class SizedGraphs:
    """Plain CUDA graphs of `forward_fn(size)`, replayed at the smallest size >= the request."""

    def __init__(self, sizes: list[int], device: torch.device) -> None:
        self.sizes = sorted(s for s in sizes if s > 0)
        self.device = device
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}

    def bucket(self, samples: int) -> int:
        return next(s for s in self.sizes if s >= samples)

    def capture(self, forward_fn: Callable[[int], None]) -> None:
        pool = current_platform.get_global_graph_pool()
        with graph_capture(device=self.device):
            for size in reversed(self.sizes):
                forward_fn(size)  # warmup
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool, stream=torch.cuda.current_stream()):
                    forward_fn(size)
                self.graphs[size] = graph

    def run(self, samples: int) -> None:
        self.graphs[self.bucket(samples)].replay()

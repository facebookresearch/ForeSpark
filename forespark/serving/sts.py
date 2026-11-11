# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Per-position temperature scaling (STS) of drafter confidence.

Raw confidence heads are overconfident (ECE 10-16% for qwen14 DSpark, 1-5%
after scaling), which reorders the survival ranking that adaptive verification
spends its budget on. One temperature per draft position, fit by
`train/calibrate_sts.py` on held-out (logit, survived) pairs.

Apply  (FORESPARK_STS=calibration.json): DFlash2-conf divides inside its own model;
       DSpark gets a scaled subclass registered for its architecture.
Collect (FORESPARK_STS_COLLECT=<stem>, eager, adaptive verification off):
       confidence logits at proposal are paired, by request slot, with the
       accepted count one step later, and written to <stem>.<n>.npz.
"""

import atexit
import json
import math
import os
import threading
from pathlib import Path

import numpy as np
import torch

from ..config import STS_COLLECT_ENV, STS_ENV

DSPARK_ARCHITECTURE = "Qwen3DSparkModel"


def load_temperatures(path: str | Path, positions: int) -> list[float]:
    values = [float(t) for t in json.loads(Path(path).read_text())["temperatures"]]
    if len(values) != positions or not all(math.isfinite(t) and t > 0 for t in values):
        raise ValueError(
            f"{path}: expected {positions} positive temperatures, got {values}"
        )
    return values


class Collector:
    """Confidence logits per request slot, paired with the next accepted count."""

    FLUSH_EVERY = 128

    def __init__(self, stem: Path, positions: int, max_num_reqs: int) -> None:
        self.stem, self.positions = stem, positions
        self._by_slot = torch.full((max_num_reqs, positions), float("nan"))
        self._logits: list[np.ndarray] = []
        self._survived: list[np.ndarray] = []
        self._shard = self.pairs = 0
        self._lock = threading.Lock()

    def record_proposal(self, slots: torch.Tensor, logits: torch.Tensor) -> None:
        slots, logits = slots.long().cpu(), logits.detach().float().cpu()
        real = slots >= 0  # dummy runs (memory profiling, warmup) pad slots with -1
        with self._lock:
            self._by_slot.index_copy_(0, slots[real], logits[real])

    def record_verification(
        self, slots: torch.Tensor, drafted: np.ndarray, accepted: torch.Tensor
    ) -> None:
        slots = slots.long().cpu()
        full = np.flatnonzero(
            (drafted == self.positions) & (slots.numpy() >= 0)
        )  # untrimmed blocks of real requests
        if not full.size:
            return
        rows = torch.as_tensor(full)
        with self._lock:
            logits = self._by_slot.index_select(0, slots[rows]).numpy()
        usable = ~np.isnan(logits).any(axis=1)
        counts = accepted.long().cpu()[rows].clamp(0, self.positions).numpy()
        survived = (np.arange(self.positions)[None, :] < counts[:, None]).astype(
            np.float32
        )
        with self._lock:
            self._logits.append(logits[usable])
            self._survived.append(survived[usable])
            self.pairs += int(usable.sum())
            if len(self._logits) >= self.FLUSH_EVERY:
                self._flush()

    def _flush(self) -> None:
        if self._logits:
            self.stem.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                f"{self.stem}.{self._shard}.npz",
                confidence_logits=np.concatenate(self._logits),
                survived=np.concatenate(self._survived),
            )
            self._logits.clear()
            self._survived.clear()
            self._shard += 1

    def close(self) -> None:
        with self._lock:
            self._flush()
        print(
            f"[sts] {self.pairs} pairs in {self._shard} shard(s) at {self.stem}.*.npz",
            flush=True,
        )


_collector: Collector | None = None


def collector(positions: int, max_num_reqs: int) -> Collector:
    global _collector
    if _collector is None:
        _collector = Collector(
            Path(os.environ[STS_COLLECT_ENV]), positions, max_num_reqs
        )
        atexit.register(_collector.close)
    return _collector


def _is_rank_zero() -> bool:
    from vllm.distributed import get_tensor_model_parallel_rank

    return get_tensor_model_parallel_rank() == 0


def install() -> None:
    if os.environ.get(STS_ENV):
        from vllm import ModelRegistry

        ModelRegistry.register_model(
            DSPARK_ARCHITECTURE,
            "forespark.serving.dspark_sts:StsQwen3DSparkForCausalLM",
        )
    if os.environ.get(STS_COLLECT_ENV):
        _install_dspark_collection()
        _install_verification_recorder()


def _install_dspark_collection() -> None:
    """DSpark's head inputs are locals of `_sample_sequential`; recompute them after it."""
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    original = DSparkSpeculator._sample_sequential

    def sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        original(self, num_reqs, head_hidden)
        if not _is_rank_zero():
            return
        steps = self.num_speculative_steps
        with torch.no_grad():
            sample_hidden = head_hidden[self.sample_indices[: num_reqs * steps]]
            anchor = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
            previous = torch.cat(
                [anchor.view(num_reqs, 1), self.draft_tokens[:num_reqs, : steps - 1]],
                dim=1,
            )
            logits = self.model.model.confidence_head(
                sample_hidden, self.model.markov_embed(previous.reshape(-1))
            )
        slots = self.sample_idx_mapping[: num_reqs * steps].view(num_reqs, steps)[:, 0]
        collector(steps, self.max_num_reqs).record_proposal(
            slots, logits.view(num_reqs, steps)
        )

    DSparkSpeculator._sample_sequential = sample_sequential


def _install_verification_recorder() -> None:
    """Report every verification's accepted count to the collector."""
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    original = RejectionSampler.__call__

    def __call__(self, logits, input_batch, draft_logits=None):
        output = original(self, logits, input_batch, draft_logits)
        drafted = input_batch.num_draft_tokens_per_req
        if _collector is not None and drafted is not None and _is_rank_zero():
            n = input_batch.num_reqs
            _collector.record_verification(
                input_batch.idx_mapping[:n],
                np.asarray(drafted[:n]),
                output.num_sampled - 1,
            )
        return output

    RejectionSampler.__call__ = __call__

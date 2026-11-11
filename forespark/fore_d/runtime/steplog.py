# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Rank-0 log of forced-width steps (every draft verified): the input of train/calibrate_fore_d.py.

Per step and request: the raw Fore-D logits, the realized confidence row of
its draft, its width, and num_sampled (accepted + 1) of the verification that
just finished -- which scores the request's previous draft.
Drafts are paired with their outcome offline (same request, next step). A log
run syncs every step; it measures acceptance, not speed.
"""

import atexit
from pathlib import Path

import numpy as np


class StepLog:
    def __init__(self, directory: Path, flush_every: int = 512) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory, self.flush_every = directory, flush_every
        self.records: list[tuple] = []
        self.step = 0
        self.chunk = 0
        atexit.register(self.flush)

    def record(self, req_ids, fore_d_logits, confidence, legal, num_sampled) -> None:
        self.records.append(
            (self.step, list(req_ids), fore_d_logits, confidence, legal, num_sampled)
        )
        self.step += 1
        if len(self.records) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.records:
            return
        steps, req_ids, logits, confidence, legal, num_sampled = zip(
            *self.records, strict=True
        )
        np.savez(
            self.directory / f"steps.{self.chunk:05d}.npz",
            step=np.concatenate(
                [np.full(len(r), s) for s, r in zip(steps, req_ids, strict=True)]
            ),
            batch=np.concatenate([np.full(len(r), len(r)) for r in req_ids]),
            req=np.concatenate([np.asarray(r, dtype=str) for r in req_ids]),
            fore_d_logits=np.concatenate(logits),
            confidence=np.concatenate(confidence),
            legal=np.concatenate(legal),
            num_sampled=np.concatenate(num_sampled),
        )
        self.chunk += 1
        self.records.clear()

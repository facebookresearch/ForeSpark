# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Per-request-slot rows copied to the host each step and read one step later.

The planner decides step g's totals from the rows the GPU produced at step
g-1. The copy is started right after they are written and waited on a whole
step later, when it has long since landed, so the host never stalls on the
GPU. The lag is fixed (not "newest ready"): every TP rank reads the same step,
so every rank plans identically without a collective.
"""

import numpy as np
import torch


class StaleRows:
    def __init__(self, num_slots: int, width: int, device: torch.device) -> None:
        # One extra row that padding writes land in (kernels/select.py).
        self.device_rows = torch.ones(
            num_slots + 1, width, dtype=torch.float32, device=device
        )
        self.num_slots = num_slots
        self._host = [
            torch.ones(num_slots, width, dtype=torch.float32).pin_memory()
            for _ in range(2)
        ]
        self._events = [torch.cuda.Event() for _ in range(2)]
        self._latest: int | None = None
        # Slots whose host row is not theirs yet (new request, or its copy is
        # still in flight); the planner substitutes a prior for them.
        self.missing = np.ones(num_slots, dtype=bool)
        self._published_valid: np.ndarray | None = None

    def reset(self, slot: int) -> None:
        """A new request took `slot`: whatever the host holds there is not its."""
        self.missing[slot] = True
        if self._published_valid is not None:
            self._published_valid = self._published_valid[self._published_valid != slot]

    def read(self) -> np.ndarray:
        """Host rows as of the last publish (a view: use before the next publish)."""
        if self._latest is None:
            return self._host[0].numpy()
        self._events[self._latest].synchronize()
        if self._published_valid is not None:
            self.missing[self._published_valid] = False
            self._published_valid = None
        return self._host[self._latest].numpy()

    def publish(self, valid_slots: np.ndarray) -> None:
        """Start copying `device_rows` to the host; `valid_slots` got real rows."""
        # On the current stream, behind the step that wrote the rows: a 14 KB
        # copy costs the GPU nothing, and a side stream costs the host more.
        target = 0 if self._latest != 0 else 1
        self._host[target].copy_(self.device_rows[: self.num_slots], non_blocking=True)
        self._events[target].record()
        self._latest = target
        self._published_valid = valid_slots


class PinnedStaging:
    """Double-buffered pinned host memory for small per-step host-to-device
    copies (a fresh pin_memory() per step costs more host time than the copy).
    A buffer is rewritten only after the copy that last read it (two steps
    earlier) has completed."""

    def __init__(self, size: int, dtype: torch.dtype = torch.int32) -> None:
        self._host = [torch.zeros(size, dtype=dtype).pin_memory() for _ in range(2)]
        self._events = [torch.cuda.Event() for _ in range(2)]
        self._index = 0

    def next(self) -> np.ndarray:
        self._index ^= 1
        self._events[self._index].synchronize()
        return self._host[self._index].numpy()

    def copy_to(self, out: torch.Tensor, length: int) -> None:
        out[:length].copy_(self._host[self._index][:length], non_blocking=True)
        self._events[self._index].record()

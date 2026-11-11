# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Exact decode-start barrier for pure-decode benchmarks (`sharegpt-decode`).

A fixed set of requests is admitted and prefilled; each is parked the moment
its prefill completes, so none starts decoding early. When all sit at the
decode boundary they are released together, the first all-decode batch is
scheduled, and the engine holds it until the benchmark client -- which
snapshots the server counters -- says go. The window therefore opens on an
exact all-decode step, every request at output position 1, in every arm.

File protocol in FORESPARK_DECODE_BARRIER_DIR:
  arm.json       client: {"run_id", "request_ids" (engine ids), "timeout"}
  ready.json     engine: the first all-decode batch is scheduled and held
  release.json   client: {"run_id"} -- start
  released.json  engine: running
  error.json     engine: {"run_id", "message"}
"""

import json
import os
import time
from functools import wraps
from pathlib import Path

from ..config import DECODE_BARRIER_ENV

PARKED = (1 << 63) - 1  # next_decode_eligible_step that is never reached


def write_message(root: Path, name: str, payload: dict) -> None:
    tmp = root / f".{name}.tmp"
    tmp.write_text(json.dumps(payload))
    tmp.replace(root / name)


def read_message(root: Path, name: str) -> dict | None:
    try:
        return json.loads((root / name).read_text())
    except FileNotFoundError:
        return None


class Barrier:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.state = "idle"  # idle -> admitting -> prefilling -> held -> released
        self.run_id: str | None = None
        self.request_ids: tuple[str, ...] = ()
        self.deadline = 0.0
        self.parked: dict[str, int] = {}
        self.expect_candidate = False
        self.saved_queue_depth: int | None = None

    def step(self, engine, run, empty):
        """Wraps one EngineCore step. `empty` is returned while admitting."""
        try:
            if self.state in ("idle", "released"):
                arm = read_message(self.root, "arm.json")
                if arm is None or arm["run_id"] == self.run_id:
                    return run()
                self._arm(arm)
            live = set(engine.scheduler.requests)
            if live - set(self.request_ids):
                raise RuntimeError(
                    f"foreign requests during the barrier: {sorted(live - set(self.request_ids))}"
                )
            if time.monotonic() > self.deadline:
                raise TimeoutError("decode barrier timed out")
            if self.state == "admitting":
                if live != set(self.request_ids):
                    return empty
                self._begin(engine)
            if (
                self.state == "prefilling"
                and not self.expect_candidate
                and self._all_at_boundary(engine.scheduler)
            ):
                eligible = engine.scheduler.current_step + 1
                for rid in self.request_ids:
                    engine.scheduler.requests[rid].next_decode_eligible_step = eligible
                self.expect_candidate = True
            return run()
        except Exception as error:
            self._fail(engine, error)
            raise

    def _arm(self, arm: dict) -> None:
        self.run_id = arm["run_id"]
        self.request_ids = tuple(arm["request_ids"])
        self.deadline = time.monotonic() + float(arm.get("timeout", 60.0))
        self.parked.clear()
        self.expect_candidate = False
        self.state = "admitting"

    def _begin(self, engine) -> None:
        # Depth 1 while prefilling, so parking takes effect on the very next step.
        self.saved_queue_depth = engine.batch_queue_size
        engine.batch_queue_size = 1
        scheduler = engine.scheduler
        schedule = scheduler.schedule

        @wraps(schedule)
        def wrapped(*args, **kwargs):
            output = schedule(*args, **kwargs)
            self._after_schedule(engine, output)
            return output

        scheduler.schedule = wrapped
        self.state = "prefilling"

    def _after_schedule(self, engine, output) -> None:
        if self.state != "prefilling":
            return
        scheduler = engine.scheduler
        if self.expect_candidate:
            self._hold(engine)
            return
        before = {r.req_id: r.num_computed_tokens for r in output.scheduled_new_reqs}
        cached = output.scheduled_cached_reqs
        before.update(zip(cached.req_ids, cached.num_computed_tokens, strict=True))
        for rid in output.num_scheduled_tokens:
            request = scheduler.requests[rid]
            if before[rid] < request.num_prompt_tokens <= request.num_computed_tokens:
                self.parked.setdefault(rid, request.next_decode_eligible_step)
                request.next_decode_eligible_step = PARKED

    def _all_at_boundary(self, scheduler) -> bool:
        ready = True
        for rid in self.request_ids:
            r = scheduler.requests[rid]
            if r.num_preemptions:
                raise RuntimeError(
                    "preemption before the decode boundary; lower concurrency or raise memory"
                )
            ready &= (
                r.status.name == "RUNNING"
                and r.num_computed_tokens == r.num_prompt_tokens
                and r.num_output_tokens == 1
                and r.num_output_placeholders == 0
                and not r.is_prefill_chunk
            )
        return ready

    def _hold(self, engine) -> None:
        self._restore(engine)
        self.state = "held"
        write_message(self.root, "ready.json", {"run_id": self.run_id})
        while time.monotonic() < self.deadline:
            release = read_message(self.root, "release.json")
            if release is not None and release.get("run_id") == self.run_id:
                self.state = "released"
                write_message(self.root, "released.json", {"run_id": self.run_id})
                return
            time.sleep(0.005)
        raise TimeoutError("decode barrier hold timed out")

    def _restore(self, engine) -> None:
        if self.saved_queue_depth is not None:
            engine.batch_queue_size = self.saved_queue_depth
            self.saved_queue_depth = None
        engine.scheduler.__dict__.pop("schedule", None)
        for rid, eligible in self.parked.items():
            if rid in engine.scheduler.requests:
                engine.scheduler.requests[rid].next_decode_eligible_step = eligible
        self.parked.clear()

    def _fail(self, engine, error: Exception) -> None:
        if self.state != "error":
            self._restore(engine)
            self.state = "error"
            write_message(
                self.root, "error.json", {"run_id": self.run_id, "message": repr(error)}
            )


def install() -> None:
    from vllm.v1.engine.core import EngineCore

    root = Path(os.environ[DECODE_BARRIER_ENV])
    step, queued_step = EngineCore.step, EngineCore.step_with_batch_queue

    def barrier(engine) -> Barrier:
        if not hasattr(engine, "_decode_barrier"):
            engine._decode_barrier = Barrier(root)
        return engine._decode_barrier

    def patched_step(engine, *args, **kwargs):
        return barrier(engine).step(
            engine, lambda: step(engine, *args, **kwargs), ({}, False)
        )

    def patched_queued_step(engine, *args, **kwargs):
        return barrier(engine).step(
            engine, lambda: queued_step(engine, *args, **kwargs), (None, False)
        )

    EngineCore.step = patched_step
    EngineCore.step_with_batch_queue = patched_queued_step

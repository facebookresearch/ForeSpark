# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The exact decode-start barrier's state machine, against a fake engine."""

from types import SimpleNamespace

import pytest
from forespark.serving.decode_barrier import (
    Barrier,
    PARKED,
    read_message,
    write_message,
)


class Request:
    def __init__(self, rid: str, prompt: int) -> None:
        self.req_id, self.num_prompt_tokens, self.num_computed_tokens = rid, prompt, 0
        self.num_output_tokens = self.num_output_placeholders = self.num_preemptions = 0
        self.next_decode_eligible_step = 0
        self.status = SimpleNamespace(name="RUNNING")
        self.is_prefill_chunk = True


class Scheduler:
    """Prefills `chunk` tokens per request per step; a finished prefill samples one token."""

    def __init__(self, chunk: int) -> None:
        self.requests: dict[str, Request] = {}
        self.current_step, self.chunk, self.decoded = 0, chunk, []

    def schedule(self):
        new, scheduled = [], {}
        for r in self.requests.values():
            if r.num_computed_tokens < r.num_prompt_tokens:
                new.append(
                    SimpleNamespace(
                        req_id=r.req_id, num_computed_tokens=r.num_computed_tokens
                    )
                )
                r.num_computed_tokens = min(
                    r.num_prompt_tokens, r.num_computed_tokens + self.chunk
                )
                scheduled[r.req_id] = self.chunk
                if r.num_computed_tokens == r.num_prompt_tokens:
                    r.num_output_tokens, r.is_prefill_chunk = 1, False
            elif r.next_decode_eligible_step <= self.current_step:
                scheduled[r.req_id] = 1
                self.decoded.append(r.req_id)
        cached = SimpleNamespace(req_ids=[], num_computed_tokens=[])
        return SimpleNamespace(
            scheduled_new_reqs=new,
            scheduled_cached_reqs=cached,
            num_scheduled_tokens=scheduled,
        )


def engine(chunk: int = 4) -> SimpleNamespace:
    return SimpleNamespace(scheduler=Scheduler(chunk), batch_queue_size=2)


def run(e):
    def step():
        e.scheduler.schedule()
        e.scheduler.current_step += 1
        return "ran"

    return step


def test_requests_are_released_together_at_the_decode_boundary(tmp_path):
    e, barrier = engine(), Barrier(tmp_path)
    assert barrier.step(e, run(e), "empty") == "ran"  # not armed: a normal step
    write_message(
        tmp_path, "arm.json", {"run_id": "r1", "request_ids": ["a", "b"], "timeout": 10}
    )
    e.scheduler.requests["a"] = Request("a", 4)
    assert (
        barrier.step(e, run(e), "empty") == "empty"
    )  # admitting until every request is in
    e.scheduler.requests["b"] = Request("b", 8)
    write_message(
        tmp_path, "release.json", {"run_id": "r1"}
    )  # the client's go, seen once the batch is held
    for _ in range(6):
        barrier.step(e, run(e), "empty")
        if barrier.state == "released":
            break
        # Nobody decodes before the boundary: a finished prefill stays parked.
        assert not e.scheduler.decoded
        if (
            e.scheduler.requests["a"].num_computed_tokens == 4
            and barrier.state == "prefilling"
        ):
            assert e.scheduler.requests["a"].next_decode_eligible_step in (
                PARKED,
                e.scheduler.current_step,
            )
    assert barrier.state == "released"
    assert read_message(tmp_path, "ready.json") == {"run_id": "r1"}
    assert read_message(tmp_path, "released.json") == {"run_id": "r1"}
    assert (
        e.batch_queue_size == 2 and "schedule" not in e.scheduler.__dict__
    )  # engine restored
    barrier.step(e, run(e), "empty")
    assert sorted(e.scheduler.decoded[:2]) == [
        "a",
        "b",
    ]  # the first decode step holds both


def test_foreign_request_fails_loudly(tmp_path):
    e, barrier = engine(), Barrier(tmp_path)
    write_message(
        tmp_path, "arm.json", {"run_id": "r2", "request_ids": ["a"], "timeout": 10}
    )
    e.scheduler.requests.update(a=Request("a", 4), intruder=Request("intruder", 4))
    with pytest.raises(RuntimeError, match="foreign"):
        barrier.step(e, run(e), "empty")
    assert read_message(tmp_path, "error.json")["run_id"] == "r2"


def test_times_out(tmp_path):
    e, barrier = engine(), Barrier(tmp_path)
    write_message(
        tmp_path, "arm.json", {"run_id": "r3", "request_ids": ["a"], "timeout": -1}
    )
    e.scheduler.requests["a"] = Request("a", 4)
    with pytest.raises(TimeoutError):
        barrier.step(e, run(e), "empty")

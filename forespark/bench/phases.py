# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The measurement shapes behind the benchmark phases.

fixed_count     acceptance: every prompt once, one counter window per category
                (acceptance is a property of the drafter, not load)
steady_state    sharegpt (E2E): closed loop at fixed concurrency, warm up to a
                plateau on held-out prompts, measure a window whose requests
                start at the same prompt in every arm, keep load on while they
                drain
decode_spans    sharegpt-decode: fixed batches released together after every
                request has its first token (exact barrier), each request
                timed over the same token span of its answer, repeated over
                disjoint prompt sets until enough answers are measured
open_loop       sharegpt-rate (SLO): Poisson arrivals at a fixed rate, every
                measured request scored against TTFT and TPOT objectives
"""

import asyncio
import os
import random
import secrets
import statistics
import threading
import time
from pathlib import Path

from ..serving.decode_barrier import read_message, write_message
from . import protocol as P


def request_body(
    args, row: dict, *, ignore_eos: bool = False, request_id: str | None = None
) -> dict:
    """Decoding is pinned so the drafter is the only difference between arms."""
    body = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": 1.0,
        "top_k": -1,  # unset, vLLM takes the model's generation_config default (Qwen3: 20)
        "min_p": 0.0,
    }
    if "prompt_token_ids" in row:
        body["prompt"] = row["prompt_token_ids"]
    else:
        body["messages"] = row["messages"]
        if args.template_kwargs:
            body["chat_template_kwargs"] = args.template_kwargs
    if ignore_eos:
        body["ignore_eos"] = True
    if request_id is not None:
        body["request_id"] = request_id
    return body


def request_url(base_url: str, row: dict) -> str:
    return f"{base_url}/v1/{'completions' if 'prompt_token_ids' in row else 'chat/completions'}"


# --- acceptance --------------------------------------------------------------


async def _submit_all(
    args, rows: list[dict], concurrency: int, extra: dict | None = None
) -> list[P.Completion | None]:
    gate = asyncio.Semaphore(concurrency)
    async with P.session(concurrency, args.request_timeout) as client:

        async def one(row: dict) -> P.Completion | None:
            async with gate:
                try:
                    return await P.stream(
                        client,
                        request_url(args.base_url, row),
                        request_body(args, row) | (extra or {}),
                    )
                except Exception:  # noqa: BLE001 - one bad prompt must not void the run
                    return None

        return await asyncio.gather(*(one(row) for row in rows))


def _summarise(
    args,
    results: list[P.Completion],
    failed: int,
    before: dict,
    after: dict,
    window: float,
    c: int,
) -> dict:
    tokens = P.delta(before, after, "vllm:generation_tokens_total")
    stats = {
        "concurrency": c,
        "window_seconds": window,
        "completed": len(results),
        "failed": failed,
        "output_tokens": tokens,
        "output_tokens_per_second": tokens / window if window else 0.0,
        "preemptions": P.preemptions(before, after),
        "prefix_cache_hit_rate": P.prefix_cache_hit_rate(before, after),
        "latency_p50_ms": P.percentile([r.elapsed * 1000 for r in results], 0.5),
        "latency_p90_ms": P.percentile([r.elapsed * 1000 for r in results], 0.9),
        "latency_p99_ms": P.percentile([r.elapsed * 1000 for r in results], 0.99),
        "median_tpot_ms": P.percentile([r.tpot * 1000 for r in results if r.tpot], 0.5),
        "median_ttft_ms": P.percentile([r.ttft * 1000 for r in results], 0.5),
        "p99_ttft_ms": P.percentile([r.ttft * 1000 for r in results], 0.99),
    }
    return stats | (P.acceptance(before, after) if args.draft else {})


def fixed_count(args, prompts: list[dict], concurrency: int) -> dict:
    """Every prompt once. Categories run one at a time because vLLM's counters
    are server-global: serialising is the only way to scope them."""
    groups: dict[str, list[dict]] = {}
    for row in prompts:
        groups.setdefault(row.get("category", "?"), []).append(row)
    opened, started = P.counters(args.base_url), time.monotonic()
    by_category, everything, failed_total = [], [], 0
    for name in sorted(groups):
        rows = groups[name]
        width = min(concurrency, len(rows))
        before, t0 = P.counters(args.base_url), time.monotonic()
        outcomes = asyncio.run(_submit_all(args, rows, width))
        results = [o for o in outcomes if o is not None]
        failed = len(outcomes) - len(results)
        stats = _summarise(
            args,
            results,
            failed,
            before,
            P.counters(args.base_url),
            time.monotonic() - t0,
            width,
        )
        by_category.append({"category": name, **stats})
        everything += results
        failed_total += failed
    width = max(min(concurrency, len(rows)) for rows in groups.values())
    stats = _summarise(
        args,
        everything,
        failed_total,
        opened,
        P.counters(args.base_url),
        time.monotonic() - started,
        width,
    )
    return stats | {"by_category": by_category}


# --- sharegpt (E2E) --------------------------------------------------------------

WARMUP_EVERY = 8  # one conversation (or row) in eight is held out for warmup
MIN_WINDOW_CAP_SECONDS = 1800.0  # the longest --min-window-requests holds a window


def split_warmup(prompts: list[dict]) -> tuple[list[dict], list[dict]]:
    """Held-out warmup rows and the measured rest, split by conversation so a
    measured turn never reuses a warmup turn's history."""

    def held_out(index: int, row: dict) -> bool:
        return row.get("conversation", index) % WARMUP_EVERY == WARMUP_EVERY - 1

    warm = [row for i, row in enumerate(prompts) if held_out(i, row)]
    measured = [row for i, row in enumerate(prompts) if not held_out(i, row)]
    if not warm or not measured:
        return prompts, prompts
    return warm, measured


def _stream_client(args, window: P.Window, warm: P.Cycler, measured: P.Cycler):
    """One request after another from the prompt stream (held-out rows until the window opens)."""

    async def worker(client, stop: asyncio.Event) -> None:
        while not stop.is_set():
            progress = window.dispatch()  # a request belongs to the phase it started in
            row = measured.next() if window.started else warm.next()
            try:
                url, body = request_url(args.base_url, row), request_body(args, row)
                window.finish(
                    progress, await P.stream(client, url, body, progress=progress)
                )
            except Exception as error:  # noqa: BLE001 - a failed request is a data point
                window.finish(progress, None, type(error).__name__)

    return worker


async def _measure(args, window: P.Window) -> dict:
    """The window's gauges: over --measure-seconds, then on until --min-window-requests
    have started (within a cap)."""
    minimum = args.min_window_requests
    if minimum:
        stop, cap = threading.Event(), MIN_WINDOW_CAP_SECONDS
        gauges = asyncio.to_thread(P.sample_gauges, args.base_url, cap, 2.0, stop)
        sampler, deadline = asyncio.create_task(gauges), time.monotonic() + cap
        await asyncio.sleep(args.measure_seconds)
        while window.dispatched < minimum and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
        stop.set()
        return await sampler | {"min_window_requests_met": window.dispatched >= minimum}
    return await asyncio.to_thread(P.sample_gauges, args.base_url, args.measure_seconds)


def _lag_stats(lags: list[float]) -> dict[str, float]:
    return (
        {
            "client_lag_p99_ms": 1000.0 * P.percentile(lags, 0.99),
            "client_lag_max_ms": 1000.0 * max(lags),
        }
        if lags
        else {}
    )


async def _windowed(args, concurrency: int, window: P.Window, worker):
    """`concurrency` closed-loop clients: warm up to a plateau, measure a window,
    keep load on while its requests drain, censor the rest."""
    stop = asyncio.Event()
    async with P.session(concurrency, args.request_timeout) as client:
        pool = [asyncio.create_task(worker(client, stop)) for _ in range(concurrency)]
        try:
            warmup = await asyncio.to_thread(
                P.await_steady,
                args.base_url,
                concurrency,
                args.warmup_seconds,
                args.warmup_timeout,
            )
            before, started = (
                await asyncio.to_thread(P.counters, args.base_url),
                time.monotonic(),
            )
            window.open()
            probe_stop = asyncio.Event()
            probe = asyncio.create_task(P.loop_lag(probe_stop))
            gauges = await _measure(args, window)
            after, elapsed = (
                await asyncio.to_thread(P.counters, args.base_url),
                time.monotonic() - started,
            )
            window.close()
            gauges["windows"] = [[started, started + elapsed]]
            gauges |= await asyncio.to_thread(
                P.gpu_state, args.gpus
            )  # load stays on through the drain
            probe_stop.set()
            gauges |= _lag_stats(await probe)
            gauges["foreign_gpu_processes"] = P.foreign_gpu_processes(
                getattr(args, "server_pgid", None)
            )
            # Workers keep loading the server while the window drains, so the
            # window's latencies are not flattered by an emptying queue.
            drained = await asyncio.to_thread(window.await_drain, args.drain_seconds)
            window.censor()
        finally:
            stop.set()
            for task in pool:
                task.cancel()
            await asyncio.gather(*pool, return_exceptions=True)
    await P.await_idle(args.base_url, args.drain_seconds)
    return before, after, elapsed, warmup, gauges, drained


def steady_state(args, prompts: list[dict], concurrency: int) -> dict:
    warm_rows, measured_rows = split_warmup(prompts)
    warm, measured = P.Cycler(warm_rows), P.Cycler(measured_rows)
    if args.prefix_caching == "on":  # no free hits on what an earlier point sent
        P.post(f"{args.base_url}/reset_prefix_cache")
    window = P.Window()
    worker = _stream_client(args, window, warm, measured)
    before, after, elapsed, warmup, gauges, drained = asyncio.run(
        _windowed(args, concurrency, window, worker)
    )
    tokens = P.delta(before, after, "vllm:generation_tokens_total")
    stats = {
        "concurrency": concurrency,
        "window_seconds": elapsed,
        "prompt_wraps": measured.wraps,
        "output_tokens": tokens,
        "output_tokens_per_second": tokens / elapsed if elapsed else 0.0,
        "preemptions": P.preemptions(before, after),
        "prefix_cache_hit_rate": P.prefix_cache_hit_rate(before, after),
        "drained": drained,
        "min_window_requests": args.min_window_requests,
        "host_load_1m": os.getloadavg()[0],
        **warmup,
        **gauges,
        **window.summary(),
    }
    return stats | (P.acceptance(before, after) if args.draft else {})


# --- sharegpt-decode -----------------------------------------------------------


def nearest(prompts: list[dict], count: int, target_chars: int = 1024) -> list[int]:
    """Indices of the `count` prompts closest to `target_chars` of content, closest first."""

    def distance(index: int) -> int:
        return abs(
            sum(len(m.get("content", "")) for m in prompts[index]["messages"])
            - target_chars
        )

    return sorted(range(len(prompts)), key=distance)[:count]


def decode_rows(
    prompts: list[dict], count: int, screened: list[int] | None = None
) -> list[dict]:
    """A stable, nested set of prompts (same rows in every arm): the screened ones
    (the target's answer outruns the span), else the nearest to 1k characters."""
    order = screened if screened is not None else nearest(prompts, count)
    if len(order) < count:
        raise ValueError(
            f"decode spans need {count} prompts, only {len(order)} available"
        )
    return [prompts[index] for index in order[:count]]


def answer_lengths(args, rows: list[dict], concurrency: int) -> list[int | None]:
    """Each row's answer length in tokens on the server (None where the request failed)."""
    return [
        o.tokens if o is not None else None
        for o in asyncio.run(_submit_all(args, rows, concurrency))
    ]


def span_end(
    token_ids: list[int], span: int, stops: frozenset[int]
) -> tuple[int, bool]:
    """(decode tokens in a request's span, whether its answer ended inside it).

    token_ids[0] is the first token (prefill's, held at the barrier); the span
    is the next `span` tokens, cut at an end-of-turn token, which ignore_eos
    streams past.
    """
    for index, token in enumerate(token_ids[: span + 1]):
        if token in stops:
            return index, True
    return span, False


def span_rate(
    progress: P.Progress, span: int, stops: frozenset[int]
) -> tuple[int, float, bool] | None:
    """(tokens, tokens/s, ended naturally) over one request's span: from the
    first chunk after the release to the chunk completing the span. None when
    the span is empty or never completed."""
    tokens, natural = span_end(progress.token_ids, span, stops)
    need = tokens + 1  # running counts include the first token
    released = [(t, n) for t, n in progress.arrivals if n > 1]
    if tokens < 2 or not released:
        return None
    t0, n0 = released[0]
    done = next(((t, n) for t, n in released if n >= need), None)
    if done is None or n0 >= need or done[0] <= t0:
        return None
    t1, n1 = done
    # Every token of the completing chunk arrived by t1; past a natural end they are post-answer text.
    return tokens, ((need if natural else n1) - n0) / (t1 - t0), natural


async def _launch(
    args, client, url: str, rows: list[dict], request_ids: list[str], progress=None
):
    # Token ids in every chunk: the engine holds right after the first token,
    # and a first token that ends mid UTF-8 sequence streams as empty text.
    firsts = [asyncio.Event() for _ in rows]
    tracks = progress or [None] * len(rows)
    tasks = [
        asyncio.create_task(
            P.stream(
                client,
                url,
                request_body(args, row, ignore_eos=True, request_id=rid)
                | {"return_token_ids": True},
                first,
                track,
            )  # fmt: skip
        )
        for row, rid, first, track in zip(
            rows, request_ids, firsts, tracks, strict=True
        )
    ]
    return tasks, firsts


async def _await_spans(
    progress: list[P.Progress], tasks, span: int, stops: frozenset[int], timeout: float
) -> bool:
    """Until every request has streamed its span (or its stream ended); False on timeout."""
    need, scanned = [span + 1] * len(progress), [0] * len(progress)
    pending, deadline = set(range(len(progress))), time.monotonic() + timeout
    while pending:
        for i in list(pending):
            ids = progress[i].token_ids
            limit = min(len(ids), need[i])
            if (
                end := next(
                    (j for j in range(scanned[i], limit) if ids[j] in stops), None
                )
            ) is not None:
                need[i] = end + 1
            scanned[i] = limit
            if len(ids) >= need[i] or tasks[i].done():
                pending.discard(i)
        if pending and time.monotonic() >= deadline:
            return False
        await asyncio.sleep(
            0.005 if len(pending) < 64 else 0.02
        )  # spare the client's loop at wide batches
    return True


async def _cancel(tasks) -> None:
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def _wait_for(root: Path, name: str, run_id: str, tasks, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (error := read_message(root, "error.json")) and error.get(
            "run_id"
        ) == run_id:
            raise RuntimeError(f"engine decode barrier failed: {error['message']}")
        if done := [task for task in tasks if task.done()]:
            raise RuntimeError(
                f"a plateau request ended before the decode barrier: {done[0].exception()!r}"
            )
        if (message := read_message(root, name)) and message.get("run_id") == run_id:
            return
        await asyncio.sleep(0.01)
    raise TimeoutError(f"timed out waiting for {name}")


async def _boundary_counters(args, prefill_before: dict, concurrency: int) -> dict:
    """Counters at the held boundary: exactly one token per request so far.

    Server metrics publish asynchronously, so poll until they catch up.
    """
    deadline = time.monotonic() + args.barrier_timeout
    while time.monotonic() < deadline:
        snapshot = await asyncio.to_thread(P.counters, args.base_url)
        generated = P.delta(prefill_before, snapshot, "vllm:generation_tokens_total")
        if generated == concurrency:
            return snapshot
        if generated > concurrency:
            raise RuntimeError(
                f"{generated} tokens generated before release, expected {concurrency}"
            )
        await asyncio.sleep(0.05)
    raise TimeoutError("server counters never reached the decode boundary")


async def _decode_episode(
    args, url: str, root: Path, rows: list[dict], stops: frozenset[int]
) -> dict:
    """One fixed batch: prefill, hold at the first token, release together,
    stream until every request has its span."""
    concurrency = len(rows)
    if not await P.await_idle(args.base_url, args.drain_seconds):
        raise RuntimeError("server did not go idle before a decode episode")
    if args.prefix_caching == "on":
        P.post(
            f"{args.base_url}/reset_prefix_cache"
        )  # every batch prefills the same way in every arm

    # A fresh session per batch: a stream that finished on its own leaves a
    # pooled keep-alive connection, and if the server closes it while the calls
    # above block the loop, it still looks open and the next write fails.
    async with P.session(concurrency, args.request_timeout) as client:
        run_id = secrets.token_hex(16)
        wire_ids = [f"decode-{run_id}-{i:04d}" for i in range(concurrency)]
        prefix = "cmpl-" if "prompt_token_ids" in rows[0] else "chatcmpl-"
        engine_ids = [
            f"{prefix}{rid}-0" if prefix == "cmpl-" else f"{prefix}{rid}"
            for rid in wire_ids
        ]
        write_message(
            root,
            "arm.json",
            {
                "run_id": run_id,
                "request_ids": engine_ids,
                "timeout": args.barrier_timeout,
            },
        )
        progress = [P.Progress() for _ in rows]
        prefill_before, prefill_started = P.counters(args.base_url), time.monotonic()
        tasks, firsts = await _launch(args, client, url, rows, wire_ids, progress)
        stop, sampler, gauges = threading.Event(), None, {}
        try:
            await _wait_for(root, "ready.json", run_id, tasks, args.barrier_timeout)
            await asyncio.wait_for(
                asyncio.gather(*(f.wait() for f in firsts)), args.barrier_timeout
            )
            prefill_seconds = time.monotonic() - prefill_started
            decode_before = await _boundary_counters(args, prefill_before, concurrency)
            write_message(root, "release.json", {"run_id": run_id})
            await _wait_for(root, "released.json", run_id, tasks, args.barrier_timeout)
            released = time.monotonic()
            sampler = asyncio.create_task(
                asyncio.to_thread(
                    P.sample_gauges, args.base_url, args.episode_timeout, 0.5, stop
                )
            )
            complete = await _await_spans(
                progress, tasks, args.decode_tokens, stops, args.episode_timeout
            )
            decode_after = await asyncio.to_thread(P.counters, args.base_url)
            elapsed = time.monotonic() - released
            premature = sum(task.done() for task in tasks)
        finally:
            stop.set()
            await _cancel(tasks)
            if sampler is not None:
                gauges = await sampler
    return {
        "spans": [span_rate(p, args.decode_tokens, stops) for p in progress],
        "complete": complete,
        "premature": premature,
        "prefill_seconds": prefill_seconds,
        "start_tokens": P.delta(
            prefill_before, decode_before, "vllm:generation_tokens_total"
        ),
        "prefix_cache_hit_rate": P.prefix_cache_hit_rate(prefill_before, decode_before),
        "counters": {
            name: P.delta(decode_before, decode_after, name) for name in decode_after
        },
        # The server's own rate over the batch's life, post-EOS text included: a check on client timing.
        "server_rate": P.delta(
            decode_before, decode_after, "vllm:generation_tokens_total"
        )
        / elapsed,
        "gauges": gauges,
        "window": [released, released + elapsed],
    }


def _merge_gauges(samples: list[dict]) -> dict[str, float]:
    keys = {key for sample in samples for key in sample}
    merged = {}
    for key in keys:
        values = [sample[key] for sample in samples if key in sample]
        merged[key] = (
            min(values)
            if key.endswith("_min")
            else max(values)
            if key.endswith("_peak")
            else statistics.fmean(values)
        )
    return merged


def _decode_summary(args, concurrency: int, episodes: list[dict]) -> dict:
    """Throughput per batch is the sum of its requests' span rates (all ran in
    the full batch); per-user speed is their median over every answer."""
    spans = [
        span for episode in episodes for span in episode["spans"] if span is not None
    ]
    rates = [rate for _, rate, _ in spans]
    batches = []
    for episode in episodes:
        valid = [span[1] for span in episode["spans"] if span is not None]
        if valid:
            batches.append(sum(valid) * concurrency / len(valid))
    counters: dict[str, float] = {}
    for episode in episodes:
        for name, value in episode["counters"].items():
            counters[name] = counters.get(name, 0.0) + value
    per_user = P.percentile(rates, 0.5)
    stats = {
        "concurrency": concurrency,
        "measurement_kind": "decode_span",
        "decode_tokens": args.decode_tokens,
        "stop_token_ids": sorted(args.stop_token_ids),
        "prompts_screened": args.screened_rows is not None,
        "episodes": len(episodes),
        "sequences": len(spans),
        "output_tokens_per_second": statistics.fmean(batches) if batches else 0.0,
        "output_tokens_per_second_std": statistics.stdev(batches)
        if len(batches) > 1
        else 0.0,
        "server_output_tokens_per_second": statistics.fmean(
            episode["server_rate"] for episode in episodes
        ),
        "per_user_output_tokens_per_second": per_user,
        "per_user_p10": P.percentile(rates, 0.1),
        "per_user_p90": P.percentile(rates, 0.9),
        "median_tpot_ms": 1000.0 / per_user if per_user else 0.0,
        "span_tokens_median": P.percentile([tokens for tokens, _, _ in spans], 0.5),
        "natural_ends": sum(natural for _, _, natural in spans),
        "timed_out_episodes": sum(not episode["complete"] for episode in episodes),
        "premature_completions": sum(episode["premature"] for episode in episodes),
        "prefill_seconds": statistics.fmean(
            episode["prefill_seconds"] for episode in episodes
        ),
        "measurement_start_generation_tokens": statistics.fmean(
            episode["start_tokens"] for episode in episodes
        ),
        "prefix_cache_hit_rate": statistics.fmean(
            episode["prefix_cache_hit_rate"] for episode in episodes
        ),
        "decode_prefix_cache_queries": counters.get(
            "vllm:prefix_cache_queries_total", 0.0
        ),
        "preemptions": counters.get("vllm:num_preemptions_total", 0.0),
        "output_tokens": counters.get("vllm:generation_tokens_total", 0.0),
        "completed": 0,
        "failed": 0,
        "latency_p50_ms": 0.0,
        "latency_p99_ms": 0.0,
        "host_load_1m": os.getloadavg()[0],
        "foreign_gpu_processes": P.foreign_gpu_processes(
            getattr(args, "server_pgid", None)
        ),
        "windows": [episode["window"] for episode in episodes],
        **_merge_gauges([episode["gauges"] for episode in episodes]),
    }
    return stats | (P.acceptance({}, counters) if args.draft else {})


def decode_episodes(sequences: int, concurrency: int) -> int:
    """Batches a decode point runs: whole batches of `concurrency` until `sequences` answers."""
    return -(-sequences // concurrency)


async def _decode_point(args, prompts: list[dict], concurrency: int) -> dict:
    episodes = decode_episodes(args.decode_sequences, concurrency)
    rows = decode_rows(prompts, episodes * concurrency, args.screened_rows)
    url, root, stops = (
        request_url(args.base_url, rows[0]),
        Path(args.barrier_dir),
        frozenset(args.stop_token_ids),
    )
    # Disposable warmup on the first batch, so graphs and caches are warm.
    if args.warmup_seconds > 0:
        async with P.session(concurrency, args.request_timeout) as client:
            ids = [f"warmup-{secrets.token_hex(8)}-{i}" for i in range(concurrency)]
            warm, firsts = await _launch(args, client, url, rows[:concurrency], ids)
            await asyncio.wait_for(
                asyncio.gather(*(f.wait() for f in firsts)), args.warmup_timeout
            )
            await asyncio.sleep(args.warmup_seconds)
            await _cancel(warm)
    measured = [
        await _decode_episode(
            args, url, root, rows[e * concurrency : (e + 1) * concurrency], stops
        )
        for e in range(episodes)
    ]
    await P.await_idle(args.base_url, args.drain_seconds)
    return _decode_summary(args, concurrency, measured)


def decode_spans(args, prompts: list[dict], concurrency: int) -> dict:
    return asyncio.run(_decode_point(args, prompts, concurrency))


# --- sharegpt-rate --------------------------------------------------------------

DONE, CENSORED, FAILED = 0, 1, 2


def arrival_times(rate: float, count: int, seed: str) -> list[float]:
    """Offsets (s) of `count` Poisson arrivals at `rate` req/s (exponential gaps,
    drawn as gamma of shape 1), seeded so every arm gets the same schedule."""
    rng, now, times = random.Random(seed), 0.0, []
    for _ in range(count):
        now += rng.gammavariate(1.0, 1.0 / rate)
        times.append(now)
    return times


def latency_stats(
    records: list[list], slo_ttft_ms: float, slo_tpot_ms: float
) -> dict[str, float]:
    """Percentiles, normalized latency and SLO attainment of measured requests.

    A record is [arrival s, ttft ms, tpot ms, e2e ms or None, output tokens,
    status]. Censored requests enter TTFT/TPOT as lower bounds; censored and
    failed ones miss the SLO; only finished ones have end-to-end latency.
    """
    timed = [r for r in records if r[5] != FAILED]
    ttft, tpot = [r[1] for r in timed], [r[2] for r in timed if r[2]]
    done = [r for r in records if r[5] == DONE]
    e2e = [r[3] for r in done]
    attained = sum(r[1] <= slo_ttft_ms and r[2] <= slo_tpot_ms for r in done)
    stats = {f"ttft_p{p}_ms": P.percentile(ttft, p / 100) for p in (50, 90, 99)}
    stats |= {f"tpot_p{p}_ms": P.percentile(tpot, p / 100) for p in (50, 90, 99)}
    stats |= {f"e2e_p{p}_ms": P.percentile(e2e, p / 100) for p in (50, 90, 99)}
    per_token = [r[3] / r[4] for r in done if r[4]]
    return stats | {
        "median_ttft_ms": stats["ttft_p50_ms"],
        "median_tpot_ms": stats["tpot_p50_ms"],
        "normalized_latency_ms": sum(per_token) / len(per_token) if per_token else 0.0,
        "slo_attainment": attained / len(records) if records else 0.0,
    }


def _open_record(start: float, progress: P.Progress, outcome, now: float) -> list:
    """`outcome`: the Completion, None if the request failed, or ... if it was still running."""
    if isinstance(outcome, P.Completion):
        return [progress.started - start, outcome.ttft * 1000.0, outcome.tpot * 1000.0, outcome.elapsed * 1000.0,
                outcome.tokens, DONE]  # fmt: skip
    ttft = progress.ttft if progress.ttft is not None else now - progress.started
    tokens = progress.arrivals[-1][1] if progress.arrivals else 0
    return [progress.started - start, ttft * 1000.0, progress.tpot * 1000.0, None, tokens,
            FAILED if outcome is None else CENSORED]  # fmt: skip


class _Marks:
    """Server counters and gauges around an open-loop point's measured arrivals."""

    def __init__(self, base_url: str) -> None:
        self.base_url, self.stop = base_url, threading.Event()
        self.opened = self.closed = 0.0
        self.before = self.after = self.sampler = None

    def open(self) -> None:
        self.opened = time.monotonic()
        self.before = asyncio.create_task(asyncio.to_thread(P.counters, self.base_url))
        self.sampler = asyncio.create_task(
            asyncio.to_thread(P.sample_gauges, self.base_url, 1e9, 2.0, self.stop)
        )

    def close(self) -> None:
        if self.after is None:
            self.closed = time.monotonic()
            self.after = asyncio.create_task(
                asyncio.to_thread(P.counters, self.base_url)
            )
            self.stop.set()


async def _arrive(
    args,
    client,
    times: list[float],
    warmup_n: int,
    measured_n: int,
    warm,
    measured,
    marks: _Marks,
):
    """Send the schedule: warmup, then the measured arrivals, then load until those finish."""
    counted: list[P.Progress] = []
    results: dict[int, P.Completion | None] = {}
    inflight: set[asyncio.Task] = set()
    dropped = 0

    async def one(row: dict, index: int | None) -> None:
        progress = counted[index] if index is not None else None
        try:
            result = await P.stream(
                client,
                request_url(args.base_url, row),
                request_body(args, row),
                progress=progress,
            )
        except Exception:  # noqa: BLE001 - a failed request is a data point
            result = None
        if index is not None:
            results[index] = result

    start = time.monotonic()
    for i, offset in enumerate(times):
        await asyncio.sleep(max(0.0, start + offset - time.monotonic()))
        if i == warmup_n:
            marks.open()
        if i == warmup_n + measured_n:
            marks.close()
        if i >= warmup_n + measured_n and len(results) == measured_n:
            break  # every measured request has finished
        measuring = warmup_n <= i < warmup_n + measured_n
        if len(inflight) >= args.max_inflight:
            dropped += measuring
            continue
        if measuring:
            counted.append(P.Progress())
        task = asyncio.create_task(one(warm.next() if i < warmup_n else measured.next(),
                                       len(counted) - 1 if measuring else None))  # fmt: skip
        inflight.add(task)
        task.add_done_callback(inflight.discard)
    now = time.monotonic()
    for task in list(inflight):
        task.cancel()
    await asyncio.gather(*inflight, return_exceptions=True)
    marks.close()  # the schedule can run out inside the measured arrivals
    return counted, results, dropped, now


async def _serve_rate(args, rate: float, warm: P.Cycler, measured: P.Cycler) -> dict:
    warmup_n, measured_n = (
        round(rate * args.warmup_seconds),
        max(1, round(rate * args.measure_seconds)),
    )
    count = warmup_n + measured_n + round(rate * args.drain_seconds)
    times = arrival_times(rate, count, f"{args.arrival_seed}:{rate}")
    marks = _Marks(args.base_url)
    async with P.session(args.max_inflight, args.request_timeout) as client:
        counted, results, dropped, now = await _arrive(
            args, client, times, warmup_n, measured_n, warm, measured, marks
        )
    before, after, gauges = await marks.before, await marks.after, await marks.sampler
    await P.await_idle(args.base_url, 60.0)
    records = [
        _open_record(marks.opened, p, results.get(i, ...), now)
        for i, p in enumerate(counted)
    ]
    elapsed = marks.closed - marks.opened
    stats = {
        "rate": rate,
        "measurement_kind": "open_loop",
        "arrivals_measured": measured_n,
        "window_seconds": elapsed,
        "windows": [[marks.opened, marks.closed]],
        "completed": sum(r[5] == DONE for r in records),
        "censored": sum(r[5] == CENSORED for r in records),
        "failed": sum(r[5] == FAILED for r in records),
        "dropped": dropped,
        "output_tokens": P.delta(before, after, "vllm:generation_tokens_total"),
        "output_tokens_per_second": P.delta(
            before, after, "vllm:generation_tokens_total"
        )
        / elapsed,
        "slo_ttft_ms": args.slo_ttft_ms,
        "slo_tpot_ms": args.slo_tpot_ms,
        "preemptions": P.preemptions(before, after),
        "prefix_cache_hit_rate": P.prefix_cache_hit_rate(before, after),
        "host_load_1m": os.getloadavg()[0],
        "foreign_gpu_processes": P.foreign_gpu_processes(
            getattr(args, "server_pgid", None)
        ),
        **gauges,
        **latency_stats(records, args.slo_ttft_ms, args.slo_tpot_ms),
        "requests": records,
    }
    stats["goodput_rps"] = stats["slo_attainment"] * rate
    return stats | (P.acceptance(before, after) if args.draft else {})


def open_loop(args, prompts: list[dict], rate: float) -> dict:
    warm_rows, measured_rows = split_warmup(prompts)
    if args.prefix_caching == "on":
        P.post(
            f"{args.base_url}/reset_prefix_cache"
        )  # no free hits on what an earlier point sent
    return asyncio.run(
        _serve_rate(args, rate, P.Cycler(warm_rows), P.Cycler(measured_rows))
    )

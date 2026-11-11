# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The measurement protocol: server counters, streamed timing, steady windows.

Throughput is always the server's own generation counter across a window,
never a client-side sum, so tokens from requests in flight at either boundary
count exactly once. Acceptance comes from vLLM's spec-decode counters.
"""

import asyncio
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from array import array
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp

# A point is resident when the running batch reaches 90% of the request and
# stable when its mean stays within 5% of its peak.
RESIDENT_FRACTION = 0.9
STABLE_FRACTION = 0.95
KV_FULL_FRACTION = 0.9

GAUGES = {
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "kv_cache_usage": "vllm:kv_cache_usage_perc",
}


def get(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        return response.read().decode()


def post(url: str, timeout: float = 60.0) -> None:
    request = urllib.request.Request(url, data=b"", method="POST")  # noqa: S310
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        response.read()


def parse_metrics(text: str) -> dict[str, float]:
    """vLLM's Prometheus series, summed over label sets."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        labelled, _, value = line.partition(" ")
        name = labelled.partition("{")[0]
        try:
            out[name] = out.get(name, 0.0) + float(value)
        except ValueError:
            continue
    return out


def waiting_capacity(text: str) -> float | None:
    """Requests the scheduler declined to admit (memory or a cap)."""
    for line in text.splitlines():
        if (
            line.startswith("vllm:num_requests_waiting_by_reason{")
            and 'reason="capacity"' in line
        ):
            return float(line.rpartition(" ")[2])
    return None


def counters(base_url: str) -> dict[str, float]:
    return parse_metrics(get(f"{base_url}/metrics"))


def cache_config(base_url: str) -> dict[str, str]:
    """vLLM's cache config (block size, GPU blocks, ...), from its info gauge's labels."""
    try:
        text = get(f"{base_url}/metrics")
    except (urllib.error.URLError, OSError):
        return {}
    for line in text.splitlines():
        if line.startswith("vllm:cache_config_info{"):
            return dict(re.findall(r'(\w+)="([^"]*)"', line))
    return {}


def foreign_gpu_processes(own_group: int | None) -> int | None:
    """GPU compute processes on this host outside the server's process group:
    other arms or other users sharing the machine during a measurement."""
    if own_group is None:
        return None
    try:
        listing = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=10).stdout  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return None
    foreign = 0
    for pid in {int(word) for word in listing.split() if word.isdigit()}:
        try:
            foreign += os.getpgid(pid) != own_group
        except ProcessLookupError:
            continue
        except PermissionError:
            foreign += 1
    return foreign


def gpu_state(gpus: str) -> dict[str, float]:
    """SM clock, power and temperature of the server's GPUs now: throttling shows here."""
    try:
        listing = subprocess.run(["nvidia-smi", "-i", gpus, "--query-gpu=clocks.sm,power.draw,temperature.gpu",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout  # fmt: skip
        rows = [
            [float(v) for v in line.split(",")] for line in listing.strip().splitlines()
        ]
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return {}
    if not rows:
        return {}
    clocks, power, temperature = zip(*rows, strict=True)
    return {"gpu_sm_clock_mhz": sum(clocks) / len(clocks), "gpu_power_w": sum(power) / len(power),
            "gpu_temperature_c": max(temperature)}  # fmt: skip


async def loop_lag(stop: asyncio.Event, interval: float = 0.05) -> list[float]:
    """How late the client's event loop wakes a sleeper: its own saturation."""
    lags = []
    while not stop.is_set():
        start = time.monotonic()
        await asyncio.sleep(interval)
        lags.append(time.monotonic() - start - interval)
    return lags


def delta(before: dict, after: dict, name: str) -> float:
    return after.get(name, 0.0) - before.get(name, 0.0)


def acceptance(before: dict, after: dict) -> dict[str, float]:
    """Acceptance in the papers' units, which count the bonus token.

    vLLM counts accepted drafts only (tau in [0, k]); DSpark/DFlash report
    tau in [1, k+1]. `verify_len_observed` / `draft_len_observed` are the
    positions actually verified / drafted per step (adaptive verification and
    Fore-D trim both); they equal k when nothing trims.
    """

    def d(name: str) -> float:
        return delta(before, after, f"vllm:spec_decode_{name}_total")

    drafts = d("num_drafts")
    if not drafts:
        return {}
    k = d("num_draft_tokens") / drafts
    length = 1.0 + d("num_accepted_tokens") / drafts
    verified, drafted = d("num_verified_tokens"), d("num_drafted_tokens")
    return {
        "acceptance_length": length,
        "acceptance_rate": length / (1.0 + k) if k else 0.0,
        "accepted_per_step": length - 1.0,
        "draft_steps": drafts,
        "k_observed": k,
        "verify_len_observed": verified / drafts if verified else k,
        "draft_len_observed": drafted / drafts if drafted else k,
    }


def prefix_cache_hit_rate(before: dict, after: dict) -> float:
    queries = delta(before, after, "vllm:prefix_cache_queries_total")
    return (
        delta(before, after, "vllm:prefix_cache_hits_total") / queries
        if queries
        else 0.0
    )


def preemptions(before: dict, after: dict) -> float:
    return delta(before, after, "vllm:num_preemptions_total")


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def wait_healthy(base_url: str, proc, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}")
        try:
            get(f"{base_url}/health", timeout=5.0)
            return
        except (urllib.error.URLError, OSError):
            time.sleep(2.0)
    raise TimeoutError(f"server not healthy after {timeout:.0f}s")


def await_steady(
    base_url: str, target: int, minimum: float, timeout: float, patience: int = 6
) -> dict:
    """Hold until the resident batch stops growing (or reaches the target).

    A fixed warmup opens the window mid-ramp at wide/long points (256 x 8k
    measured 31% low); waiting for a plateau also gives capacity-bound points a
    clean window, and the warnings report the width actually reached.
    """
    time.sleep(minimum)
    started = time.monotonic()
    best, stalled = -1.0, 0
    while time.monotonic() - started < timeout:
        try:
            running = counters(base_url).get(GAUGES["running"])
        except (urllib.error.URLError, OSError):
            running = None
        if running is not None:
            if running >= target * RESIDENT_FRACTION:
                return {
                    "warmup_seconds": minimum + time.monotonic() - started,
                    "steady": True,
                }
            best, stalled = (running, 0) if running > best else (best, stalled + 1)
            if stalled >= patience:
                return {
                    "warmup_seconds": minimum + time.monotonic() - started,
                    "steady": True,
                }
        time.sleep(2.0)
    return {"warmup_seconds": minimum + time.monotonic() - started, "steady": False}


def sample_gauges(
    base_url: str,
    seconds: float,
    interval: float = 2.0,
    stop: threading.Event | None = None,
) -> dict[str, float]:
    """Poll the scheduler gauges over the window (or until `stop`); min/peak/mean
    of each, and the slowest interval's token rate against the window's mean."""
    samples: dict[str, list[float]] = {
        name: [] for name in (*GAUGES, "waiting_capacity")
    }
    generated: list[tuple[float, float]] = []
    deadline = time.monotonic() + seconds
    while (remaining := deadline - time.monotonic()) > 0 and not (
        stop and stop.is_set()
    ):
        try:
            text = get(f"{base_url}/metrics")
        except (urllib.error.URLError, OSError):
            text = ""
        snapshot = parse_metrics(text)
        if "vllm:generation_tokens_total" in snapshot:
            generated.append(
                (time.monotonic(), snapshot["vllm:generation_tokens_total"])
            )
        for name, metric in GAUGES.items():
            if metric in snapshot:
                samples[name].append(snapshot[metric])
        if (capacity := waiting_capacity(text)) is not None:
            samples["waiting_capacity"].append(capacity)
        if stop is not None:
            stop.wait(min(interval, remaining))
        else:
            time.sleep(min(interval, remaining))
    out = {}
    for name, values in samples.items():
        if values:
            out |= {
                f"{name}_min": min(values),
                f"{name}_peak": max(values),
                f"{name}_mean": sum(values) / len(values),
            }
    return out | rate_floor(generated)


def rate_floor(points: list[tuple[float, float]]) -> dict[str, float]:
    """The slowest sampling interval's token rate as a fraction of the mean: a stall shows near zero."""
    rates = [
        (n1 - n0) / (t1 - t0)
        for (t0, n0), (t1, n1) in zip(points, points[1:])
        if t1 > t0
    ]
    mean = (
        (points[-1][1] - points[0][1]) / (points[-1][0] - points[0][0])
        if len(rates) >= 3
        else 0.0
    )
    return {"token_rate_floor": min(rates) / mean} if mean > 0 else {}


# The file vLLM's step timer appends (async_utils.STEP_TIMING_FIELDS in
# patches/vllm.patch): float64 rows, `monotonic` the step's host start.
STEP_TIMING_FIELDS = (
    "monotonic", "num_reqs", "num_tokens", "num_drafted", "full_cudagraph",
    "prep_ms", "forward_ms", "sample_ms", "draft_ms", "post_ms", "idle_ms",
)  # fmt: skip
STEP_SEGMENTS = STEP_TIMING_FIELDS[5:]


def step_timing(path: Path, windows: list[list[float]]) -> dict[str, float]:
    """Mean per-step breakdown of the timed steps that began inside `windows`.

    Means, not medians, so the segments add up to the step period."""
    width = len(STEP_TIMING_FIELDS)
    data = path.read_bytes() if path.exists() else b""
    values = array("d", data[: len(data) // (8 * width) * (8 * width)])
    rows = [values[i : i + width] for i in range(0, len(values), width)]
    picked = [r for r in rows if any(a <= r[0] <= b for a, b in windows)]
    if not picked:
        return {"step_timed": 0}
    n = len(picked)
    columns = {"step_reqs": 1, "step_tokens": 2, "step_drafted": 3}
    columns["step_full_cudagraph"] = 4
    columns |= {f"step_{name}": 5 + i for i, name in enumerate(STEP_SEGMENTS)}
    stats = {"step_timed": n, "step_ms": sum(sum(r[5:]) for r in picked) / n}
    stats |= {key: sum(r[col] for r in picked) / n for key, col in columns.items()}
    return stats


async def await_idle(base_url: str, timeout: float = 60.0) -> bool:
    """Wait until nothing is running or waiting (three samples in a row)."""
    deadline, idle = time.monotonic() + timeout, 0
    while time.monotonic() < deadline:
        try:
            snap = await asyncio.to_thread(counters, base_url)
            idle = (
                idle + 1
                if snap.get(GAUGES["running"]) == 0 and snap.get(GAUGES["waiting"]) == 0
                else 0
            )
        except (urllib.error.URLError, OSError):
            idle = 0
        if idle >= 3:
            return True
        await asyncio.sleep(0.1)
    return False


@dataclass
class Completion:
    """One streamed response: ttft is the wait for the first token, tpot the
    mean gap after it (prefill and queueing stay out of per-token figures)."""

    tokens: int
    elapsed: float
    ttft: float
    tpot: float
    token_ids: list[int] | None = (
        None  # when the request asked for them (return_token_ids)
    )


@dataclass(eq=False)
class Progress:
    """A stream while it runs: its start and (arrival time, tokens so far) per chunk."""

    started: float = 0.0
    arrivals: list[tuple[float, int]] = field(default_factory=list)
    token_ids: list[int] = field(
        default_factory=list
    )  # filled when the request asks for them

    @property
    def ttft(self) -> float | None:
        return self.arrivals[0][0] - self.started if self.arrivals else None

    @property
    def tpot(self) -> float:
        """The mean gap after the first chunk, as Completion.tpot."""
        if len(self.arrivals) < 2:
            return 0.0
        (t0, n0), (t1, n1) = self.arrivals[0], self.arrivals[-1]
        return (t1 - t0) / (n1 - n0) if n1 > n0 else 0.0


def session(concurrency: int, timeout: float) -> aiohttp.ClientSession:
    return aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=concurrency, limit_per_host=concurrency),
        timeout=aiohttp.ClientTimeout(total=timeout),
    )


async def stream(
    client,
    url: str,
    body: dict,
    first_token: asyncio.Event | None = None,
    progress: Progress | None = None,
) -> Completion:
    """POST with SSE streaming and time the tokens as they arrive.

    On the event loop, not a thread per request: at 512 threads the client
    becomes the concurrency limit (as vLLM's own benchmark client notes).
    With `progress`, every chunk's arrival and running token count is kept
    as it streams (usage on every chunk, so the counts are exact).
    """
    options = {"include_usage": True} | (
        {"continuous_usage_stats": True} if progress is not None else {}
    )
    payload = {**body, "stream": True, "stream_options": options}
    started, ttft, tokens = time.monotonic(), None, 0
    track = progress if progress is not None else Progress()
    track.started = started
    token_ids: list[int] | None = (
        track.token_ids if body.get("return_token_ids") else None
    )
    try:
        async with client.post(url, json=payload) as response:
            response.raise_for_status()
            async for raw in response.content:
                line = raw.decode().strip()
                if (
                    not line.startswith("data:")
                    or (text := line[5:].strip()) == "[DONE]"
                ):
                    continue
                chunk = json.loads(text)
                if chunk.get("error") is not None:
                    raise RuntimeError(f"in-band error: {chunk['error']}")
                if usage := chunk.get("usage"):
                    tokens = int(usage.get("completion_tokens") or 0)
                for choice in chunk.get("choices") or []:
                    if token_ids is not None:
                        token_ids += choice.get("token_ids") or []
                    content = (choice.get("delta") or {}).get("content") or choice.get(
                        "text"
                    )
                    if ttft is None and (content or choice.get("token_ids")):
                        ttft = time.monotonic() - started
                        if first_token is not None:
                            first_token.set()
                seen = len(token_ids) if token_ids is not None else tokens
                if seen > (track.arrivals[-1][1] if track.arrivals else 0):
                    track.arrivals.append((time.monotonic(), seen))
    finally:
        if first_token is not None:
            first_token.set()
    elapsed = time.monotonic() - started
    first = ttft if ttft is not None else elapsed
    tpot = (elapsed - first) / (tokens - 1) if tokens > 1 else 0.0
    return Completion(tokens, elapsed, first, tpot, token_ids)


class Cycler:
    """The same prompt sequence for every arm, whatever its speed.

    Round-robin across categories, in order within each, wrapping at the end.
    """

    def __init__(self, prompts: list[dict]) -> None:
        buckets: dict[str, list[dict]] = {}
        for row in prompts:
            buckets.setdefault(row.get("category", "?"), []).append(row)
        self._buckets = [buckets[k] for k in sorted(buckets)]
        self._positions = [0] * len(self._buckets)
        self._next = 0
        self._lock = threading.Lock()
        self.wraps = 0

    def next(self) -> dict:
        with self._lock:
            index, self._next = self._next, (self._next + 1) % len(self._buckets)
            bucket, position = self._buckets[index], self._positions[index]
            self._positions[index] += 1
            if position and position % len(bucket) == 0:
                self.wraps += 1
            return bucket[position % len(bucket)]


class Window:
    """Per-request stats for requests dispatched inside the measurement window.

    Requests still running when the drain ends are censored there: their TTFT
    and TPOT cover what streamed under load, and nothing after (the server then
    empties, which would flatter them); one still waiting for its first token
    counts its wait so far. Only end-to-end latency needs a finish.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._open = False
        self.started = False  # set at open and kept: dispatches then come from the measured prompts
        self.dispatched = 0
        self.inflight: set[Progress] = set()
        self.completed = self.failed = 0
        self.latencies: list[float] = []
        self.ttfts: list[float] = []
        self.tpots: list[float] = []
        self.censored: list[
            tuple[float, float]
        ] = []  # (ttft, tpot) frozen at the drain deadline
        self.waiting_at_censor = 0
        self.failures: dict[str, int] = {}

    def open(self) -> None:
        with self._lock:
            self._open = self.started = True

    def close(self) -> None:
        with self._lock:
            self._open = False

    def dispatch(self) -> Progress | None:
        """A Progress to stream into when the request counts, else None."""
        with self._lock:
            if not self._open:
                return None
            self.dispatched += 1
            progress = Progress()
            self.inflight.add(progress)
            return progress

    def finish(
        self, progress: Progress | None, result: Completion | None, cause: str = ""
    ) -> None:
        with self._lock:
            if progress not in self.inflight:  # not counted, or censored already
                return
            self.inflight.discard(progress)
            if result is None:
                self.failed += 1
                self.failures[cause] = self.failures.get(cause, 0) + 1
                return
            self.completed += 1
            self.latencies.append(result.elapsed * 1000.0)
            self.ttfts.append(result.ttft * 1000.0)
            if result.tpot:
                self.tpots.append(result.tpot * 1000.0)

    def await_drain(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if not self.inflight:
                    return True
            time.sleep(0.5)
        return False

    def censor(self) -> None:
        now = time.monotonic()
        with self._lock:
            self.censored = [
                (p.ttft if p.ttft is not None else now - p.started, p.tpot)
                for p in self.inflight
            ]
            self.waiting_at_censor = sum(p.ttft is None for p in self.inflight)
            self.inflight.clear()

    def summary(self) -> dict:
        tpots = self.tpots + [tpot * 1000.0 for _, tpot in self.censored if tpot]
        ttfts = self.ttfts + [ttft * 1000.0 for ttft, _ in self.censored]
        return {
            "completed": self.completed,
            "failed": self.failed,
            "censored": len(self.censored),
            "censored_before_first_token": self.waiting_at_censor,
            "latency_p50_ms": percentile(self.latencies, 0.5),
            "latency_p90_ms": percentile(self.latencies, 0.9),
            "latency_p99_ms": percentile(self.latencies, 0.99),
            "median_tpot_ms": percentile(tpots, 0.5),
            "median_ttft_ms": percentile(ttfts, 0.5),
            "p99_ttft_ms": percentile(ttfts, 0.99),
            "failure_causes": dict(
                sorted(self.failures.items(), key=lambda kv: -kv[1])
            ),
        }


def decode_warnings(row: dict) -> list[str]:
    """A decode row's own caveats: its batch, admissions, and the spans it timed."""
    messages = []
    if row.get("running_min") is not None and row["running_min"] < row["concurrency"]:
        messages.append(
            f"the decode batch fell to {row['running_min']:.0f} of {row['concurrency']} in the window"
        )
    if row.get("decode_prefix_cache_queries"):
        messages.append("requests were admitted inside the decode window")
    sequences, natural = row.get("sequences") or 0, row.get("natural_ends") or 0
    if sequences and natural > 0.1 * sequences:
        messages.append(f"{natural} of {sequences} answers ended before {row['decode_tokens']} tokens "
                        f"(median span {row['span_tokens_median']:.0f})")  # fmt: skip
    if row.get("stop_token_ids") == []:
        messages.append(
            "no end-of-turn ids: spans may run past an answer's natural end"
        )
    if row.get("prompts_screened") is False:
        messages.append(
            "prompts not screened for answer length: run `eval.py decode-prompts MODEL`"
        )
    if row.get("timed_out_episodes"):
        messages.append(
            f"{row['timed_out_episodes']} episodes hit --episode-timeout before every span ended"
        )
    if row.get("premature_completions"):
        messages.append(
            f"{row['premature_completions']} requests ended before their span"
        )
    client, server = (
        row.get("output_tokens_per_second"),
        row.get("server_output_tokens_per_second"),
    )
    if (
        row.get("prompts_screened")
        and client
        and server
        and abs(client / server - 1) > 0.1
    ):
        messages.append(
            f"client-timed {client:.0f} tok/s against the server's {server:.0f}: check client lag"
        )
    return messages


def environment_warnings(row: dict) -> list[str]:
    """What else on the host may have shaped a point: stalls, other GPU work, a saturated client."""
    messages = []
    if (floor := row.get("token_rate_floor")) is not None and floor < 0.3:
        messages.append(
            f"the token rate fell to {floor:.0%} of its mean in one sampling interval (a stall)"
        )
    if row.get("foreign_gpu_processes"):
        messages.append(
            f"{row['foreign_gpu_processes']} other GPU processes ran on this host during the window"
        )
    if (row.get("client_lag_p99_ms") or 0) > 50:
        messages.append(
            f"the client's event loop lagged {row['client_lag_p99_ms']:.0f} ms (p99): it may pace the load"
        )
    return messages


def open_loop_warnings(row: dict) -> list[str]:
    """An open-loop point has no target batch: its caveats are the requests it could not measure cleanly."""
    messages = environment_warnings(row)
    if row.get("dropped"):
        messages.append(
            f"{row['dropped']} measured arrivals dropped at --max-inflight: the client could not hold the load"
        )
    if row.get("censored"):
        messages.append(
            f"{row['censored']} measured requests outlived the drain: SLO misses, TTFT/TPOT lower bounds"
        )
    failed, total = row.get("failed", 0), row.get("arrivals_measured", 0)
    if failed and total and failed / total > 0.01:
        messages.append(f"{failed} of {total} measured requests failed")
    if row.get("preemptions"):
        messages.append(f"{row['preemptions']:.0f} preemptions (includes recompute)")
    return messages


def row_warnings(row: dict) -> list[str]:
    """Reasons a point did not measure what its concurrency label claims."""
    if row.get("measurement_kind") == "open_loop":
        return open_loop_warnings(row)
    decode = row.get("measurement_kind") == "decode_span"
    messages = (decode_warnings(row) if decode else []) + environment_warnings(row)
    requested, peak, mean = (
        row["concurrency"],
        row.get("running_peak"),
        row.get("running_mean"),
    )
    if peak is not None and peak < requested * RESIDENT_FRACTION:
        kv = row.get("kv_cache_usage_peak")
        cause = (
            f"the KV cache is full ({kv:.0%})"
            if kv is not None and kv >= KV_FULL_FRACTION
            else "a cap binds (likely max_num_batched_tokens)"
            if row.get("waiting_capacity_mean")
            else "cause unclear"
        )
        messages.append(
            f"server ran at most {peak:.0f} sequences ({peak / requested:.0%}); {cause}"
        )
    if peak and mean is not None and mean < peak * STABLE_FRACTION:
        idle = (row.get("waiting_mean") or 0) < 1 and (
            row.get("kv_cache_usage_peak") or 0
        ) < KV_FULL_FRACTION
        messages.append(f"the engine averaged {mean:.0f} of {requested} requests with none waiting: "
                        "the client or API servers lag" if idle else
                        f"batch averaged {mean:.0f} against a peak of {peak:.0f}; raise --warmup-seconds")  # fmt: skip
    if row.get("steady") is False:
        messages.append("the batch never settled within --warmup-timeout")
    if row.get("preemptions"):
        messages.append(f"{row['preemptions']:.0f} preemptions (includes recompute)")
    failed, total = row.get("failed", 0), row.get("failed", 0) + row.get("completed", 0)
    if failed and total and failed / total > 0.01:
        messages.append(
            f"{failed} of {total} requests failed: {row.get('failure_causes')}"
        )
    if row.get("censored"):
        messages.append(f"{row['censored']} window requests outlived the drain: TTFT/TPOT count them to there, "
                        "end-to-end latency covers finished requests only")  # fmt: skip
    elif row.get("drained") is False:
        messages.append(
            "the window did not drain; latency percentiles cover finished requests only"
        )
    return messages

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Build the training corpus for one target, in three resumable stages.

    ./data.py qwen14                     # all stages
    ./data.py qwen14 --stage regenerate --gpus 4,5,6,7

  pool        Magpie + ultrachat prompts (benchmark prompts removed)    -> data/pool/train.jsonl
  regenerate  responses sampled from the target itself                  -> data/regenerated/<model>.jsonl
  prepare     tokenized for the trainer                                 -> data/prepared/<model>/

Only the prompts come from the datasets: each target regenerates the responses,
so its drafter learns that target's distribution. Duplicates are kept; sampled
responses make each one a fresh draw. The pool leaves out the benchmark's
prompts (`eval.py prepare`'s ShareGPT stream).
"""

import argparse
import contextlib
import hashlib
import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from ..bench.server import capture_sizes
from ..config import for_gpus, jit_env, MODELS, POOL_ENV, target_snapshot
from ..paths import BENCH_DATA, DATA, ROOT

# RedHatAI's drafter corpus: ~500k conversations, at the revisions the paper's pool streamed.
SOURCES = (
    ("Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered", "1a982eea9ece373700dd8dfd04a4de08c2578c24", "train",
     "conversations"),
    ("HuggingFaceH4/ultrachat_200k", "8049631c405ae6576f93f445c6b8166f76f5505a", "train_sft", "messages"),
)  # fmt: skip
ROLES = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}
FALLBACK_SAMPLING = {"temperature": 0.7}  # greedy regeneration is too narrow a slice
STARTUP_TIMEOUT = 3600  # dp loads the whole checkpoint per replica
# Regeneration context: a prompt of up to this many tokens plus a --seq-len
# response. A target's own maximum (1M for GLM-5.3) can need more KV cache for
# one request than a node has.
PROMPT_ROOM = 32768
LOOPBACK = "http://127.0.0.1"  # "localhost" resolves to ::1 first; vLLM binds IPv4


def prompt_key(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()


def loose_key(text: str) -> str:
    return hashlib.sha256(
        re.sub(r"[^a-z0-9]+", "", text.casefold()).encode()
    ).hexdigest()


def benchmark_keys() -> tuple[set[str], set[str]]:
    """Every benchmark prompt (`eval.py prepare`'s files): a corpus overlap
    inflates acceptance, and acceptance drives throughput."""
    exact, loose = set(), set()
    for path in sorted(BENCH_DATA.glob("*.jsonl")):
        for line in path.read_text(errors="replace").splitlines():
            if line.strip():
                row = json.loads(line)
                text = "\n".join(
                    m.get("content", "")
                    for m in row.get("messages", [])
                    if m.get("role") == "user"
                ).strip()
                if text:
                    exact.add(prompt_key(text))
                    loose.add(loose_key(text))
    return exact, loose


def source_rows(name: str, revision: str, split: str):
    """A source split's rows in order: its parquet shards at the pinned
    revision, each fetched into the Hugging Face cache when reached and read
    from disk. (Streaming them through `datasets` left a reader thread that
    aborted the interpreter at exit.)"""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, HfApi

    files = HfApi().list_repo_files(name, repo_type="dataset", revision=revision)
    shards = [f for f in files if Path(f).name.startswith(f"{split}-")]
    for file in sorted(f for f in shards if f.endswith(".parquet")):
        path = hf_hub_download(name, file, repo_type="dataset", revision=revision)
        for batch in pq.ParquetFile(path).iter_batches():
            yield from batch.to_pylist()


def build_pool(out: Path, limit: int | None, exclude: bool) -> None:
    exact, loose = benchmark_keys() if exclude else (set(), set())
    out.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    with (out / "train.jsonl").open("w") as handle:
        for name, revision, split, column in SOURCES:
            for row in source_rows(name, revision, split):
                turns = [{"role": ROLES.get(m.get("from") or m.get("role", "")), "content": m.get("value") or m.get("content")}
                         for m in row.get(column) or []]  # fmt: skip
                if (
                    not turns
                    or any(t["role"] is None or t["content"] is None for t in turns)
                    or turns[0]["role"] != "user"
                ):
                    continue
                prompts = "\n".join(t["content"] for t in turns if t["role"] == "user")
                key = prompt_key(prompts)
                if key in exact or loose_key(prompts) in loose:
                    dropped += 1
                    continue
                handle.write(
                    json.dumps(
                        {
                            "id": f"{key}-{kept}",
                            "primary_id": key,
                            "conversations": turns,
                        }
                    )
                    + "\n"
                )
                kept += 1
                if limit and kept >= limit:
                    break
            if limit and kept >= limit:
                break
    print(f"pool: {kept:,} rows, {dropped:,} benchmark overlaps dropped -> {out}")


@contextlib.contextmanager
def target_server(args, model: str):
    """A plain vLLM server for regeneration on the target's pinned snapshot
    (`model`, config.target_snapshot): dp replicas on --gpus, torn down after."""
    cfg = for_gpus(MODELS[args.model], args.gpus)
    parallel = args.parallel or cfg.get("parallel", "dp")
    ep = (
        ["--enable-expert-parallel"]
        if cfg.get("expert_parallel") and parallel == "tp"
        else []
    )
    cmd = [
        str(ROOT / ".venv/bin/vllm"), "serve", model, *[str(a) for a in cfg.get("vllm_args", [])],
        "--port", str(args.port), "--dtype", "bfloat16",
        "--data-parallel-size" if parallel == "dp" else "--tensor-parallel-size", str(len(args.gpus.split(","))),
        *ep, "--trust-remote-code",
        *(["--max-num-seqs", str(n)] if (n := args.max_num_seqs or cfg.get("max_num_seqs")) else []),
    ]  # fmt: skip
    cmd += ["--max-model-len", str(args.seq_len + PROMPT_ROOM)]
    width = args.max_num_seqs or cfg.get("max_num_seqs")
    explicit = any(
        str(a).startswith("--compilation-config") for a in cfg.get("vllm_args", [])
    )
    if parallel == "tp" and "," in args.gpus and width and not explicit:
        # Wide graphs, as eval.py serves under TP: a step past vLLM's 512-token
        # sizes (decode plus new prompts' prefill) runs eagerly, dear at TP8.
        sizes = capture_sizes(cfg, width, width * (cfg["k"] + 1))
        graphs = json.dumps({"cudagraph_capture_sizes": sizes}, separators=(",", ":"))
        cmd += ["--compilation-config", graphs]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus, "VLLM_ENGINE_READY_TIMEOUT_S": str(STARTUP_TIMEOUT),
           **jit_env(cfg)}  # fmt: skip
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)

    def relay(
        signum, _frame
    ):  # the default SIGTERM skips `finally` and orphans the server
        raise KeyboardInterrupt(f"signal {signum}")

    previous = {s: signal.signal(s, relay) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while True:
            if proc.poll() is not None:
                raise SystemExit(f"server exited with code {proc.returncode}")
            if time.monotonic() > deadline:
                raise SystemExit("server did not come up in time")
            with contextlib.suppress(urllib.error.URLError, OSError):
                urllib.request.urlopen(f"{LOOPBACK}:{args.port}/health", timeout=5)  # noqa: S310
                break
            time.sleep(5)
        # A healthy port is not proof it is our server.
        with urllib.request.urlopen(
            f"{LOOPBACK}:{args.port}/v1/models", timeout=30
        ) as response:  # noqa: S310
            served = [m["id"] for m in json.loads(response.read())["data"]]
        if model not in served:
            raise SystemExit(
                f"port {args.port} serves {served}, not {model}; pick another --port"
            )
        yield f"{LOOPBACK}:{args.port}/v1/chat/completions", proc
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        stop(proc, 120)


def stop(proc: subprocess.Popen, wait: float) -> None:
    if proc.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=wait)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)


def chat_url(endpoint: str) -> str:
    """The regeneration client posts to exactly this URL; accept a base URL too."""
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/chat/completions"):
        return endpoint
    return (
        f"{endpoint}/chat/completions"
        if endpoint.endswith("/v1")
        else f"{endpoint}/v1/chat/completions"
    )


def regenerate(
    args,
    model: str,
    pool: Path,
    out: Path,
    endpoint: str,
    server: subprocess.Popen | None,
) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    sampling = MODELS[args.model].get("sampling", FALLBACK_SAMPLING)
    cmd = [
        str(ROOT / ".venv/bin/python"), str(ROOT / ".venv/src/speculators/scripts/response_regeneration/script.py"),
        "--dataset", "pool", "--outfile", str(out), "--endpoint", chat_url(endpoint),
        "--model", model, "--sampling-params", json.dumps(sampling),
        "--max-tokens", str(args.seq_len),
        # The client's default of 64 leaves a 512-slot server eight-way under-fed.
        "--concurrency", str(args.concurrency or 512 * len(args.gpus.split(","))),
        "--resume",
    ]  # fmt: skip
    print(f"regenerate: {args.model} ({sampling}) via {endpoint}")
    client = subprocess.Popen(
        cmd, env={**os.environ, POOL_ENV: str(pool)}, start_new_session=True
    )
    try:
        while True:
            try:
                return client.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # The client never notices its server dying (or being replaced).
                if server is not None and server.poll() is not None:
                    raise SystemExit(
                        f"the vLLM server exited ({server.returncode}); rerun to resume"
                    ) from None
    finally:
        stop(client, 60)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("model", choices=sorted(MODELS))
    p.add_argument(
        "--stage",
        choices=["pool", "regenerate", "prepare"],
        help="one stage instead of all three",
    )
    p.add_argument("--limit", type=int, help="cap the pool (smoke tests)")
    p.add_argument(
        "--no-exclude-benchmark",
        action="store_true",
        help="smoke tests only: keeps benchmark prompts",
    )
    p.add_argument(
        "--gpus", default="0,1,2,3", help="devices for the regeneration server"
    )
    p.add_argument("--port", type=int, help="default: the target's port")
    p.add_argument("--parallel", choices=["dp", "tp"])
    p.add_argument("--max-num-seqs", type=int)
    p.add_argument(
        "--concurrency",
        type=int,
        help="in-flight requests in total (default 512 per GPU)",
    )
    p.add_argument(
        "--endpoint",
        help="reuse a running plain vLLM server instead of starting one; it must serve the target's pinned "
        "snapshot (config.target_snapshot), the model the client names",
    )
    p.add_argument("--seq-len", type=int, default=8192)
    args = p.parse_args(argv)
    args.port = args.port or MODELS[args.model]["port"]

    pool, regenerated, prepared = (
        DATA / "pool",
        DATA / "regenerated" / f"{args.model}.jsonl",
        DATA / "prepared" / args.model,
    )
    for stage in [args.stage] if args.stage else ["pool", "regenerate", "prepare"]:
        if stage == "pool":
            if (pool / "train.jsonl").exists() and not args.limit:
                print(f"pool: reusing {pool}")
                continue
            if not args.no_exclude_benchmark and not any(BENCH_DATA.glob("*.jsonl")):
                raise SystemExit(
                    f"no benchmark prompts at {BENCH_DATA}: run './eval.py prepare' first"
                )
            build_pool(pool, args.limit, not args.no_exclude_benchmark)
        elif stage == "regenerate":
            model = target_snapshot(MODELS[args.model])
            server = (
                contextlib.nullcontext((chat_url(args.endpoint), None))
                if args.endpoint
                else target_server(args, model)
            )
            with server as (endpoint, proc):
                if code := regenerate(args, model, pool, regenerated, endpoint, proc):
                    return code
        else:
            info = prepared / "dataset_info.json"  # save_to_disk writes it last
            if (
                info.is_file()
                and regenerated.is_file()
                and info.stat().st_mtime > regenerated.stat().st_mtime
            ):
                print(f"prepare: reusing {prepared}, newer than {regenerated}")
                continue
            cmd = [str(ROOT / ".venv/bin/python"), str(ROOT / ".venv/src/speculators/scripts/prepare_data.py"),
                   "--model", target_snapshot(MODELS[args.model]), "--data", str(regenerated),
                   "--seq-length", str(args.seq_len), "--output", str(prepared)]  # fmt: skip
            if prepared.exists():
                # Unfinished, or older than the corpus: prepare_data.py would keep it.
                cmd.append("--overwrite")
            print(f"prepare: {args.model} -> {prepared}")
            if code := subprocess.call(cmd):
                return code
    return 0

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Launch and stop the vLLM server for one arm.

Every arm of a comparison gets the same caps (context, batch width, token
budget), so the drafter is the only thing that differs.
"""

import json
import os
import shlex
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from ..config import (
    CPU_RANKING_ENV,
    DECODE_BARRIER_ENV,
    ForeDConfig,
    GC_THRESHOLD_ENV,
    jit_env,
    METHODS,
    revision_cached,
    STEP_TIMING_FILE_ENV,
    STS_COLLECT_ENV,
    STS_ENV,
    TP_CONSISTENT_VERIFY_ENV,
)  # fmt: skip
from ..paths import ROOT

MAX_MODEL_LEN = 40960
VLLM_DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048
SHUTDOWN_GRACE_SECONDS = 30
GRAPH_STRIDE = 64  # past vLLM's sizes, as Fore-D's drafter graphs (extend_sizes)
# Tokens per streamed chunk after the first (sent at once). At one per token the
# single-process client's event loop lagged 45 ms (p99) at TP8 C=512 against
# 2 ms at eight; only the API servers read it, never the engine.
STREAM_INTERVAL = 8
# Python's collector at 50000/10/100 in every server process: with the defaults
# the engine core ran a ~200 ms full collection every few seconds at TP8 C=512,
# idling the GPU behind each one (every speculative arm, ~4% of throughput).
GC_THRESHOLD = "50000,10,100"


def fore_d(args) -> bool:
    """Whether this server runs Fore-D: its method does, except in
    the acceptance phase, which serves the drafter at fixed k (STS collection)."""
    return bool(
        args.draft and METHODS[args.method].fore_d and args.command != "acceptance"
    )


def parallel_label(args) -> str:
    return f"tp{len(args.gpus.split(','))}"


def batch_width(args, cfg: dict) -> int:
    """--max-num-seqs, else the target's value, else the widest point.

    A sweep point above it would silently measure the narrower batch, so that
    is refused.
    """
    widest = max(args.concurrency)
    width = args.max_num_seqs or cfg.get("max_num_seqs") or widest
    if width < widest:
        raise SystemExit(
            f"max_num_seqs {width} is below the widest concurrency {widest}"
        )
    return width


def token_budget(args, cfg: dict, width: int) -> int:
    """A speculative decode step needs width*(k+1) tokens; 3x leaves prefill room.

    The same for every arm, baseline included, so arms also interleave prefill
    the same way.
    """
    return args.max_num_batched_tokens or max(
        VLLM_DEFAULT_MAX_NUM_BATCHED_TOKENS, 3 * width * (cfg["k"] + 1)
    )


def capture_sizes(cfg: dict, width: int, budget: int) -> list[int]:
    """vLLM's sizes to 512, then every 64 tokens and every decode batch to width*(k+1).

    Every decode batch's full verify (k+1 tokens per request) has its graph:
    adaptive verification's varlen decode graphs are keyed by tokens alone, so
    without them a lone request verifying all of its drafts replays a larger
    graph (+0.5 ms at C = 1 on GLM-5.3-Flash, whose MoE pays for every row)."""
    query = cfg["k"] + 1
    widest = min(max(width * query, 512), budget)
    requests = [1, 2, 4, *range(8, 256, 8), *range(256, width + 1, 16), width]
    sizes = {1, 2, 4, *range(8, 256, 8), *range(256, 513, 16)}
    sizes |= {*range(512 + GRAPH_STRIDE, widest + 1, GRAPH_STRIDE)}
    sizes |= {n * query for n in requests}
    return sorted(s for s in sizes if s <= widest)


def graph_capture(args, cfg: dict) -> str:
    """eager, explicit (target vllm_args), wide (capture_sizes, under TP > 1) or vllm (to 512)."""
    if args.enforce_eager:
        return "eager"
    if any(str(a).startswith("--compilation-config") for a in cfg.get("vllm_args", ())):
        return "explicit"
    # Wide graphs cost KV cache: cheap split over TP ranks, dear on one GPU.
    return "wide" if "," in args.gpus else "vllm"


def draft_sample_method(args) -> str:
    """Greedy proposals at temperature 0, probabilistic (rejection) above it."""
    return "greedy" if args.temperature == 0 else "probabilistic"


def speculative_config(args, cfg: dict) -> dict:
    spec = {
        "method": METHODS[args.method].serve_method,
        "model": str(Path(args.draft).resolve())
        if Path(args.draft).exists()
        else args.draft,
        # Fore-D drafts up to k (--drafter-width forces narrower widths inside it).
        "num_speculative_tokens": cfg["k"],
        "draft_sample_method": draft_sample_method(args),
        # vLLM drops a prefix hit's last block for EAGLE, whose draft KV at a
        # position also reads the next token. These drafters' context KV reads
        # only the target's hidden state there, so the block is clean; the drop
        # also cost GLM-5.3-Flash's speculative arms every ShareGPT prefix hit.
        "disable_eagle_block_drop": True,
    }
    if args.draft_attention_backend:
        spec["attention_backend"] = args.draft_attention_backend
    # Fore-D always runs its verification quota through the adaptive manager.
    if args.adaptive_verification or fore_d(args):
        spec["enable_adaptive_verification"] = True
    return spec


def server_env(args, cfg: dict, max_model_len: int) -> dict[str, str]:
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": args.gpus,
        "VLLM_ENGINE_READY_TIMEOUT_S": str(int(args.startup_timeout)),
        GC_THRESHOLD_ENV: GC_THRESHOLD,
        **jit_env(cfg),
    }
    for name in (STS_ENV, STS_COLLECT_ENV, CPU_RANKING_ENV, TP_CONSISTENT_VERIFY_ENV, DECODE_BARRIER_ENV,
                 STEP_TIMING_FILE_ENV, *ForeDConfig.ENV.values()):  # fmt: skip
        env.pop(name, None)
    if "HF_HUB_OFFLINE" not in os.environ:
        # vLLM lists the model's files on the Hub at every start, even when
        # cached: a Hub outage killed servers.
        if revision_cached(cfg["repo"], cfg.get("revision")):
            env["HF_HUB_OFFLINE"] = "1"
    if "," in args.gpus and "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR" not in os.environ:
        # On a restart with the same config only rank 0 finds FlashInfer's tuned
        # tactics: the other ranks tune and wait in its all-reduce forever.
        # Every multi-GPU server tunes afresh.
        env["VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"] = tempfile.mkdtemp(
            prefix="forespark-autotune-"
        )
    if getattr(args, "step_timing_file", None):
        env[STEP_TIMING_FILE_ENV] = str(Path(args.step_timing_file).resolve())
    if getattr(args, "sts_collect", None):
        env[STS_COLLECT_ENV] = str(Path(args.sts_collect).resolve() / "sts")
    if args.sts:
        env[STS_ENV] = str(Path(args.sts).resolve())
    if args.cpu_ranking:
        env[CPU_RANKING_ENV] = "1"
    if getattr(args, "barrier_dir", None):
        env[DECODE_BARRIER_ENV] = str(args.barrier_dir)
        env["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"  # engine ids == client ids
    if args.prefix_caching == "on":
        env["VLLM_SERVER_DEV_MODE"] = "1"  # enables /reset_prefix_cache
    if fore_d(args):
        env |= ForeDConfig(
            enabled=True,
            fore_d_calibration=Path(args.fore_d_calibration).resolve()
            if args.fore_d_calibration
            else None,
            costs=Path(args.fore_d_costs).resolve() if args.fore_d_costs else None,
            fixed_width=args.drafter_width,
            log=Path(args.fore_d_log).resolve() if args.fore_d_log else None,
        ).to_env()
    if args.draft and "," in args.gpus:
        # Every rank must verify the same drafts with the same budget.
        env[TP_CONSISTENT_VERIFY_ENV] = "1"
    if args.adaptive_verification or fore_d(args):
        env["VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN"] = str(
            profile_context(args, max_model_len)
        )
    return env


def profile_context(args, max_model_len: int) -> int:
    """Context length of the startup dummy steps that price adaptive verification.

    Upstream profiles N target tokens as N one-token requests, so every token
    is priced as a whole new request reading this much KV, and a long context
    makes extra draft positions look far costlier than they are. ShareGPT uses
    1k: at 8k, adaptive verification on qwen14 verified 3.1 instead of 4.9
    drafts per step at C8 and lost 16% throughput.
    """
    if args.command == "acceptance":
        return min(max_model_len, 8192)
    return 1024


def serve(
    args, cfg: dict, max_model_len: int, width: int, budget: int
) -> subprocess.Popen:
    cmd = [
        str(ROOT / ".venv/bin/vllm"), "serve", cfg["repo"],
        *[str(a) for a in cfg.get("vllm_args", [])],
        *(["--revision", cfg["revision"], "--tokenizer-revision", cfg["revision"]] if cfg.get("revision") else []),
        "--served-model-name", args.model,
        "--port", str(args.port),
        "--dtype", "bfloat16",
        "--max-model-len", str(max_model_len),
        "--tensor-parallel-size", str(len(args.gpus.split(","))),
        *(["--enable-expert-parallel"] if cfg.get("expert_parallel") else []),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-num-seqs", str(width),
        "--max-num-batched-tokens", str(budget),
        "--enable-prefix-caching" if args.prefix_caching == "on" else "--no-enable-prefix-caching",
        "--stream-interval", str(STREAM_INTERVAL),
        "--trust-remote-code",
        "--shutdown-timeout", str(SHUTDOWN_GRACE_SECONDS),
    ]  # fmt: skip
    spec = speculative_config(args, cfg) if args.draft else {}
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    elif graph_capture(args, cfg) == "wide":
        sizes = {"cudagraph_capture_sizes": capture_sizes(cfg, width, budget)}
        cmd += ["--compilation-config", json.dumps(sizes, separators=(",", ":"))]
    if args.api_server_count > 1:
        cmd += ["--api-server-count", str(args.api_server_count)]
    if spec:
        cmd += ["--speculative-config", json.dumps(spec, separators=(",", ":"))]
    print("$ " + shlex.join(cmd), flush=True)
    return subprocess.Popen(
        cmd, env=server_env(args, cfg, max_model_len), start_new_session=True
    )


def stop_server(proc: subprocess.Popen) -> None:
    """SIGTERM the process group, escalate to SIGKILL, wait for it to vanish."""
    for sig, wait in (
        (signal.SIGTERM, SHUTDOWN_GRACE_SECONDS * 2),
        (signal.SIGKILL, 60),
    ):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            proc.poll()  # reap the leader: an unreaped zombie keeps the group alive
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.5)

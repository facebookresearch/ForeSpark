# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Serve target hidden states for online drafter training (speculators' launch_vllm.py).

    ./serve.py qwen14 --gpus 0,1            # port = target port + 100 (train.py's default)

Runs in the foreground for the whole training run. Each request writes a
~250-420 MB file under the hidden-states path (configs/paths.yaml). It serves
the target's pinned snapshot, as the trainer reads it (config.target_snapshot).
"""

import argparse
import os
import shlex
import signal
import subprocess
from pathlib import Path

from ..config import for_gpus, jit_env, MODELS, target_snapshot
from ..paths import HIDDEN_STATES, ROOT
from .data import stop


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("model", choices=sorted(MODELS))
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--port", type=int, help="default: target port + 100")
    # Must match train.py --seq-len: longer requests are rejected, and the
    # trainer turns a rejection into a dropped sample, not an error.
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument(
        "--parallel",
        choices=["dp", "tp"],
        help="dp needs the whole target on one GPU; default: per target",
    )
    p.add_argument(
        "--max-num-seqs", type=int, help="default: per target (vLLM's own if unset)"
    )
    # Prefill plus one token and bound on writing hidden states: graphs buy
    # little, and capture is where oversized batches fail.
    p.add_argument(
        "--enforce-eager", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        help="leave room for a trainer on the same GPUs",
    )
    p.add_argument("--hidden-states-path", default=str(HIDDEN_STATES))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    cfg = for_gpus(MODELS[args.model], args.gpus)
    port = args.port or cfg["port"] + 100
    parallel = args.parallel or cfg.get("parallel", "dp")
    ep = (
        ["--enable-expert-parallel"]
        if cfg.get("expert_parallel") and parallel == "tp"
        else []
    )
    Path(args.hidden_states_path).mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ROOT / ".venv/bin/python"), "-m", "forespark.train.entry",  # it reads the target's config too
        str(ROOT / ".venv/src/speculators/scripts/launch_vllm.py"), target_snapshot(cfg),
        *[str(a) for a in cfg.get("vllm_args", [])],
        "--hidden-states-backend", "file", "--hidden-states-path", args.hidden_states_path,
        "--target-layer-ids", *map(str, cfg["layers"]),
        "--",
        "--data-parallel-size" if parallel == "dp" else "--tensor-parallel-size", str(len(args.gpus.split(","))),
        *ep,
        "--port", str(port),
        # One token for the sample on top of the prompt, one of headroom.
        "--max-model-len", str(args.seq_len + 2),
        *(["--enforce-eager"] if args.enforce_eager else []),
        *(["--gpu-memory-utilization", str(args.gpu_memory_utilization)] if args.gpu_memory_utilization else []),
    ]  # fmt: skip
    if max_num_seqs := args.max_num_seqs or cfg.get("max_num_seqs"):
        cmd += ["--max-num-seqs", str(max_num_seqs)]
    print(
        f"{args.model} on GPUs {args.gpus} ({parallel}) port {port}, hidden states -> {args.hidden_states_path}"
    )
    if args.dry_run:
        print(shlex.join(cmd))
        return 0
    # dp on a large target cannot load within vLLM's default 600 s engine wait.
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": args.gpus,
        "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
        **jit_env(cfg),
    }
    # Own process group, torn down on SIGTERM/SIGINT: vLLM's engine processes
    # outlive a launcher killed alone, holding the GPUs.
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)

    def forward(signum, _frame):
        stop(proc, 120)
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, forward)
    return proc.wait()

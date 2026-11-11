# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Build and run the speculators training command for one (target, method).

Everything the two drafters must share -- block size from k, loss blend,
optimizer, schedule, data split -- is derived here, so they cannot drift.
Unknown flags pass through to speculators' train.py and win.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from ..config import METHODS, model_config, MODELS, target_snapshot
from ..paths import DATA, ROOT, RUNS

# RedHatAI's published recipes: TV-dominant blend (acceptance under sampling is
# 1 - d_TV), exponential position decay, AdamW without weight decay.
LOSS_FN = '{"ce": 0.1, "tv": 0.9}'
# RedHatAI's cards: DFlash2 trains at twice DSpark's rate.
LR = {"dspark": "3e-4", "forespark": "6e-4"}
CONFIDENCE = {"--enable-confidence-head": None, "--confidence-head-alpha": "1.0"}
# (flag, value) pairs; value None is a bare flag. Some are speculators defaults,
# stated so the logged command records the architecture. Both drafters attend
# bidirectionally inside the draft block: DFlash2's default and the original
# DSpark release's (speculators defaults DSpark to causal).
ARCH = {
    "dspark": {
        "--markov-rank": "256",
        "--markov-head-type": "vanilla",
        **CONFIDENCE,
        "--confidence-head-with-markov": None,
        "--sliding-window-non-causal": None,
    },
    # DFlash2 + confidence head (DSpark eq. 12) + a width-conditioned Fore-D
    # head: one row per width; each step also drafts at one narrower width for
    # that row's labels, from every 4th anchor of the step.
    "forespark": {
        "--conv-kernel-size": "2",
        "--conv-group-size": "16",
        "--selector-rank": "256",
        "--selector-top-k": "16",
        **CONFIDENCE,
        "--enable-fore-d-predictor": None,
        "--fore-d-rank": "256",
        "--fore-d-loss-alpha": "1.0",
        "--fore-d-narrow-stride": "4",
        "--sliding-window-non-causal": None,
    },
}
ATTENTION_FLAGS = {"--sliding-window-non-causal", "--no-sliding-window-non-causal"}


def gamma_for(block: int) -> int:
    """DFlash paper: loss-decay gamma 7 at block 16, 5 at 10, 4 at 8."""
    return 7 if block >= 14 else 5 if block >= 10 else 4


def run_name(model: str, method: str, k: int | None = None) -> str:
    return f"{model}-{method}-k{k or MODELS[model]['k']}"


def check_resume(checkpoints: Path, method: str, block: int) -> None:
    """Refuse to resume a checkpoint trained at another block size or without
    the method's attention/head contract."""
    if not checkpoints.is_dir():
        return
    saved = sorted(
        (int(c.name), c / "config.json")
        for c in checkpoints.iterdir()
        if c.name.isdigit()
    )
    if not saved or not saved[-1][1].is_file():
        return
    config = json.loads(saved[-1][1].read_text())
    if config.get("block_size", block) != block:
        raise SystemExit(
            f"cannot resume {saved[-1][1]}: block size {config['block_size']}, this run trains {block}"
        )
    if method != "forespark":
        return
    actual = {
        "sliding_window_non_causal": config.get("sliding_window_non_causal", True),
        "enable_fore_d_predictor": config.get("enable_fore_d_predictor", False),
    }
    if not all(actual.values()):
        raise SystemExit(
            f"cannot resume {saved[-1][1]}: saved {actual}, forespark needs both"
        )


def command(args, passthrough: list[str]) -> tuple[list[str], Path]:
    cfg, method = model_config(args.model, args.k), METHODS[args.method]
    block = method.block_size(cfg["k"])
    out = args.out or RUNS / run_name(args.model, args.method, cfg["k"])
    endpoint = (
        args.endpoint or f"http://127.0.0.1:{cfg['port'] + 100}/v1"
    )  # hidden_server.py's port
    trainer = str(ROOT / ".venv/src/speculators/scripts/train.py")
    draft = []
    if overrides := cfg.get("draft_overrides"):
        # MoE / MLA verifiers misreport the dense decoder shape the trainer
        # derives; pin it (and drop --num-layers, which the trainer then refuses).
        path = out / "draft_config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(overrides, indent=2) + "\n")
        draft = ["--draft-config", str(path)]
    overridden = {part.split("=", 1)[0] for part in passthrough}
    if overridden & ATTENTION_FLAGS:
        raise SystemExit(
            f"{args.method} fixes the drafter's attention; drop {sorted(overridden & ATTENTION_FLAGS)}"
        )
    flags = {
        "--per-position-loss-weight": "fixed-exp-decay",
        "--loss-fn": LOSS_FN,
        "--lr": LR[args.method],
        **ARCH[args.method],
    }
    cmd = [
        str(ROOT / ".venv/bin/torchrun"), "--standalone", f"--nproc_per_node={len(args.gpus.split(','))}",
        # The entry registers what the trainer lacks (dflash2_conf, GLM configs).
        "-m", "forespark.train.entry", trainer,
        "--speculator-type", method.speculator_type,
        *([] if draft else ["--num-layers", "5"]),
        "--block-size", str(block),
        "--dflash-decay-gamma", str(gamma_for(block)),
        "--max-anchors", str(cfg["max_anchors"]),
        "--verifier-name-or-path", target_snapshot(cfg),  # hidden_server.py serves the same directory
        "--data-path", str(DATA / "prepared" / args.model),
        "--vllm-endpoint", endpoint,
        "--save-path", str(out / "checkpoints"),
        "--log-dir", str(out / "logs"),
        "--logger", args.logger,
        "--target-layer-ids", *map(str, cfg["layers"]),
        "--total-seq-len", str(args.seq_len),
        "--epochs", str(args.epochs or cfg["epochs"]),
        "--optimizer", "adamw", "--weight-decay", "0.0",
        "--scheduler-type", "cosine", "--scheduler-warmup-ratio", "0.04",
        "--seed", "42",
        # A positional tail slice: the 5% validation split.
        "--train-data-ratio", "0.95",
        "--on-missing", "generate", "--on-generate", "delete",
        # In-flight hidden-state requests = workers x prefetch x ranks; more
        # than this timed out on the server's write lock.
        "--num-workers", str(args.num_workers), "--prefetch-factor", str(args.prefetch_factor),
        "--checkpoint-freq", str(args.checkpoint_freq),
        *(["--dry-run"] if args.init_only else []),
        *(["--fsdp-shard"] if args.fsdp_shard else []),
        *(["--no-resume-from-checkpoint"] if args.no_resume else []),
        *draft,
        *[p for flag, value in flags.items() if flag not in overridden for p in (flag, *([value] if value else []))],
        *passthrough,
    ]  # fmt: skip
    return cmd, out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a drafter against a running hidden-state server (serve.py)."
    )
    p.add_argument("model", choices=sorted(MODELS))
    p.add_argument("method", choices=sorted(METHODS))
    p.add_argument("--k", type=int, help="draft length (default: the target's); sets the block size, the loss "
                   "decay and the run name (-k15) together")  # fmt: skip
    p.add_argument("--gpus", default="4,5,6,7")
    p.add_argument(
        "--endpoint",
        help="hidden-state server (default: serve.py's port for the target)",
    )
    p.add_argument("--epochs", type=int, help="default: the target's")
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--out", type=Path)
    p.add_argument(
        "--no-resume", action="store_true", help="start fresh even if checkpoints exist"
    )
    p.add_argument(
        "--fsdp-shard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="shard drafter params/grads/optimizer across trainer ranks (a memory decision)",
    )
    p.add_argument(
        "--checkpoint-freq", type=float, default=0.1, help="epochs between checkpoints"
    )
    p.add_argument(
        "--logger",
        default="tensorboard",
        help="'' for none (curves.py reads the console log)",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument(
        "--dry-run", action="store_true", help="print the command, run nothing"
    )
    p.add_argument(
        "--init-only",
        action="store_true",
        help="save an untrained checkpoint and exit (vLLM load check)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args, passthrough = build_parser().parse_known_args(argv)

    cmd, out = command(args, passthrough)
    k = model_config(args.model, args.k)["k"]
    block = METHODS[args.method].block_size(k)
    if not args.no_resume:
        check_resume(out / "checkpoints", args.method, block)
    print(
        f"{args.model} {args.method}  k={k} block={block} gamma={gamma_for(block)}  -> {out}"
    )
    if args.dry_run:
        print(shlex.join(cmd))
        return 0
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus}
    log = out / "logs" / "train.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"logging to {log}  (./curves.py {out} --watch)", flush=True)
    # Tee, appending: curves.py reads this, and a resumed run continues it.
    with (
        log.open("a") as handle,
        subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        ) as proc,
    ):
        for line in proc.stdout:
            sys.stdout.write(line)
            handle.write(line)
            handle.flush()
    return proc.returncode

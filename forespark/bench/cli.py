# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark ForeSpark and DSpark against their target.

./eval.py prepare                                   # the ShareGPT prompt stream
./eval.py decode-prompts qwen14 --gpus 0            # once per target: prompts whose answers outrun the span
./eval.py sharegpt-decode qwen14 --draft CKPT --method forespark --sts sts.json \\
    --fore-d-calibration fore_d.json --fore-d-costs costs.json      # decode only
./eval.py sharegpt qwen14 --draft CKPT --method dspark --adaptive-verification --sts sts.json   # E2E
./eval.py sharegpt-rate qwen14 --rates 4 8 16                       # open loop, SLOs (no --draft: the target alone)
./eval.py acceptance qwen14 --draft CKPT --method forespark --prompts P.jsonl   # fixed k, every draft verified
./eval.py report --model qwen14 --results-root results/qwen14/bench
"""

import argparse
import hashlib
import json
import socket
import subprocess
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

from .. import paths
from ..config import checkpoint_k, for_gpus, METHODS, model_config, MODELS
from ..paths import ROOT
from . import datasets, phases, report
from .protocol import cache_config, row_warnings, step_timing, wait_healthy
from .server import (
    batch_width,
    fore_d,
    GC_THRESHOLD,
    graph_capture,
    MAX_MODEL_LEN,
    parallel_label,
    serve,
    stop_server,
    token_budget,
)  # fmt: skip

PHASES = ("acceptance", "sharegpt", "sharegpt-decode", "sharegpt-rate")
DECODE_PHASE = "sharegpt-decode"  # phases.decode_spans, on the screened prompts


def arm_name(args) -> str:
    """The result's label: --arm-label, else the drafter's run directory, plus the settings that differ."""
    default_cache = "off" if args.command == "acceptance" else "on"
    settings = "".join([
        f"+T{args.temperature:g}" if args.temperature else "",
        f"+prefixCache{args.prefix_caching.capitalize()}" if args.prefix_caching != default_cache else "",
        "+stepTiming" if getattr(args, "step_timing", False) else "",
    ])  # fmt: skip
    if not args.draft:
        return "no_spec" + settings
    if args.arm_label:
        return args.arm_label + settings
    parts = [p for p in Path(args.draft).parts if p not in ("/", "")]
    leaf = parts[-1]
    if leaf.startswith("checkpoint") or leaf.isdigit():
        leaf = f"{parts[-3] if len(parts) > 2 else leaf}@{leaf}"
    return f"{leaf}+{args.method}" + settings


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(args, cfg: dict) -> dict:
    """What a rerun must match: software, hardware and inputs, beyond the caps recorded beside it."""
    try:
        gpus = subprocess.run(["nvidia-smi", "-i", args.gpus, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip().splitlines()  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        gpus = []
    code = hashlib.sha256()
    for path in sorted(
        [*(ROOT / "forespark").rglob("*.py"), *(ROOT / "patches").glob("*.patch")]
    ):
        code.update(str(path.relative_to(ROOT)).encode() + path.read_bytes())
    draft = Path(args.draft) / "config.json" if args.draft else None
    inputs = {
        "fore_d": args.fore_d_calibration,
        "costs": args.fore_d_costs,
        "sts": args.sts,
    }
    return {
        "vllm": version("vllm"),
        "torch": version("torch"),
        "host": socket.gethostname(),
        "gpus": gpus,
        "command": sys.argv,
        "harness_sha256": code.hexdigest(),
        "draft_config_sha256": digest(draft) if draft and draft.is_file() else None,
        "inputs_sha256": {name: digest(Path(p)) for name, p in inputs.items() if p},
    }


def _draft_config(draft: str) -> dict:
    config = Path(draft) / "config.json"
    return json.loads(config.read_text()) if config.exists() else {}


def validate(args, phase: str, cfg: dict) -> None:
    method = METHODS[args.method]
    if args.adaptive_verification and (method.fore_d or phase == "acceptance"):
        raise SystemExit(
            "--adaptive-verification is DSpark's (ForeSpark verifies adaptively under Fore-D); "
            "the acceptance phase verifies every draft"
        )
    if args.sts_collect and (phase != "acceptance" or not args.draft or args.sts):
        raise SystemExit(
            "--sts-collect records uncalibrated confidences in the acceptance phase: a --draft, no --sts"
        )
    if fore_d(args) and _draft_config(args.draft).get("fore_d_output_semantics") != (
        "width_conditioned_survival"
    ):
        raise SystemExit(
            f"{args.draft} has no width-conditioned Fore-D head: not a ForeSpark checkpoint"
        )
    if args.sts and not (args.adaptive_verification or fore_d(args)):
        raise SystemExit(
            "--sts calibrates the confidence that verification ranks: DSpark with --adaptive-verification, or ForeSpark"
        )
    fore_d_flags = (
        args.fore_d_calibration,
        args.fore_d_costs,
        args.fore_d_log,
        args.drafter_width,
    )
    if any(v is not None for v in fore_d_flags) and not fore_d(args):
        raise SystemExit(
            "--fore-d-calibration, --fore-d-costs, --fore-d-log and --drafter-width are Fore-D's "
            "(--method forespark, not in the acceptance phase)"
        )
    # A backend plans from host lengths: rank verification (and draft widths) on the host.
    args.cpu_ranking = bool(cfg.get("cpu_ranking")) and (
        args.adaptive_verification or fore_d(args)
    )


def target_config(args) -> dict:
    """The target's config at this run's k (--k), refusing a local drafter trained for another."""
    cfg = for_gpus(model_config(args.model, args.k), args.gpus)
    if (trained := checkpoint_k(args.draft)) is not None and trained != cfg["k"]:
        raise SystemExit(
            f"{args.draft} was trained for k = {trained}, the run is at k = {cfg['k']}: pass --k {trained}"
        )
    return cfg


def decode_inputs(args, cfg: dict, prompt_path: Path, prompts: list[dict]) -> None:
    """The decode phase's end-of-turn ids and screened prompts on args, refused before any server starts if they
    cannot serve every point (rather than at the widest point, after the narrower ones ran)."""
    args.stop_token_ids = datasets.stop_token_ids(cfg)
    args.screened_rows = datasets.load_screening(
        args.model,
        cfg.get("revision"),
        prompt_path,
        args.template_kwargs,
        args.decode_tokens,
    )
    need, widest = max((phases.decode_episodes(args.decode_sequences, c) * c, c) for c in args.concurrency)  # fmt: skip
    if args.screened_rows is not None and len(args.screened_rows) < need:
        raise SystemExit(f"C={widest} needs {need} prompts, but only {len(args.screened_rows)} screened answers "
                         f"outrun a {args.decode_tokens}-token span: pass a shorter --decode-tokens")  # fmt: skip
    if args.screened_rows is None and len(prompts) < need:
        raise SystemExit(
            f"C={widest} needs {need} prompts, only {len(prompts)} in {prompt_path}"
        )


def run_phase(args, phase: str) -> int:
    cfg = target_config(args)
    validate(args, phase, cfg)
    if phase == DECODE_PHASE and args.max_tokens <= args.decode_tokens:
        raise SystemExit("--max-tokens must exceed --decode-tokens")
    args.template_kwargs = cfg.get("sampling", {}).get("chat_template_kwargs", {})
    args.prefix_caching = args.prefix_caching or (
        "off" if phase == "acceptance" else "on"
    )
    args.api_server_count = 8 if phase in ("sharegpt", "sharegpt-rate") else 1
    args.enforce_eager = bool(
        args.sts_collect
    )  # the recorder is Python inside the draft path
    prompt_path = Path(args.prompts) if args.prompts else datasets.SHAREGPT_FILE
    prompts = datasets.load_prompts(prompt_path)
    width, arm = batch_width(args, cfg), arm_name(args)
    budget = token_budget(args, cfg, width)
    run = {"acceptance": phases.fixed_count, "sharegpt": phases.steady_state,
           DECODE_PHASE: phases.decode_spans, "sharegpt-rate": phases.open_loop}[phase]  # fmt: skip
    if phase == DECODE_PHASE:
        decode_inputs(args, cfg, prompt_path, prompts)

    stem = f"{args.model}-{arm}-{phase}-{parallel_label(args)}"
    out = Path(args.out or paths.RESULTS / f"{stem}{'' if args.round is None else f'-r{args.round}'}.json")  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    args.step_timing_file = out.with_suffix(".steps.bin") if args.step_timing else None
    barrier = (
        tempfile.TemporaryDirectory(prefix="forespark-decode-barrier-")
        if phase == DECODE_PHASE
        else None
    )
    args.barrier_dir = barrier.name if barrier else None
    args.base_url = f"http://127.0.0.1:{args.port}"
    proc = None
    rows = []
    failure = None  # (point, error) of a point that raised: the finished points go to RESULT.partial.json
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        proc = serve(args, cfg, MAX_MODEL_LEN, width, budget)
        args.server_pgid = proc.pid  # its own session: see foreign_gpu_processes
        wait_healthy(args.base_url, proc, args.startup_timeout)
        kv = cache_config(args.base_url)
        print(
            f"\n{args.model} / {arm} / {phase}: {len(prompts)} prompts, max_num_seqs {width}, budget {budget}\n"
        )
        for point in args.rates if phase == "sharegpt-rate" else args.concurrency:
            try:
                rows.append(run(args, prompts, point))
            # Re-raised once the finished points are written; interrupts too.
            except (Exception, KeyboardInterrupt) as error:
                failure = (point, error)
                break
    finally:
        if proc is not None:
            stop_server(proc)
        if barrier is not None:
            barrier.cleanup()
    if failure is not None and not rows:
        raise failure[1]

    for row in rows:
        if args.step_timing_file:  # the worker has flushed: the server is gone
            row |= step_timing(args.step_timing_file, row.get("windows", []))
        row["warnings"] = row_warnings(row)
        point = (
            f"rate={row['rate']:<5g}" if "rate" in row else f"C={row['concurrency']:<5}"
        )
        latency = (f"  ttft p99 {row['ttft_p99_ms']:.0f} ms  tpot p99 {row['tpot_p99_ms']:.1f} ms  "
                   f"SLO {row['slo_attainment']:.0%}" if "rate" in row else "")  # fmt: skip
        steps = (f"  step {row['step_ms']:.2f} ms: verify {row['step_forward_ms'] + row['step_sample_ms']:.2f} "
                 f"draft {row['step_draft_ms']:.2f} idle {row['step_idle_ms']:.2f} ({row['step_timed']} timed)"
                 if row.get("step_timed") else "")  # fmt: skip
        print(
            f"{'!' if row['warnings'] else ' '} {point} "
            f"{row['output_tokens_per_second']:>10.1f} tok/s  "
            f"accept_len {row.get('acceptance_length', float('nan')):.3f}  "
            f"D/step {row.get('draft_len_observed', float('nan')):.2f}  "
            f"B/step {row.get('verify_len_observed', float('nan')):.2f}{latency}{steps}"
        )
        for message in row["warnings"]:
            print(f"    WARNING: {message}", file=sys.stderr)
    meta = {
        "model": args.model,
        "target_revision": cfg.get("revision"),
        "arm": arm,
        "phase": phase,
        "parallel": parallel_label(args),
        "round": args.round,
        "method": args.method if args.draft else None,
        "draft": args.draft,
        "adaptive_verification": bool(args.adaptive_verification),
        "sts": args.sts,
        "fore_d_calibration": args.fore_d_calibration,
        "fore_d_costs": args.fore_d_costs,
        "k": cfg["k"],
        "drafter_width": args.drafter_width,
        "prefix_caching": args.prefix_caching,
        "temperature": args.temperature,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": width,
        "max_num_batched_tokens": budget,
        "cuda_graphs": graph_capture(args, cfg),
        "kv_cache_tokens": int(kv["num_gpu_blocks"]) * int(kv["block_size"])
        if kv.get("num_gpu_blocks")
        else None,
        "step_timing_file": args.step_timing_file and str(args.step_timing_file),
        "gc_threshold": GC_THRESHOLD,
        "started": started,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "provenance": provenance(args, cfg),
        "max_tokens": args.max_tokens,
        "prompts": str(prompt_path.resolve()),
        "prompts_sha256": datasets.prompt_digest(prompt_path),
        "results": rows,
    }
    write_result(out, meta, failure)
    return 0


def write_result(out: Path, meta: dict, failure: tuple | None) -> None:
    """The result; after a failed point, the finished ones beside it instead (a script takes an existing result
    as done and would never rerun the cell), then the point's error re-raised."""
    if failure is None:
        out.write_text(json.dumps(meta, indent=2))
        print(f"\n-> {out}")
        return
    point, error = failure
    partial = out.with_suffix(".partial.json")
    partial.write_text(
        json.dumps(meta | {"failed_point": point, "error": repr(error)}, indent=2)
    )
    print(
        f"\n-> {partial}: {len(meta['results'])} points before {point} raised {error!r}"
    )
    raise error


def run_decode_prompts(args) -> int:
    """Screen the decode phase's prompts on the target alone, greedy: keep those
    whose answer runs a quarter past the span, nearest 1k characters first.
    Every arm then draws the same batches from them, all inside answers. Each
    answer's length (to the cap) is kept, so the screening serves any shorter
    span too."""
    cfg = for_gpus(MODELS[args.model], args.gpus)
    cap = datasets.answer_floor(args.decode_tokens)
    base = build_parser().parse_args([
        DECODE_PHASE, args.model, "--gpus", args.gpus, "--port", str(args.port), "--prefix-caching", "off",
        "--max-tokens", str(cap), "--startup-timeout", str(args.startup_timeout),
        "--request-timeout", str(args.request_timeout),
    ])  # fmt: skip
    base.template_kwargs = cfg.get("sampling", {}).get("chat_template_kwargs", {})
    base.base_url, base.api_server_count, base.enforce_eager, base.cpu_ranking = (
        f"http://127.0.0.1:{args.port}", 1, False, False,
    )  # fmt: skip
    prompt_path = datasets.SHAREGPT_FILE
    prompts = datasets.load_prompts(prompt_path)
    order = phases.nearest(prompts, args.candidates)
    width = batch_width(base, cfg)
    proc = serve(base, cfg, MAX_MODEL_LEN, width, token_budget(base, cfg, width))
    try:
        wait_healthy(base.base_url, proc, args.startup_timeout)
        lengths = phases.answer_lengths(
            base, [prompts[i] for i in order], args.concurrency
        )
    finally:
        stop_server(proc)
    kept = [
        index
        for index, length in zip(order, lengths, strict=True)
        if length is not None and length >= cap
    ]
    out = datasets.screening_file(args.model)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "target_revision": cfg.get("revision"),
        "prompts": str(prompt_path.resolve()), "prompts_sha256": datasets.prompt_digest(prompt_path),
        "chat_template_kwargs": base.template_kwargs, "decode_tokens": args.decode_tokens, "max_tokens": cap,
        "screened": len(order), "rows": kept, "order": order, "lengths": lengths,
    }, indent=2))  # fmt: skip
    print(f"{len(kept)} of {len(order)} prompts answer past {cap} tokens -> {out}")
    return 0


def add_phase_parser(sub, phase: str) -> None:
    decode, steady = phase == DECODE_PHASE, phase != "acceptance"
    s = sub.add_parser(phase)
    s.add_argument("model", choices=sorted(MODELS))
    s.add_argument("--draft", help="drafter checkpoint; omit for the target alone")
    s.add_argument("--method", choices=sorted(METHODS), default="forespark")
    s.add_argument(
        "--arm-label",
        help="the result's arm name (default: the drafter's run directory)",
    )
    s.add_argument("--k", type=int, help="draft length, for every arm's sizes and budget (default: the target's); "
                   "a local drafter must have been trained for it")  # fmt: skip
    s.add_argument("--gpus", default="0")
    s.add_argument("--port", type=int, default=8100)
    s.add_argument("--step-timing", action="store_true",
                   help="per-step draft/verify breakdown from vLLM's sampled step timer")  # fmt: skip
    s.add_argument("--prompts", help="JSONL prompt file (default: the ShareGPT stream)")
    s.add_argument("--temperature", type=float, default=0.0)
    s.add_argument(
        "--adaptive-verification",
        action="store_true",
        help="DSpark: confidence-sized verification (AV)",
    )
    s.add_argument(
        "--sts", metavar="PATH", help="confidence temperatures (calibrate.py sts)"
    )
    s.add_argument(
        "--fore-d-calibration",
        metavar="PATH",
        help="ForeSpark: Fore-D calibration (calibrate.py fore-d)",
    )
    s.add_argument(
        "--fore-d-costs",
        metavar="PATH",
        help="ForeSpark: measured step costs for the planner (calibrate.py costs)",
    )
    s.add_argument(
        "--drafter-width",
        type=int,
        metavar="D",
        help="ForeSpark calibration: every request drafts D, every draft verified",
    )
    s.add_argument(
        "--fore-d-log",
        metavar="DIR",
        help="ForeSpark calibration: log Fore-D logits and outcomes (calibrate.py fore-d)",
    )
    if phase == "acceptance":
        s.add_argument(
            "--sts-collect",
            metavar="DIR",
            help="record confidence logits vs acceptance (calibrate.py sts collect)",
        )
    s.add_argument("--draft-attention-backend")
    s.add_argument(
        "--prefix-caching",
        choices=("on", "off"),
        help="default: on, off for acceptance",
    )
    s.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    s.add_argument("--max-num-seqs", type=int)
    s.add_argument("--max-num-batched-tokens", type=int)
    s.add_argument("--startup-timeout", type=float, default=1800.0)
    s.add_argument("--request-timeout", type=float, default=1800.0)
    s.add_argument("--out")
    s.add_argument(
        "--round",
        type=int,
        help="repeat index: an arm's rounds are samples, which `report` averages",
    )
    s.add_argument("--max-tokens", type=int, default=16384 if decode else 4096)
    s.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 8, 32, 128] if phase == "sharegpt" else [1],
    )
    if steady:
        s.add_argument("--warmup-seconds", type=float, default=2.0 if decode else 30.0)
        s.add_argument("--warmup-timeout", type=float, default=600.0)
        s.add_argument("--drain-seconds", type=float, default=60.0 if decode else 600.0)
    if phase in ("sharegpt", "sharegpt-rate"):
        s.add_argument("--measure-seconds", type=float, default=120.0)
    if phase == "sharegpt":
        s.add_argument("--min-window-requests", type=int, metavar="N",
                       help="keep the window open past --measure-seconds until N requests started in it")  # fmt: skip
    if phase == "sharegpt-rate":
        s.add_argument(
            "--rates",
            type=float,
            nargs="+",
            default=[1.0, 2.0, 4.0, 8.0],
            help="request rates (req/s), one point each",
        )
        s.add_argument(
            "--arrival-seed",
            type=int,
            default=0,
            help="seeds the Poisson arrival schedule, the same in every arm",
        )
        s.add_argument("--slo-ttft-ms", type=float, default=1000.0)
        s.add_argument("--slo-tpot-ms", type=float, default=50.0)
        s.add_argument(
            "--max-inflight",
            type=int,
            default=4096,
            help="arrivals beyond this many in flight are dropped, warned",
        )
    if decode:
        s.add_argument("--barrier-timeout", type=float, default=60.0)
        s.add_argument(
            "--decode-tokens",
            type=int,
            default=512,
            help="tokens timed per answer, after its first",
        )
        s.add_argument("--decode-sequences", type=int, default=64,
                       help="answers per point: batches repeat on disjoint prompts until this many")  # fmt: skip
        s.add_argument(
            "--episode-timeout",
            type=float,
            default=300.0,
            help="cap on one batch's spans",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="build the ShareGPT prompt stream")
    prep.add_argument("--sharegpt-conversations", type=int, default=20000)
    prep.add_argument("--force", action="store_true")
    rep = sub.add_parser("report", help="summarise result files")
    rep.add_argument("--model")
    rep.add_argument("--results-root", default=str(paths.RESULTS))
    rep.add_argument(
        "--compare", nargs="?", const="output_tokens_per_second", metavar="METRIC"
    )
    rep.add_argument("--phase", help="only this phase (e.g. sharegpt)")
    rep.add_argument("--arms", nargs="+", metavar="LABEL=NAME",
                     help="only these arms, in this order, shown as NAME (LABEL: the arm name up to its first '+')")  # fmt: skip
    rep.add_argument("--prefix-caching", choices=["on", "off"], help="only these runs")
    rep.add_argument("--markdown", action="store_true", help="--compare in Markdown")
    rep.add_argument("--out", help="--compare's CSV")
    screen = sub.add_parser(
        "decode-prompts",
        help="screen the decode phase's prompts by the target's answer length",
    )
    screen.add_argument("model", choices=sorted(MODELS))
    screen.add_argument("--gpus", default="0")
    screen.add_argument("--port", type=int, default=8100)
    screen.add_argument(
        "--decode-tokens",
        type=int,
        default=512,
        help="the longest span the kept prompts support",
    )
    screen.add_argument(
        "--candidates",
        type=int,
        default=4096,
        help="prompts screened, nearest 1k characters first",
    )
    screen.add_argument("--concurrency", type=int, default=256)
    screen.add_argument("--startup-timeout", type=float, default=1800.0)
    screen.add_argument("--request-timeout", type=float, default=1800.0)
    for phase in PHASES:
        add_phase_parser(sub, phase)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "prepare":
        datasets.prepare_sharegpt(args.sharegpt_conversations, 4, 16000, 0, args.force)
        return 0
    if args.command == "decode-prompts":
        return run_decode_prompts(args)
    if args.command == "report":
        root = Path(args.results_root)
        cells = report.load_cells(root, args.model, args.prefix_caching)
        cells = {
            k: v for k, v in cells.items() if args.phase is None or k[1] == args.phase
        }
        order = None
        if args.arms:
            cells, order = report.select_arms(cells, args.arms)
        if not cells:
            raise SystemExit(f"no results in {args.results_root}")
        if args.compare:
            csv_out = Path(args.out or root / f"compare-{args.compare}.csv")
            report.print_comparison(cells, args.compare, csv_out, args.markdown, order)
        else:
            report.print_tables(cells, order)
        return 0
    if not hasattr(args, "sts_collect"):
        args.sts_collect = None
    return run_phase(args, args.command)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""eval.py: how each arm maps onto the vLLM command and worker environment,
what the steady, decode and open-loop phases measure, and how report averages rounds."""

import asyncio
import json
import math
import os
import statistics
import time
from argparse import Namespace

import pytest
from forespark.bench import cli, datasets, phases, protocol as P, report, server
from forespark.config import (
    checkpoint_k,
    CPU_RANKING_ENV,
    ForeDConfig,
    GC_THRESHOLD_ENV,
    model_config,
    MODELS,
    STEP_TIMING_FILE_ENV,
    STS_COLLECT_ENV,
    STS_ENV,
    TP_CONSISTENT_VERIFY_ENV,
)  # fmt: skip

CFG = MODELS["qwen14"]


def parse(*argv: str):
    """The phase's arguments as eval.py holds them before serving: main supplies
    --sts-collect outside the acceptance phase, validate settles host ranking and
    run_phase eager serving."""
    args = cli.build_parser().parse_args(list(argv))
    for name, default in (
        ("sts_collect", None),
        ("cpu_ranking", False),
        ("enforce_eager", False),
    ):
        if not hasattr(args, name):
            setattr(args, name, default)
    return args


def checkpoint(tmp_path, semantics: str | None = "width_conditioned_survival"):
    """A drafter checkpoint's config: ForeSpark's (non-causal, width-conditioned
    Fore-D head) unless `semantics` says otherwise."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = {"sliding_window_non_causal": True}
    if semantics is not None:
        config["fore_d_output_semantics"] = semantics
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


def test_forespark_serves_k_wide_through_adaptive_verification(tmp_path):
    ckpt = checkpoint(tmp_path)
    args = parse(
        "sharegpt",
        "qwen14",
        "--draft",
        ckpt,
        "--method",
        "forespark",
        "--drafter-width",
        "3",
    )
    cli.validate(args, "sharegpt", CFG)
    spec = server.speculative_config(args, CFG)
    assert (
        spec["num_speculative_tokens"] == CFG["k"]
    )  # widths vary under k; --drafter-width is a Fore-D knob
    assert spec["enable_adaptive_verification"] and spec["method"] == "dflash"
    env = server.server_env(args, CFG, 40960)
    assert env["FORESPARK_FORE_D"] == "1" and env["FORESPARK_FORE_D_FIXED_WIDTH"] == "3"
    assert env["VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN"] == "1024"
    assert "VLLM_DISABLE_COMPILE_CACHE" not in env  # Fore-D still serves k tokens


def test_forespark_and_dspark_keep_the_last_prefix_hit_block(tmp_path):
    ckpt = checkpoint(tmp_path)
    for method in ("forespark", "dspark"):
        args = parse("sharegpt", "qwen14", "--draft", ckpt, "--method", method)
        assert server.speculative_config(args, CFG)["disable_eagle_block_drop"] is True


def test_servers_skip_the_hub_when_the_pinned_revision_is_cached(monkeypatch):
    args = parse("sharegpt", "qwen14")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(server, "revision_cached", lambda repo, revision: True)
    assert server.server_env(args, CFG, 40960)["HF_HUB_OFFLINE"] == "1"
    # Not downloaded yet: the first server fetches it.
    monkeypatch.setattr(server, "revision_cached", lambda repo, revision: False)
    assert "HF_HUB_OFFLINE" not in server.server_env(args, CFG, 40960)
    monkeypatch.setattr(server, "revision_cached", lambda repo, revision: True)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")  # the user's choice stands
    assert server.server_env(args, CFG, 40960)["HF_HUB_OFFLINE"] == "0"


def test_dspark_serves_k_and_nothing_leaks_from_the_callers_shell(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(STS_ENV, "/stale.json")
    monkeypatch.setenv("FORESPARK_FORE_D", "1")
    args = parse(
        "sharegpt",
        "qwen14",
        "--draft",
        str(tmp_path),
        "--method",
        "dspark",
        "--adaptive-verification",
    )
    assert server.speculative_config(args, CFG)["num_speculative_tokens"] == CFG["k"]
    env = server.server_env(args, CFG, 40960)
    assert (
        STS_ENV not in env and "FORESPARK_FORE_D" not in env
    )  # nothing leaks from the caller's shell
    assert env["VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN"] == "1024"


def test_validation_rules(tmp_path):
    ckpt = checkpoint(tmp_path)
    # Only a checkpoint with the width-conditioned head runs Fore-D.
    for semantics in (None, "position_survival"):
        other = checkpoint(tmp_path / str(semantics), semantics)
        with pytest.raises(SystemExit, match="no width-conditioned Fore-D head"):
            cli.validate(
                parse("sharegpt", "qwen14", "--draft", other, "--method", "forespark"),
                "sharegpt",
                CFG,
            )
    native = parse("sharegpt", "qwen14", "--draft", ckpt, "--method", "forespark")
    cli.validate(native, "sharegpt", CFG)
    assert not native.cpu_ranking  # qwen14's backends plan on the device
    with pytest.raises(SystemExit, match="--sts calibrates"):
        cli.validate(parse("sharegpt", "qwen14", "--draft", ckpt, "--method", "dspark", "--sts", "x.json"),
                     "sharegpt", CFG)  # fmt: skip
    with pytest.raises(SystemExit, match="--adaptive-verification is DSpark's"):
        cli.validate(parse("sharegpt", "qwen14", "--draft", ckpt, "--method", "forespark", "--adaptive-verification"),
                     "sharegpt", CFG)  # fmt: skip
    for flag in (
        ("--drafter-width", "3"),
        ("--fore-d-costs", "c.json"),
        ("--fore-d-calibration", "e.json"),
    ):
        with pytest.raises(SystemExit, match="are Fore-D's"):
            cli.validate(parse("sharegpt", "qwen14", "--draft", ckpt, "--method", "dspark", *flag),
                         "sharegpt", CFG)  # fmt: skip
    # The target alone: no drafter, whatever the default method.
    target = parse("sharegpt", "qwen14")
    cli.validate(target, "sharegpt", CFG)
    assert not server.fore_d(target) and not target.cpu_ranking


def test_every_tp_rank_verifies_rank_zeros_drafts(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import torch
    from forespark.serving.tp_consistency import (
        broadcast_drafts,
        broadcast_verdict,
        consistent_propose,
    )

    ckpt = checkpoint(tmp_path)
    arms = (("--method", "dspark", "--adaptive-verification"), ("--method", "forespark"),
            ("--method", "dspark"))  # fmt: skip
    for arm in arms:  # adaptive or fixed k: every speculative arm samples per rank
        tp8 = parse(
            "sharegpt", "qwen14", "--draft", ckpt, *arm, "--gpus", "0,1,2,3,4,5,6,7"
        )
        assert server.server_env(tp8, CFG, 40960)[TP_CONSISTENT_VERIFY_ENV] == "1"
        tp1 = parse("sharegpt", "qwen14", "--draft", ckpt, *arm, "--gpus", "0")
        assert TP_CONSISTENT_VERIFY_ENV not in server.server_env(tp1, CFG, 40960)
    assert TP_CONSISTENT_VERIFY_ENV not in server.server_env(
        parse("sharegpt", "qwen14", "--gpus", "0,1"), CFG, 40960
    )

    received = []

    def broadcast(tensor):  # what a peer receives: rank 0's values (here: all 9s)
        assert tensor.is_contiguous()
        received.append(tuple(tensor.shape))
        tensor.fill_(9)

    confidence = torch.zeros(
        4, 3
    )  # [max_num_reqs, K]; rows past num_reqs are untouched
    drafts = torch.zeros(6, 3, dtype=torch.long)  # per request slot
    broadcast_drafts(
        broadcast, confidence, drafts, torch.tensor([5, 1, 0, 0]), num_reqs=2
    )
    assert confidence[:2].eq(9).all() and confidence[2:].eq(0).all()
    assert drafts[5].eq(9).all() and drafts[1].eq(9).all() and drafts[0].eq(0).all()
    broadcast_drafts(
        None, confidence, drafts, torch.tensor([0]), num_reqs=1
    )  # TP1 or off: nothing to agree on

    block = torch.zeros(4, 8, dtype=torch.long)
    tokens = consistent_propose(
        lambda: block[:, :7], broadcast
    )()  # a non-contiguous view is staged
    assert tokens.eq(9).all() and block[:, 7].eq(0).all()

    output = SimpleNamespace(sampled_token_ids=torch.zeros(2, 8, dtype=torch.long))
    sampled, rejected = (
        torch.zeros(2, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
    )
    received.clear()
    broadcast_verdict(broadcast, (output, sampled, rejected), num_draft_tokens=14)
    assert (
        received == [(2, 8), (2,), (2,)]
        and sampled.eq(9).all()
        and rejected.eq(9).all()
    )
    received.clear()
    broadcast_verdict(
        broadcast, (output, sampled, None), num_draft_tokens=0
    )  # a plain decode step
    assert received == []


def test_glm_ranks_on_the_host_whenever_verification_is_adaptive(tmp_path):
    glm = MODELS["glm53flash"]
    ckpt = checkpoint(tmp_path)
    adaptive = ("--method", "dspark", "--adaptive-verification")
    for arm in (adaptive, ("--method", "forespark")):
        args = parse("sharegpt", "glm53flash", "--draft", ckpt, *arm)
        cli.validate(args, "sharegpt", glm)
        env = server.server_env(args, glm, 8194)
        assert args.cpu_ranking and env[CPU_RANKING_ENV] == "1"
    fixed = parse("sharegpt", "glm53flash", "--draft", ckpt, "--method", "dspark")
    cli.validate(fixed, "sharegpt", glm)
    assert not fixed.cpu_ranking  # fixed k verifies whole blocks: nothing to rank
    # The acceptance phase serves ForeSpark's drafter at fixed k too.
    acceptance = parse(
        "acceptance", "glm53flash", "--draft", ckpt, "--method", "forespark"
    )
    cli.validate(acceptance, "acceptance", glm)
    assert not acceptance.cpu_ranking


def test_host_ranking_captures_the_gdn_spec_kernels(monkeypatch):
    # AV captures its varlen FULL graphs as N requests x 1 token; GDN (GLM's
    # KDA) must still record its spec kernels, or replayed verify rows run the
    # plain decode ones (lossy).
    from types import SimpleNamespace

    import torch
    from forespark.serving.cpu_ranking import relax_gates
    from vllm.v1.attention.backend import AttentionBackend
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
    from vllm.v1.worker.gpu.spec_decode import adaptive_verification as av

    patched = [
        (AttentionBackend, "supports_device_cpu_query_lens_mismatch"),
        (GDNAttentionMetadataBuilder, "build_for_cudagraph_capture"),
    ]
    for owner, name in patched:  # raw descriptors, restored after the test
        monkeypatch.setattr(owner, name, owner.__dict__[name])
    create = av.maybe_create_adaptive_verification_manager
    monkeypatch.setattr(av, "maybe_create_adaptive_verification_manager", create)
    relax_gates()
    seen = {}
    builder = SimpleNamespace(use_spec_decode=True, num_spec=7, decode_cudagraph_max_bs=64,
                              build=lambda *args: seen.setdefault("args", args))  # fmt: skip
    batch = SimpleNamespace(num_reqs=4, num_actual_tokens=4, query_start_loc=torch.arange(5),
                            query_start_loc_cpu=torch.arange(5))  # fmt: skip
    GDNAttentionMetadataBuilder.build_for_cudagraph_capture(builder, batch)
    _, _, accepted, drafts = seen["args"]
    assert accepted.tolist() == [1, 1, 1, 1] and drafts.tolist() == [7, 7, 7, 7]


def test_sts_collection_runs_eager_at_fixed_k(monkeypatch, tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        json.dumps(
            {"category": "alpaca", "messages": [{"role": "user", "content": "hi"}]}
        )
        + "\n"
    )
    served = []
    monkeypatch.setattr(
        cli, "serve", lambda args, *a: served.append(args) or Namespace(pid=None)
    )
    monkeypatch.setattr(cli, "wait_healthy", lambda *a: None)
    monkeypatch.setattr(cli, "stop_server", lambda proc: None)
    monkeypatch.setattr(cli, "cache_config", lambda url: {})
    monkeypatch.setattr(
        cli.phases,
        "fixed_count",
        lambda args, rows, c: {"concurrency": c, "output_tokens_per_second": 1.0},
    )
    draft = checkpoint(tmp_path / "draft")
    argv = ["acceptance", "qwen14", "--draft", draft, "--method", "forespark", "--prompts", str(prompts),
            "--sts-collect", str(tmp_path / "raw"), "--out", str(tmp_path / "collect.json")]  # fmt: skip
    assert cli.main(argv) == 0
    (args,) = served
    assert args.enforce_eager and not server.fore_d(
        args
    )  # the drafter at fixed k, every draft verified
    spec = server.speculative_config(args, CFG)
    assert (
        "enable_adaptive_verification" not in spec
        and spec["num_speculative_tokens"] == CFG["k"]
    )
    env = server.server_env(args, CFG, 40960)
    assert env[STS_COLLECT_ENV].endswith("raw/sts") and "FORESPARK_FORE_D" not in env
    with pytest.raises(SystemExit, match="--adaptive-verification is DSpark's"):
        cli.validate(parse("acceptance", "qwen14", "--draft", draft, "--method", "dspark",
                           "--adaptive-verification", "--sts-collect", "x"), "acceptance", CFG)  # fmt: skip
    with pytest.raises(SystemExit, match="--sts-collect records uncalibrated"):
        cli.validate(parse("acceptance", "qwen14", "--draft", draft, "--method", "dspark", "--sts", "s.json",
                           "--sts-collect", "x"), "acceptance", CFG)  # fmt: skip


def test_fore_d_costs_reach_the_runtime(tmp_path):
    ckpt = checkpoint(tmp_path)
    fore_d = ("--draft", ckpt, "--method", "forespark", "--fore-d-costs", "c.json")
    args = parse("sharegpt", "qwen14", *fore_d)
    cli.validate(args, "sharegpt", CFG)
    assert server.server_env(args, CFG, 40960)["FORESPARK_FORE_D_COSTS"].endswith(
        "c.json"
    )
    fixed = parse("sharegpt", "qwen14", *fore_d[:3], "dspark", *fore_d[4:])
    with pytest.raises(SystemExit, match="are Fore-D's"):
        cli.validate(fixed, "sharegpt", CFG)


def test_sts_collection_forwards_the_drafter_backend(tmp_path, monkeypatch):
    from forespark.train import calibrate_sts

    calls = []
    monkeypatch.setattr(calibrate_sts, "alpaca_rows", lambda limit: [{"prompt": "hi"}])
    monkeypatch.setattr(calibrate_sts.subprocess, "call", lambda cmd: calls.append(cmd))
    argv = ["collect", "qwen14", "--draft", "d", "--out", str(tmp_path)]
    calibrate_sts.main([*argv, "--draft-attention-backend", "TRITON_ATTN"])
    assert calls[0][calls[0].index("--draft-attention-backend") + 1] == "TRITON_ATTN"
    assert (
        calls[0][2] == "acceptance"
        and calls[0][calls[0].index("--method") + 1] == "forespark"
    )


def test_arm_names_record_what_differs(tmp_path):
    ckpt = tmp_path / "run" / "checkpoints" / "2"
    ckpt.mkdir(parents=True)
    args = parse(
        "sharegpt",
        "qwen14",
        "--draft",
        str(ckpt),
        "--method",
        "forespark",
        "--prefix-caching",
        "on",
    )
    assert cli.arm_name(args) == "run@2+forespark"
    best = tmp_path / "run" / "checkpoints" / "checkpoint_best"
    best.mkdir()
    args = parse(
        "sharegpt",
        "qwen14",
        "--draft",
        str(best),
        "--method",
        "dspark",
        "--prefix-caching",
        "on",
    )
    assert cli.arm_name(args) == "run@checkpoint_best+dspark"
    assert cli.arm_name(
        parse("sharegpt", "qwen14", "--prefix-caching", "on")
    ).startswith("no_spec")
    labelled = parse(
        "sharegpt",
        "qwen14",
        "--draft",
        str(ckpt),
        "--arm-label",
        "forespark",
        "--prefix-caching",
        "on",
    )
    assert cli.arm_name(labelled) == "forespark"


def test_token_budget_and_width():
    args = parse("sharegpt", "qwen14", "--concurrency", "8", "128")
    width = server.batch_width(args, CFG)
    assert width == CFG["max_num_seqs"]
    assert server.token_budget(args, CFG, width) == 3 * width * (CFG["k"] + 1)
    with pytest.raises(SystemExit, match="below the widest"):
        server.batch_width(parse("sharegpt", "qwen14", "--concurrency", "1024"), CFG)


def test_k_sizes_every_arm_and_refuses_a_drafter_trained_for_another(tmp_path):
    cfg = model_config("qwen14", 7)
    assert cfg["k"] == 7 and MODELS["qwen14"]["k"] == 15
    assert model_config("qwen14", 15) is MODELS["qwen14"]
    flags = ["--draft", str(tmp_path), "--method", "dspark", "--k", "7"]
    args = parse("sharegpt", "qwen14", *flags)
    width = server.batch_width(args, cfg)
    # The baseline's budget too, so every arm interleaves prefill alike.
    assert server.token_budget(args, cfg, width) == 3 * width * 8
    budget = server.token_budget(args, cfg, 512)
    assert server.capture_sizes(cfg, 512, budget)[-1] == 512 * 8
    assert server.speculative_config(args, cfg)["num_speculative_tokens"] == 7
    proposal = {"proposal_type": "greedy", "speculative_tokens": 15}
    trained = {"speculators_config": {"proposal_methods": [proposal]}}
    (tmp_path / "config.json").write_text(json.dumps(trained))
    assert checkpoint_k(tmp_path) == 15
    assert checkpoint_k("incoai/GLM-5.3-Flash-DFlash2") is None
    with pytest.raises(SystemExit, match="trained for k = 15, the run is at k = 7"):
        cli.run_phase(args, "sharegpt")
    with pytest.raises(SystemExit, match="at least 1"):
        model_config("qwen14", 0)


def test_every_arm_captures_the_full_verify_batches(monkeypatch, tmp_path):
    """AV's varlen decode graphs are keyed by tokens alone; every arm serves k
    drafts, so the full-verify batches are the decode batches every server
    captures, and every arm captures the same graphs."""
    sizes = server.capture_sizes(CFG, 512, 12288)
    assert {n * (CFG["k"] + 1) for n in (1, 2, 4, 8, 512)} <= set(sizes)
    monkeypatch.setattr(server.subprocess, "Popen", lambda cmd, **kwargs: cmd)

    def graphs(*flags):
        args = parse(
            "sharegpt",
            "qwen14",
            "--gpus",
            "0,1,2,3,4,5,6,7",
            "--draft",
            checkpoint(tmp_path),
            *flags,
        )
        args.api_server_count, args.prefix_caching = 1, "off"
        cmd = server.serve(args, CFG, 40960, 512, 12288)
        return json.loads(cmd[cmd.index("--compilation-config") + 1])[
            "cudagraph_capture_sizes"
        ]

    assert graphs("--method", "dspark", "--adaptive-verification") == sizes
    assert graphs("--method", "forespark") == sizes
    assert graphs("--method", "dspark") == sizes


def test_target_graphs_reach_the_widest_speculative_step_under_tp(monkeypatch):
    tp8 = parse("sharegpt", "qwen14", "--gpus", "0,1,2,3,4,5,6,7")
    assert server.graph_capture(tp8, CFG) == "wide"
    tp1 = parse("sharegpt", "qwen14", "--gpus", "0")
    assert server.graph_capture(tp1, CFG) == "vllm"  # there KV cache is dearer
    sizes = server.capture_sizes(CFG, 512, 12288)
    assert sizes[-1] == 512 * (CFG["k"] + 1)
    assert {1, 2, 4, 8, 248, 256, 272, 512, 576, 640} <= set(sizes)
    assert server.capture_sizes(CFG, 512, 2048)[-1] == 2048  # within the budget
    assert server.capture_sizes({"k": 4}, 64, 12288)[-1] == 512  # vLLM's own range
    monkeypatch.setattr(server.subprocess, "Popen", lambda cmd, **kwargs: cmd)
    tp8.api_server_count, tp8.prefix_caching = 1, "off"
    cmd = server.serve(tp8, CFG, 40960, 512, 12288)
    config = json.loads(cmd[cmd.index("--compilation-config") + 1])
    assert config == {"cudagraph_capture_sizes": sizes}
    eager = parse("sharegpt", "qwen14", "--gpus", "0,1")
    eager.enforce_eager = True  # STS collection
    assert server.graph_capture(eager, CFG) == "eager"
    own = {**CFG, "vllm_args": ["--compilation-config", "{}"]}
    assert server.graph_capture(tp8, own) == "explicit"  # a target's own config wins


def test_sts_collector_skips_padded_slots(tmp_path):
    # Dummy runs (memory profiling, warmup) pad request slots with -1.
    import numpy as np
    import torch
    from forespark.serving.sts import Collector

    collector = Collector(tmp_path / "sts", 3, 4)
    collector.record_proposal(torch.tensor([-1, -1]), torch.zeros(2, 3))
    collector.record_proposal(
        torch.tensor([2, -1]), torch.tensor([[1.0, 2.0, 3.0], [9.0, 9.0, 9.0]])
    )
    collector.record_verification(
        torch.tensor([2, -1]), np.array([3, 3]), torch.tensor([1, 2])
    )
    assert collector.pairs == 1
    assert collector._logits[0].tolist() == [[1.0, 2.0, 3.0]] and collector._survived[
        0
    ].tolist() == [[1.0, 0.0, 0.0]]


def test_blackwell_only_args_are_appended_on_sm100_only(monkeypatch):
    from forespark import config

    glm = MODELS["glm53flash"]
    monkeypatch.setattr(
        config, "gpu_arch", lambda gpus: 9
    )  # H100: vLLM refuses an FP8 cache with FlashInfer's SM90 sparse MLA
    hopper = config.for_gpus(glm, "0")
    assert (
        hopper["vllm_args"] == glm["vllm_args"]
        and hopper["vllm_args"][-1] == "bfloat16"
    )
    monkeypatch.setattr(config, "gpu_arch", lambda gpus: 10)
    blackwell = config.for_gpus(glm, "0")
    assert (
        blackwell["vllm_args"][: len(glm["vllm_args"])] == glm["vllm_args"]
    )  # appended to, not replaced
    assert blackwell["vllm_args"][len(glm["vllm_args"]) :] == [
        "--kv-cache-dtype",
        "fp8",
    ]
    assert config.for_gpus(CFG, "0") is CFG  # no blackwell block: untouched


def test_nvcc_targets_get_the_matching_local_toolkit_unless_cuda_home_is_set(
    monkeypatch, tmp_path
):
    import torch
    from forespark import config

    monkeypatch.setattr(config, "CUDA_TOOLKITS", tmp_path)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    with pytest.raises(
        SystemExit, match="set CUDA_HOME"
    ):  # no matching toolkit: say so before vLLM fails cryptically
        config.jit_env(MODELS["glm53"])
    nvcc = tmp_path / f"cuda-{torch.version.cuda}/bin/nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.touch()
    monkeypatch.setenv("PATH", "/usr/bin")
    # The venv's bin goes last: it supplies ninja and shadows nothing.
    venv_bin = f"/usr/bin{os.pathsep}{config.ROOT / '.venv/bin'}"
    expected = {"CUDA_HOME": str(nvcc.parents[1]), "PATH": venv_bin}
    assert config.jit_env(MODELS["glm53"]) == expected
    assert config.jit_env(CFG) == {}  # qwen14 compiles nothing at startup
    monkeypatch.setenv("CUDA_HOME", "/opt/cuda")
    # The caller's own CUDA_HOME passes through.
    assert config.jit_env(MODELS["glm53"]) == {"PATH": venv_bin}


def progress(
    token_ids: list[int], arrivals: list[tuple[float, int]], started: float = 0.0
) -> P.Progress:
    return P.Progress(started=started, arrivals=arrivals, token_ids=token_ids)


def test_decode_spans_time_each_answer_over_the_same_tokens():
    arrivals = [
        (0.0, 1),
        (1.0, 3),
        (1.1, 5),
        (1.2, 7),
    ]  # first token held at the barrier, then released
    free = progress([9, 9, 9, 9, 9, 9, 9], arrivals)
    assert phases.span_end(free.token_ids, 4, frozenset({2})) == (4, False)
    assert phases.span_rate(free, 4, frozenset({2})) == pytest.approx(
        (4, 20.0, False)
    )  # 2 tokens in 0.1 s
    ended = progress(
        [9, 9, 9, 2, 9, 9, 9], arrivals
    )  # the answer ends at its fourth token
    assert phases.span_rate(ended, 4, frozenset({2})) == pytest.approx(
        (3, 10.0, True)
    )  # post-EOS text uncounted
    assert (
        phases.span_rate(progress([2, 9, 9], arrivals[:2]), 4, frozenset({2})) is None
    )  # empty answer
    assert (
        phases.span_rate(progress([9, 9, 9], arrivals[:2]), 4, frozenset()) is None
    )  # never completed
    coarse = progress(
        [9] * 9, [(0.0, 1), (1.0, 3), (1.2, 9)]
    )  # eight-token chunks overshoot the span
    assert phases.span_rate(coarse, 4, frozenset()) == pytest.approx(
        (4, 30.0, False)
    )  # all 6 arrived by 1.2 s


def test_decode_batches_repeat_on_disjoint_nested_prompts():
    prompts = [{"messages": [{"content": "x" * (1000 + i)}]} for i in range(40)]
    assert (
        phases.decode_rows(prompts, 8) == phases.decode_rows(prompts, 32)[:8]
    )  # same rows at every width
    with pytest.raises(ValueError, match="need 64"):
        phases.decode_rows(prompts, 64)


def test_decode_prompts_come_from_a_matching_screening(monkeypatch, tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"messages": []}) + "\n")
    monkeypatch.setattr(datasets, "RESULTS", tmp_path)
    assert (
        datasets.load_screening("m", "rev", prompts, {}, 512) is None
    )  # unscreened: nearest 1k characters, warned
    screening = {"prompts_sha256": datasets.prompt_digest(prompts), "target_revision": "rev",
                 "chat_template_kwargs": {}, "decode_tokens": 512, "rows": [3, 1], "order": [3, 1],
                 "lengths": [700, 700]}  # fmt: skip
    path = datasets.screening_file("m")
    assert path == tmp_path / "m" / "calibration" / "decode-prompts.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(screening))
    assert datasets.load_screening("m", "rev", prompts, {}, 256) == [3, 1]
    with pytest.raises(SystemExit, match="512-token spans"):
        datasets.load_screening("m", "rev", prompts, {}, 1024)
    with pytest.raises(SystemExit, match="other prompts, weights"):
        datasets.load_screening("m", "new-rev", prompts, {}, 512)
    rows = [{"messages": [{"content": str(i)}]} for i in range(4)]
    assert phases.decode_rows(rows, 2, [3, 1]) == [rows[3], rows[1]]


def test_screening_serves_every_shorter_span(monkeypatch, tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"messages": []}) + "\n")
    monkeypatch.setattr(datasets, "RESULTS", tmp_path)
    screening = {"prompts_sha256": datasets.prompt_digest(prompts), "target_revision": "rev",
                 "chat_template_kwargs": {}, "decode_tokens": 512, "rows": [5], "order": [5, 2, 7],
                 "lengths": [641, 321, None]}  # fmt: skip
    path = datasets.screening_file("m")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(screening))
    assert (datasets.answer_floor(512), datasets.answer_floor(256)) == (641, 321)
    assert datasets.load_screening("m", "rev", prompts, {}, 512) == [5]
    assert datasets.load_screening("m", "rev", prompts, {}, 256) == [5, 2]
    with pytest.raises(SystemExit, match="512-token spans"):
        datasets.load_screening("m", "rev", prompts, {}, 1024)


def test_decode_prompts_screens_the_nearest_prompts_keeping_each_length(
    monkeypatch, tmp_path, capsys
):
    prompts = tmp_path / "sharegpt.jsonl"
    contents = {
        "a": 1000,
        "b": 1030,
        "c": 900,
        "d": 2000,
    }  # characters: b, a, c, d nearest 1k
    prompts.write_text("".join(json.dumps({"category": "sharegpt", "messages": [{"role": "user", "content": c * n}]})
                               + "\n" for c, n in contents.items()))  # fmt: skip
    monkeypatch.setattr(datasets, "SHAREGPT_FILE", prompts)
    monkeypatch.setattr(datasets, "RESULTS", tmp_path)
    monkeypatch.setattr(cli, "serve", lambda *a: Namespace(pid=None))
    monkeypatch.setattr(cli, "wait_healthy", lambda *a: None)
    monkeypatch.setattr(cli, "stop_server", lambda proc: None)
    answer = {"a": 700, "b": 700, "c": 400, "d": 700}
    monkeypatch.setattr(
        cli.phases,
        "answer_lengths",
        lambda args, rows, c: [answer[r["messages"][0]["content"][0]] for r in rows],
    )
    assert cli.main(["decode-prompts", "qwen14"]) == 0
    data = json.loads(datasets.screening_file("qwen14").read_text())
    assert (data["order"], data["lengths"], data["rows"]) == (
        [1, 0, 2, 3],
        [700, 700, 400, 700],
        [1, 0, 3],
    )
    assert data["max_tokens"] == 641 and data["decode_tokens"] == 512
    assert "3 of 4 prompts answer past 641 tokens" in capsys.readouterr().out


def test_decode_runs_refuse_a_screening_too_small_for_the_widest_point(
    monkeypatch, tmp_path
):
    args = cli.build_parser().parse_args(["sharegpt-decode", "qwen14"])
    assert (args.decode_tokens, args.max_tokens, args.concurrency) == (512, 16384, [1])
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("".join(json.dumps({"category": "sharegpt", "messages": [{"role": "user", "content": str(i)}]})
                               + "\n" for i in range(3)))  # fmt: skip
    monkeypatch.setattr(datasets, "RESULTS", tmp_path)
    monkeypatch.setattr(datasets, "stop_token_ids", lambda cfg: [0])
    monkeypatch.setattr(
        cli, "serve", lambda *a: pytest.fail("refused before any server starts")
    )
    cfg = model_config("qwen14", None)
    path = datasets.screening_file("qwen14")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"prompts_sha256": datasets.prompt_digest(prompts), "target_revision": cfg["revision"],
                                "chat_template_kwargs": cfg["sampling"]["chat_template_kwargs"], "decode_tokens": 512,
                                "rows": [0, 1, 2], "order": [0, 1, 2], "lengths": [700, 700, 700]}))  # fmt: skip
    assert (
        phases.decode_episodes(64, 8) * 8 == 64 and phases.decode_episodes(64, 512) == 1
    )
    with pytest.raises(
        SystemExit,
        match="C=8 needs 64 prompts, but only 3 screened answers outrun a 512-token",
    ):
        cli.main(
            [
                "sharegpt-decode",
                "qwen14",
                "--prompts",
                str(prompts),
                "--concurrency",
                "1",
                "8",
            ]
        )


def test_decode_summary_sums_span_rates_per_batch():
    args = Namespace(
        decode_tokens=4, stop_token_ids=[2], draft=None, screened_rows=[0, 1, 2, 3]
    )
    episode = {"complete": True, "premature": 0, "prefill_seconds": 0.5, "start_tokens": 2.0,
               "prefix_cache_hit_rate": 0.0, "gauges": {"running_min": 2.0, "running_peak": 2.0},
               "counters": {"vllm:generation_tokens_total": 10.0}, "server_rate": 41.0}  # fmt: skip
    episodes = [episode | {"spans": [(4, 10.0, False), (3, 30.0, True)], "window": [1.0, 2.0]},
                episode | {"spans": [(4, 20.0, False), None], "window": [5.0, 6.0]}]  # fmt: skip
    row = phases._decode_summary(args, 2, episodes)
    assert row["windows"] == [[1.0, 2.0], [5.0, 6.0]]  # where timed steps count
    assert row["output_tokens_per_second"] == pytest.approx(
        (40.0 + 40.0) / 2
    )  # a lost span scales its batch
    assert row["per_user_output_tokens_per_second"] == pytest.approx(20.0)
    assert (row["sequences"], row["natural_ends"], row["output_tokens"]) == (3, 1, 20.0)
    assert any("answers ended before 4 tokens" in w for w in P.row_warnings(row))


def test_warmup_holds_out_whole_conversations():
    prompts = [{"conversation": c, "turn": t} for c in range(16) for t in range(2)]
    warm, measured = phases.split_warmup(prompts)
    assert {r["conversation"] for r in warm} == {7, 15}
    assert not {r["conversation"] for r in warm} & {r["conversation"] for r in measured}
    lone = [{"conversation": 7}]
    assert phases.split_warmup(lone) == (lone, lone)


def test_window_censors_requests_that_outlive_the_drain():
    window = P.Window()
    assert (
        window.dispatch() is None and not window.started
    )  # warmup requests do not count
    window.open()
    done, running = window.dispatch(), window.dispatch()
    window.close()
    window.finish(done, P.Completion(tokens=11, elapsed=1.5, ttft=0.5, tpot=0.1))
    running.started, running.arrivals = 9.9, [(10.0, 1), (10.5, 11)]
    window.censor()
    window.finish(
        running, P.Completion(tokens=40, elapsed=9.0, ttft=0.1, tpot=0.001)
    )  # after the drain: ignored
    summary = window.summary()
    assert (summary["completed"], summary["censored"]) == (1, 1)
    assert summary["median_tpot_ms"] == pytest.approx(
        (100.0 + 50.0) / 2
    )  # the censored one to the deadline
    assert summary["latency_p50_ms"] == pytest.approx(
        1500.0
    )  # end-to-end: finished requests only


def test_requests_carry_no_seed_and_every_setting_names_the_arm():
    args = parse("sharegpt", "qwen14")
    args.template_kwargs = {}
    assert "seed" not in phases.request_body(args, {"messages": []})
    varied = parse(
        "sharegpt", "qwen14", "--temperature", "1.0", "--prefix-caching", "off"
    )
    assert cli.arm_name(varied) == "no_spec+T1+prefixCacheOff"
    assert (
        cli.arm_name(parse("sharegpt-decode", "qwen14", "--prefix-caching", "on"))
        == "no_spec"
    )  # its default
    assert (
        cli.arm_name(parse("acceptance", "qwen14", "--prefix-caching", "on"))
        == "no_spec+prefixCacheOn"
    )


def test_a_censored_request_still_waiting_counts_its_wait():
    window = P.Window()
    window.open()
    waiting = window.dispatch()
    waiting.started = time.monotonic() - 5.0  # dispatched 5 s ago, no first token yet
    window.censor()
    summary = window.summary()
    assert (
        summary["censored_before_first_token"] == 1
        and summary["median_ttft_ms"] >= 5000.0
    )


def test_environment_warnings_name_stalls_other_gpu_work_and_client_lag():
    assert P.rate_floor([(0.0, 0.0), (1.0, 100.0), (2.0, 100.0), (3.0, 200.0)]) == {
        "token_rate_floor": 0.0
    }
    row = {"concurrency": 512, "token_rate_floor": 0.1, "foreign_gpu_processes": 2, "client_lag_p99_ms": 80.0,
           "running_peak": 510.0, "running_mean": 451.0, "waiting_mean": 0.0, "kv_cache_usage_peak": 0.15}  # fmt: skip
    warnings = " | ".join(P.row_warnings(row))
    for text in (
        "a stall",
        "2 other GPU processes",
        "lagged 80 ms",
        "451 of 512 requests with none waiting",
    ):
        assert text in warnings
    decode = {"concurrency": 8, "measurement_kind": "decode_span", "prompts_screened": True,
              "output_tokens_per_second": 700.0, "server_output_tokens_per_second": 1000.0}  # fmt: skip
    assert any("client-timed 700 tok/s" in w for w in P.row_warnings(decode))


def test_cache_config_reads_the_info_gauge_labels(monkeypatch):
    text = 'vllm:cache_config_info{block_size="16",engine="0",num_gpu_blocks="14706"} 1.0\n'
    monkeypatch.setattr(P, "get", lambda url, timeout=10.0: text)
    assert P.cache_config("http://x") == {
        "block_size": "16",
        "engine": "0",
        "num_gpu_blocks": "14706",
    }


def test_open_loop_arrivals_are_seeded_and_poisson():
    poisson = phases.arrival_times(4.0, 4000, "0:4.0")
    assert poisson == phases.arrival_times(
        4.0, 4000, "0:4.0"
    )  # the same schedule in every arm
    gaps = [b - a for a, b in zip([0.0, *poisson], poisson)]
    assert statistics.fmean(gaps) == pytest.approx(0.25, rel=0.05)
    assert statistics.stdev(gaps) == pytest.approx(
        0.25, rel=0.1
    )  # exponential: its mean


def test_open_loop_counts_unfinished_and_failed_requests_as_slo_misses():
    records = [[0.0, 100.0, 20.0, 5000.0, 250, phases.DONE],       # meets TTFT 1 s / TPOT 50 ms
               [0.5, 2500.0, 20.0, 9000.0, 250, phases.DONE],      # TTFT miss
               [1.0, 3000.0, 30.0, None, 40, phases.CENSORED],     # outlived the drain
               [1.5, 0.0, 0.0, None, 0, phases.FAILED]]  # fmt: skip
    stats = phases.latency_stats(records, slo_ttft_ms=1000.0, slo_tpot_ms=50.0)
    assert stats["slo_attainment"] == pytest.approx(0.25)
    assert (
        stats["ttft_p99_ms"] > 2500.0 and stats["e2e_p99_ms"] <= 9000.0
    )  # failed excluded; censored a lower bound
    assert stats["normalized_latency_ms"] == pytest.approx((20.0 + 36.0) / 2)
    row = {
        "measurement_kind": "open_loop",
        "dropped": 3,
        "censored": 1,
        "failed": 1,
        "arrivals_measured": 4,
    }
    warnings = " | ".join(P.row_warnings(row))
    assert (
        "3 measured arrivals dropped" in warnings
        and "outlived the drain" in warnings
        and "1 of 4" in warnings
    )


def test_open_loop_runs_one_point_per_rate(monkeypatch, tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n"
    )
    monkeypatch.setattr(cli.paths, "RESULTS", tmp_path)
    monkeypatch.setattr(cli, "serve", lambda *a: Namespace(pid=None))
    monkeypatch.setattr(cli, "wait_healthy", lambda *a: None)
    monkeypatch.setattr(cli, "stop_server", lambda proc: None)
    fake = {"measurement_kind": "open_loop", "output_tokens_per_second": 1.0, "ttft_p99_ms": 1.0, "tpot_p99_ms": 1.0,
            "slo_attainment": 1.0}  # fmt: skip
    monkeypatch.setattr(
        cli.phases, "open_loop", lambda args, rows, rate: fake | {"rate": rate}
    )
    argv = [
        "sharegpt-rate",
        "qwen14",
        "--prompts",
        str(prompts),
        "--rates",
        "2",
        "4",
    ]
    assert cli.main(argv) == 0
    (out,) = tmp_path.glob("*.json")
    meta = json.loads(out.read_text())
    assert [r["rate"] for r in meta["results"]] == [2.0, 4.0] and meta[
        "arm"
    ] == "no_spec"


def write_result(path, arm: str, tok_s: float, round_: int | None):
    rows = [
        {"concurrency": 8, "output_tokens_per_second": tok_s, "median_tpot_ms": 10.0}
    ]
    path.write_text(json.dumps({"model": "m", "phase": "sharegpt", "parallel": "tp1", "arm": arm, "round": round_,
                                "results": rows}))  # fmt: skip


def test_report_averages_rounds_with_intervals(tmp_path):
    write_result(
        tmp_path / "old.json", "a", 50.0, None
    )  # superseded by the tagged rounds
    for round_, tok_s in ((1, 100.0), (2, 110.0), (3, 120.0)):
        write_result(tmp_path / f"a-r{round_}.json", "a", tok_s, round_)
    write_result(tmp_path / "b1.json", "b", 70.0, None)
    write_result(tmp_path / "b2.json", "b", 80.0, None)  # newer untagged: a correction
    cells = report.load_cells(tmp_path)
    a, b = (
        cells[("m", "sharegpt", "tp1", "a", 8)],
        cells[("m", "sharegpt", "tp1", "b", 8)],
    )
    assert (a["rounds"], a["output_tokens_per_second"]) == (3, pytest.approx(110.0))
    assert a["output_tokens_per_second_ci"] == pytest.approx(
        4.303 * 10.0 / math.sqrt(3)
    )
    assert a["per_user_tokens_per_second"] == pytest.approx(100.0)
    assert (b["rounds"], b["output_tokens_per_second"]) == (1, 80.0)


def test_report_splits_runs_by_prefix_caching(tmp_path, capsys):
    runs = {"ref": 100.0, "forespark": 110.0, "ref+off": 80.0, "forespark+off": 90.0}
    for arm, tok_s in runs.items():
        path = tmp_path / f"{arm}.json"
        write_result(path, arm, tok_s, 1)
        caching = "off" if arm.endswith("+off") else "on"
        path.write_text(
            json.dumps(json.loads(path.read_text()) | {"prefix_caching": caching})
        )
    cached = report.load_cells(tmp_path, prefix_caching="on")
    assert {key[3] for key in cached} == {"ref", "forespark"}
    argv = ["report", "--results-root", str(tmp_path), "--phase", "sharegpt"]
    assert cli.main([*argv, "--prefix-caching", "off", "--compare"]) == 0
    out = capsys.readouterr().out
    assert (
        "80.0000" in out and "90.0000" in out and "110.0000" not in out
    )  # the uncached runs alone


def test_comparison_prints_markdown(tmp_path, capsys):
    runs = [("a", "code", 5.0), ("a", "math", 4.0)]
    runs += [("b", "code", 4.5), ("b", "math", 3.0)]
    cells = {}
    for arm, category, value in runs:
        cells[("m", "acceptance", "tp1", arm, category)] = {"acceptance_length": value}
    report.print_comparison(cells, "acceptance_length", tmp_path / "c.csv", True)
    lines = capsys.readouterr().out.splitlines()
    assert "| category | a | b |" in lines and "| code | 5.000 | 4.500 |" in lines
    assert "| macro-average | 4.500 | 3.750 |" in lines
    assert not (tmp_path / "c.csv").exists()  # the Markdown replaces the CSV


def test_rounds_name_their_result_files(monkeypatch, tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n"
    )
    monkeypatch.setattr(cli.paths, "RESULTS", tmp_path)
    monkeypatch.setattr(cli, "serve", lambda *a: Namespace(pid=None))
    monkeypatch.setattr(cli, "wait_healthy", lambda *a: None)
    monkeypatch.setattr(cli, "stop_server", lambda proc: None)
    monkeypatch.setattr(
        cli.phases,
        "steady_state",
        lambda args, rows, c: {"concurrency": c, "output_tokens_per_second": 1.0},
    )
    assert (
        cli.main(
            [
                "sharegpt",
                "qwen14",
                "--prompts",
                str(prompts),
                "--concurrency",
                "1",
                "--round",
                "2",
            ]
        )
        == 0
    )
    (out,) = tmp_path.glob("*-r2.json")
    meta = json.loads(out.read_text())
    assert meta["round"] == 2 and meta["gc_threshold"] == server.GC_THRESHOLD
    assert {"vllm", "torch", "gpus", "command", "harness_sha256"} <= set(
        meta["provenance"]
    )
    with pytest.raises(SystemExit, match="must exceed"):
        cli.main(["sharegpt-decode", "qwen14", "--max-tokens", "100"])


def test_a_failed_point_keeps_the_finished_ones_beside_the_result(
    monkeypatch, tmp_path
):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n"
    )
    monkeypatch.setattr(cli.paths, "RESULTS", tmp_path)
    monkeypatch.setattr(cli, "serve", lambda *a: Namespace(pid=None))
    monkeypatch.setattr(cli, "wait_healthy", lambda *a: None)
    stopped = []
    monkeypatch.setattr(cli, "stop_server", lambda proc: stopped.append(proc))

    def point(args, rows, c):
        if c == 8:
            raise RuntimeError("decode spans need 8 prompts")
        return {"concurrency": c, "output_tokens_per_second": 1.0}

    monkeypatch.setattr(cli.phases, "steady_state", point)
    out = tmp_path / "cell.json"
    with pytest.raises(RuntimeError, match="need 8 prompts"):
        cli.main(
            [
                "sharegpt",
                "qwen14",
                "--prompts",
                str(prompts),
                "--concurrency",
                "1",
                "8",
                "32",
                "--out",
                str(out),
            ]
        )
    assert (
        stopped and not out.exists()
    )  # the server stopped; no result, so a script reruns the cell
    partial = json.loads((tmp_path / "cell.partial.json").read_text())
    assert [r["concurrency"] for r in partial["results"]] == [1]
    assert partial["failed_point"] == 8 and "need 8 prompts" in partial["error"]


def write_steps(path, rows):
    from array import array

    path.write_bytes(array("d", [value for row in rows for value in row]).tobytes())


def test_step_timing_averages_the_steps_inside_windows(tmp_path):
    path = tmp_path / "x.steps.bin"
    write_steps(path, [  # monotonic, reqs, tokens, drafted, full graph, prep, forward, sample, draft, post, idle
        [1.0, 8, 24, 56, 1, 0.1, 4.0, 0.5, 2.0, 0.1, 0.3],
        [2.0, 8, 24, 56, 0, 0.1, 6.0, 0.5, 2.0, 0.1, 0.3],
        [9.0, 8, 24, 56, 1, 9.9, 9.9, 9.9, 9.9, 9.9, 9.9],  # after the window
    ])  # fmt: skip
    stats = P.step_timing(path, [[0.5, 2.5]])
    assert stats["step_timed"] == 2 and stats["step_drafted"] == 56
    assert stats["step_full_cudagraph"] == pytest.approx(0.5)  # a share of the steps
    assert stats["step_forward_ms"] == pytest.approx(5.0)
    parts = sum(stats[f"step_{name}"] for name in P.STEP_SEGMENTS)
    assert stats["step_ms"] == pytest.approx(parts) == pytest.approx(8.0)
    assert P.step_timing(path, [[3.0, 4.0]]) == {"step_timed": 0}
    assert P.step_timing(tmp_path / "none.bin", [[0.0, 9.0]]) == {"step_timed": 0}
    with path.open("ab") as f:  # a row still being written
        f.write(b"\0" * 20)
    assert P.step_timing(path, [[0.5, 2.5]])["step_timed"] == 2


def test_step_timing_reads_the_format_vllm_writes():
    timing = pytest.importorskip("vllm.v1.worker.gpu.async_utils")
    if not hasattr(timing, "STEP_TIMING_FIELDS"):
        pytest.skip("vLLM without patches/vllm.patch")
    assert timing.STEP_TIMING_FIELDS == P.STEP_TIMING_FIELDS
    assert timing.STEP_TIMING_FILE_ENV == STEP_TIMING_FILE_ENV


def test_step_timing_is_tagged_and_reaches_the_server(monkeypatch, tmp_path):
    monkeypatch.setenv(STEP_TIMING_FILE_ENV, "/elsewhere")
    plain = server.server_env(parse("sharegpt", "qwen14"), CFG, 40960)
    assert STEP_TIMING_FILE_ENV not in plain
    args = parse("sharegpt", "qwen14", "--step-timing", "--prefix-caching", "on")
    args.step_timing_file = tmp_path / "x.steps.bin"
    assert cli.arm_name(args) == "no_spec+stepTiming"
    env = server.server_env(args, CFG, 40960)
    assert env[STEP_TIMING_FILE_ENV] == str(args.step_timing_file.resolve())


def test_vllm_step_timer_never_waits_on_the_gpu(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    timing = pytest.importorskip("vllm.v1.worker.gpu.async_utils")
    if not hasattr(timing, "STEP_TIMING_FIELDS"):
        pytest.skip("vLLM without patches/vllm.patch")
    path = tmp_path / "t.steps.bin"
    monkeypatch.setenv(timing.STEP_TIMING_FILE_ENV, str(path))
    monkeypatch.setenv(timing.STEP_TIMING_INTERVAL_ENV, "0")  # time every step
    collector, x = timing.StepTimingCollector(), torch.randn(2048, 2048, device="cuda")

    def waited(*_args, **_kwargs):
        raise AssertionError("the step timer waited on the GPU")

    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda.Event, "synchronize", waited)
        patch.setattr(torch.cuda, "synchronize", waited)
        for _ in range(20):
            collector.step_begin()
            collector.record_batch(Namespace(num_tokens=24, num_reqs=8), True)
            collector.forward_start()
            x @ x
            collector.forward_end()
            collector.sample_end()
            x @ x
            collector.drafter_end(56)
            collector.step_end()
    collector._runtime.close()
    stats = P.step_timing(path, [[0.0, math.inf]])
    assert stats["step_timed"] == 19  # the last step has no next start
    assert min(stats["step_forward_ms"], stats["step_draft_ms"]) > 0
    assert stats["step_drafted"] == 56


def test_every_arm_serves_with_the_gc_thresholds(monkeypatch):
    import gc

    import forespark

    args = parse("sharegpt", "qwen14", "--prefix-caching", "on")
    assert server.server_env(args, CFG, 40960)[GC_THRESHOLD_ENV] == server.GC_THRESHOLD
    assert cli.arm_name(args) == "no_spec"  # every arm's: no tag
    saved = gc.get_threshold()
    monkeypatch.setenv(GC_THRESHOLD_ENV, "50000,10,100")
    try:
        forespark.apply_gc_threshold()  # what the plugin does in every server process
        assert gc.get_threshold() == (50000, 10, 100)
    finally:
        gc.set_threshold(*saved)


def test_min_window_requests_holds_the_window_open(monkeypatch):
    def gauges(url, seconds, interval=2.0, stop=None):
        while stop is not None and not stop.is_set():
            time.sleep(0.01)
        return {"running_mean": 1.0}

    monkeypatch.setattr(phases.P, "sample_gauges", gauges)
    window = P.Window()
    window.open()
    args = Namespace(base_url="http://x", measure_seconds=0.05, min_window_requests=3)

    async def measure():
        async def feed():  # low concurrency: one request every 0.1 s
            for _ in range(3):
                await asyncio.sleep(0.1)
                window.dispatch()

        started, feeder = time.monotonic(), asyncio.create_task(feed())
        stats = await phases._measure(args, window)
        await feeder
        return stats, time.monotonic() - started

    stats, elapsed = asyncio.run(measure())
    assert stats["min_window_requests_met"] and window.dispatched == 3
    assert elapsed >= 0.3  # past --measure-seconds, until the third request started


def test_each_point_resets_the_prefix_cache(monkeypatch):
    posts = []
    monkeypatch.setattr(
        phases.P, "post", lambda url, *args, **kwargs: posts.append(url)
    )

    async def windowed(args, concurrency, window, worker):  # no server: an empty window
        return {}, {}, 1.0, {}, {}, True

    monkeypatch.setattr(phases, "_windowed", windowed)
    for name in ("delta", "preemptions", "prefix_cache_hit_rate"):
        monkeypatch.setattr(phases.P, name, lambda *args, **kwargs: 0.0)
    rows = [{"category": "a", "messages": [{"role": "user", "content": "x"}]}]
    point = {"base_url": "http://x", "min_window_requests": None, "draft": None}
    for caching, resets in (("on", ["http://x/reset_prefix_cache"]), ("off", [])):
        posts.clear()
        phases.steady_state(Namespace(**point, prefix_caching=caching), rows, 8)
        assert posts == resets


def test_report_arms_show_the_papers_names_in_order():
    record = {"output_tokens_per_second": 1.0}
    arms = (
        "no_spec+stepTiming",
        "forespark+stepTiming",
        "dspark-old+adaptive",
        "dspark+stepTiming",
    )
    cells = {("m", "sharegpt", "tp8", arm, 8): record for arm in arms}
    picked, order = report.select_arms(
        cells, ["no_spec=AR", "dspark=DSpark", "forespark=ForeSpark"]
    )
    assert order == ["AR", "DSpark", "ForeSpark"]
    assert sorted(key[3] for key in picked) == [
        "AR",
        "DSpark",
        "ForeSpark",
    ]  # dspark-old is not dspark
    assert report.ranked({"ForeSpark", "zzz", "AR", "DSpark"}, order) == [
        "AR",
        "DSpark",
        "ForeSpark",
        "zzz",
    ]
    cells[("m", "sharegpt", "tp8", "forespark+other", 8)] = record
    with pytest.raises(SystemExit, match="both show as ForeSpark"):
        report.select_arms(cells, ["forespark=ForeSpark"])


def test_fore_d_config_round_trips_through_the_environment(monkeypatch, tmp_path):
    config = ForeDConfig(enabled=True, fore_d_calibration=tmp_path / "e.json", costs=tmp_path / "c.json",
                        fixed_width=3, log=tmp_path / "log")  # fmt: skip
    for name, value in config.to_env().items():
        monkeypatch.setenv(name, value)
    assert ForeDConfig.from_env() == config

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""train.py builds the right speculators command for each drafter."""

import json

import pytest
from forespark.train import launch


@pytest.fixture(autouse=True)
def pinned(monkeypatch):
    """The target's pinned snapshot, without the Hugging Face cache."""

    def snapshot(cfg):
        return f"/snapshots/{cfg['repo']}@{cfg['revision']}"

    monkeypatch.setattr(launch, "target_snapshot", snapshot)


def build(method: str, *extra: str) -> list[str]:
    args, passthrough = launch.build_parser().parse_known_args(
        ["qwen14", method, "--gpus", "0,1", *extra]
    )
    return launch.command(args, passthrough)[0]


def value(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def test_forespark_trains_non_causal_with_fore_d_through_our_entry():
    cmd = build("forespark")
    assert value(cmd, "--speculator-type") == "dflash2_conf"
    assert cmd[cmd.index("-m") + 1] == "forespark.train.entry"
    assert (
        "--sliding-window-non-causal" in cmd
        and "--no-sliding-window-non-causal" not in cmd
    )  # width-conditioned
    assert {"--enable-fore-d-predictor", "--enable-confidence-head"} <= set(cmd)
    assert value(cmd, "--block-size") == "16"  # k = 15, dflash feeds the anchor
    assert "--fsdp-shard" in cmd and "--fsdp-shard" in build(
        "dspark"
    )  # every drafter sharded by default
    assert "--fsdp-shard" not in build("forespark", "--no-fsdp-shard")
    assert value(cmd, "--fore-d-narrow-stride") == "4"  # narrow rows: every 4th anchor
    every = build("forespark", "--fore-d-narrow-stride", "1")
    assert value(every, "--fore-d-narrow-stride") == "1"
    assert every.count("--fore-d-narrow-stride") == 1


def test_methods_keep_their_recipes():
    forespark, dspark = build("forespark"), build("dspark")
    assert "--enable-fore-d-predictor" not in dspark
    assert (
        "--enable-confidence-head" in dspark
        and "--confidence-head-with-markov" in dspark
    )
    for cmd in (forespark, dspark):  # our entry registers what the trainer lacks
        assert cmd[cmd.index("-m") + 1] == "forespark.train.entry"
    assert (
        value(dspark, "--block-size") == "15"
        and value(dspark, "--max-anchors") == "3072"
    )
    for cmd in (forespark, dspark):  # bidirectional draft blocks, DSpark included
        assert "--sliding-window-non-causal" in cmd
    # each method at its RedHatAI card's rate
    assert value(forespark, "--lr") == "6e-4"
    assert value(dspark, "--lr") == "3e-4"


def test_passthrough_replaces_the_default_and_attention_is_fixed():
    cmd = build("forespark", "--lr", "1e-4")
    assert cmd.count("--lr") == 1 and value(cmd, "--lr") == "1e-4"
    with pytest.raises(SystemExit, match="fixes the drafter's attention"):
        build("forespark", "--sliding-window-non-causal")


def test_resume_refuses_a_different_contract(tmp_path):
    saved = tmp_path / "checkpoints" / "3"
    saved.mkdir(parents=True)
    (saved / "config.json").write_text(
        json.dumps(
            {"sliding_window_non_causal": True, "enable_fore_d_predictor": False}
        )
    )
    # DSpark has no Fore-D contract: fine
    launch.check_resume(tmp_path / "checkpoints", "dspark", 8)
    with pytest.raises(
        SystemExit, match="cannot resume"
    ):  # a plain non-causal drafter lacks the Fore-D head
        launch.check_resume(tmp_path / "checkpoints", "forespark", 8)
    (saved / "config.json").write_text(
        json.dumps(
            {"sliding_window_non_causal": False, "enable_fore_d_predictor": True}
        )
    )
    with pytest.raises(
        SystemExit, match="cannot resume"
    ):  # ... and a causal one the width-conditioned head's attention
        launch.check_resume(tmp_path / "checkpoints", "forespark", 8)


def test_k_moves_block_size_decay_and_run_name_together(tmp_path):
    cmd = build("forespark")
    # The DFlash paper's block-16 recipe at the targets' k = 15.
    assert value(cmd, "--block-size") == "16"
    assert value(cmd, "--dflash-decay-gamma") == "7"
    assert value(cmd, "--save-path").endswith("-forespark-k15/checkpoints")
    dspark = build("dspark")  # equal draft length
    assert value(dspark, "--block-size") == "15"
    assert value(dspark, "--dflash-decay-gamma") == "7"
    short = build("forespark", "--k", "7")
    assert (value(short, "--block-size"), value(short, "--dflash-decay-gamma")) == (
        "8",
        "4",
    )
    assert value(short, "--save-path").endswith("-forespark-k7/checkpoints")
    saved = tmp_path / "checkpoints" / "1"
    saved.mkdir(parents=True)
    (saved / "config.json").write_text(json.dumps({"block_size": 8}))
    launch.check_resume(tmp_path / "checkpoints", "dspark", 8)
    with pytest.raises(SystemExit, match="block size 8, this run trains 16"):
        launch.check_resume(tmp_path / "checkpoints", "forespark", 16)


def test_glm_pins_its_drafter_and_every_method_registers_its_config(tmp_path):
    for method in ("dspark", "forespark"):
        args, passthrough = launch.build_parser().parse_known_args(
            ["glm53flash", method, "--gpus", "0,1", "--out", str(tmp_path / method)]
        )
        cmd = launch.command(args, passthrough)[0]
        # transformers < 5.16 lacks glm5_next: every trainer goes through our entry.
        assert cmd[cmd.index("-m") + 1] == "forespark.train.entry"
        assert "--num-layers" not in cmd
        config = tmp_path / method / "draft_config.json"
        pinned = json.loads(config.read_text())
        assert value(cmd, "--draft-config") == str(config)
        shape = (pinned["model_type"], pinned["hidden_size"], pinned["head_dim"])
        assert shape == ("qwen3", 4096, 128)
        # Qwen3Config keeps sliding_window only with use_sliding_window; vLLM refuses a windowless drafter.
        assert (pinned["sliding_window"], pinned["use_sliding_window"]) == (2048, True)
        start = cmd.index("--target-layer-ids") + 1
        assert cmd[start : start + 5] == ["6", "15", "25", "34", "43"]


def test_glm53_pins_the_published_six_layer_drafter(tmp_path):
    for method in ("forespark", "dspark"):
        argv = ["glm53", method, "--gpus", "0,1", "--out", str(tmp_path / method)]
        args, passthrough = launch.build_parser().parse_known_args(argv)
        cmd = launch.command(args, passthrough)[0]
        pinned = json.loads((tmp_path / method / "draft_config.json").read_text())
        assert (pinned["num_hidden_layers"], pinned["hidden_size"]) == (6, 6144)
        start = cmd.index("--target-layer-ids") + 1
        assert cmd[start : start + 6] == ["6", "20", "34", "48", "62", "76"]
        assert "--fsdp-shard" in cmd  # a ~2B-parameter drafter: sharded


def test_regeneration_serves_a_bounded_context(monkeypatch):
    from argparse import Namespace

    from forespark.train import data

    class Launched(Exception):
        pass

    def popen(cmd, **kwargs):
        raise Launched(cmd)

    monkeypatch.setattr(data, "for_gpus", lambda cfg, gpus: cfg)  # no nvidia-smi probe
    pytest.importorskip(
        "torch"
    )  # GLM's jit_env imports it, and a first import runs ldconfig through Popen
    monkeypatch.setattr(data.subprocess, "Popen", popen)
    args = Namespace(model="glm53", gpus="0,1", parallel=None, port=8205)
    args.max_num_seqs, args.seq_len = None, 8192
    with pytest.raises(Launched) as launched, data.target_server(args, "/snapshot"):
        pass
    cmd = launched.value.args[0]
    assert cmd[cmd.index("--max-model-len") + 1] == "40960"  # not GLM-5.3's 1M
    assert cmd[cmd.index("serve") + 1] == "/snapshot"
    assert "--revision" not in cmd  # the snapshot is the pin


def test_regeneration_captures_wide_graphs_only_under_tp(monkeypatch):
    from argparse import Namespace

    from forespark.train import data

    class Launched(Exception):
        pass

    def popen(cmd, **kwargs):
        raise Launched(cmd)

    monkeypatch.setattr(data, "for_gpus", lambda cfg, gpus: cfg)  # no nvidia-smi probe
    pytest.importorskip(
        "torch"
    )  # GLM's jit_env imports it, and a first import runs ldconfig through Popen
    monkeypatch.setattr(data.subprocess, "Popen", popen)

    def command(model, gpus):
        args = Namespace(model=model, gpus=gpus, parallel=None, port=8204)
        args.max_num_seqs, args.seq_len = None, 8192
        with pytest.raises(Launched) as launched, data.target_server(args, "/snapshot"):
            pass
        return launched.value.args[0]

    cmd = command("glm53flash", "0,1,2,3,4,5,6,7")  # tp, max_num_seqs 512, k = 15
    graphs = json.loads(cmd[cmd.index("--compilation-config") + 1])
    sizes = graphs["cudagraph_capture_sizes"]
    assert (min(sizes), max(sizes)) == (1, 512 * 16)
    # dp replicas keep vLLM's sizes
    assert "--compilation-config" not in command("qwen14", "0,1")


def test_pool_drops_benchmark_prompts(monkeypatch, tmp_path):
    from forespark.train import data

    bench = tmp_path / "bench"
    bench.mkdir()
    turn = {"messages": [{"role": "user", "content": "A ShareGPT turn"}]}
    (bench / "sharegpt.jsonl").write_text(json.dumps(turn) + "\n")
    monkeypatch.setattr(data, "BENCH_DATA", bench)
    asked = []

    def source_rows(name, revision, split):
        asked.append(revision)
        if "ultrachat" not in name:
            return []
        # A loose match of the ShareGPT turn is dropped; the other prompt stays.
        prompts = ("a sharegpt   TURN!", "Something new")
        return [{"messages": [{"role": "user", "content": p}, {"role": "assistant", "content": "x"}]}
                for p in prompts]  # fmt: skip

    monkeypatch.setattr(data, "source_rows", source_rows)
    data.build_pool(tmp_path / "pool", None, exclude=True)
    rows = (tmp_path / "pool" / "train.jsonl").read_text().splitlines()
    kept = [json.loads(r)["conversations"][0]["content"] for r in rows]
    assert kept == ["Something new"]
    assert len(asked) == len(data.SOURCES) and all(asked)  # pinned revisions


def test_training_reads_the_target_at_its_pinned_revision(monkeypatch):
    from forespark import config

    cfg = config.MODELS["glm53flash"]
    assert value(build("forespark"), "--verifier-name-or-path") == (
        "/snapshots/Qwen/Qwen3-14B@40c069824f4251a91eefaf281ebe4c544efd3e18"
    )
    hub = pytest.importorskip("huggingface_hub")
    fetched = []

    def snapshot_download(repo, **kw):
        fetched.append((repo, kw))
        return "/cache"

    monkeypatch.setattr(hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(config, "revision_cached", lambda repo, revision: True)
    assert config.target_snapshot(cfg) == "/cache"
    repo, kw = fetched[0]
    assert (repo, kw["revision"]) == (cfg["repo"], cfg["revision"])
    assert kw["local_files_only"]  # cached: no Hub call


def test_gamma_follows_block_size():
    assert [launch.gamma_for(b) for b in (8, 10, 16)] == [4, 5, 7]

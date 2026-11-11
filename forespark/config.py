# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Targets, drafting methods and runtime switches -- defined once.

Training, serving and benchmarking all read these tables, so an arm cannot
silently train one way and serve another.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from .paths import ROOT

MODELS: dict[str, dict] = yaml.safe_load((ROOT / "configs/models.yaml").read_text())


def model_config(name: str, k: int | None = None) -> dict:
    """A target's entry with its draft length k overridden (--k). Block sizes, run
    names, serving sizes and token budgets all follow k, so one flag moves them
    together (passing --block-size to the trainer alone would not)."""
    cfg = MODELS[name]
    if k is None or k == cfg["k"]:
        return cfg
    if k < 1:
        raise SystemExit(f"--k {k}: the draft length must be at least 1")
    return {**cfg, "k": k}


def checkpoint_k(draft: str | Path | None) -> int | None:
    """The draft length a local speculators checkpoint was trained for (its
    config's first proposal method), or None for a Hub id or an unrecorded one."""
    path = Path(draft) / "config.json" if draft else None
    if path is None or not path.is_file():
        return None
    spec = json.loads(path.read_text()).get("speculators_config") or {}
    return (spec.get("proposal_methods") or [{}])[0].get("speculative_tokens")


def for_gpus(cfg: dict, gpus: str) -> dict:
    """A target's config on these GPUs: its `blackwell:` args are appended on
    sm_100 only (vLLM refuses them elsewhere and picks its own fallbacks)."""
    extra = cfg.get("blackwell")
    if not extra or gpu_arch(gpus) != 10:
        return cfg
    return {**cfg, **{key: [*cfg.get(key, []), *args] for key, args in extra.items()}}


def gpu_arch(gpus: str) -> int:
    """Major compute capability of the first listed GPU (9 Hopper, 10 Blackwell datacenter); 0 if unknown."""
    query = [
        "nvidia-smi",
        "-i",
        gpus.split(",")[0],
        "--query-gpu=compute_cap",
        "--format=csv,noheader",
    ]
    try:
        return int(
            float(
                subprocess.run(
                    query, capture_output=True, text=True, check=True, timeout=60
                ).stdout.split()[0]
            )
        )
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return 0


# What a server or trainer reads of a target: config, tokenizer and chat template, weights.
TARGET_FILES = [
    "*.json",
    "*.safetensors",
    "*.jinja",
    "*.txt",
    "*.model",
    "*.tiktoken",
    "tokenizer*",
]


@functools.cache
def revision_cached(repo: str, revision: str | None) -> bool:
    """Whether a pinned revision's config and every weight shard are in the local
    Hugging Face cache. Not snapshot_download's completeness: vLLM fetches only
    what it serves, so a repo's other files are never there."""
    if not revision:
        return False
    from huggingface_hub import try_to_load_from_cache

    config = try_to_load_from_cache(repo, "config.json", revision=revision)
    if not isinstance(config, str):
        return False
    snapshot = Path(config).parent
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        return all(
            (snapshot / shard).is_file()
            for shard in set(json.loads(index.read_text())["weight_map"].values())
        )
    return any(snapshot.glob("*.safetensors"))


def target_snapshot(cfg: dict) -> str:
    """The target at its pinned revision, as a local directory (fetched first if
    missing). Regeneration, tokenization, the hidden-state server and the trainer
    all read it: speculators' trainer and launcher take no revision, and given the
    repo id they read the Hub's main, which moves past the pin."""
    from huggingface_hub import snapshot_download

    revision = cfg.get("revision")
    return snapshot_download(cfg["repo"], revision=revision, allow_patterns=TARGET_FILES,
                             local_files_only=revision_cached(cfg["repo"], revision))  # fmt: skip


CUDA_TOOLKITS = Path("/usr/local")  # cuda-<version>/


def jit_env(cfg: dict) -> dict[str, str]:
    """For targets whose kernels vLLM compiles at startup (`nvcc: true`): CUDA_HOME,
    the caller's own, else the local toolkit matching the CUDA torch was built
    with (not the venv's pip nvcc: its compiler and headers disagree on version),
    and the venv's bin last on PATH, for the `ninja` FlashInfer's JIT runs."""
    if not cfg.get("nvcc"):
        return {}
    path = os.pathsep.join([os.environ.get("PATH", ""), str(ROOT / ".venv/bin")])
    env = {"PATH": path}
    if os.environ.get("CUDA_HOME"):
        return env
    import torch

    home = CUDA_TOOLKITS / f"cuda-{torch.version.cuda}"
    if not (home / "bin/nvcc").exists():
        raise SystemExit(
            f"{cfg['repo']} compiles kernels with nvcc at startup: set CUDA_HOME to a CUDA {torch.version.cuda} toolkit"
        )
    return {**env, "CUDA_HOME": str(home)}


# The architecture ForeSpark's checkpoints declare. vLLM's speculator dispatch
# keys on it (see patches/vllm.patch), so it must not change.
DFLASH2_CONF_ARCHITECTURE = "DFlash2ConfDraftModel"


@dataclass(frozen=True)
class Method:
    name: str
    speculator_type: str  # speculators --speculator-type
    serve_method: str  # vLLM speculative_config["method"]
    samples_anchor: (
        bool  # dspark predicts from the anchor; dflash treats it as the bonus
    )
    fore_d: bool  # Fore-D: per-request draft widths and a joint verify budget, planned from the Fore-D head

    def block_size(self, k: int) -> int:
        """Equal draft length k for both methods: dflash also feeds the anchor."""
        return k if self.samples_anchor else k + 1


METHODS: dict[str, Method] = {
    m.name: m
    for m in (
        # The baseline: DSpark (Markov and confidence heads), served with adaptive verification.
        Method("dspark", "dspark", "dspark", True, False),
        # DFlash2 with confidence and width-conditioned Fore-D heads, under Fore-D.
        Method("forespark", "dflash2_conf", "dflash", False, True),
    )
}

# Environment variables are the only channel from the benchmark launcher into
# vLLM's worker processes; every one the package reads is named here.
STS_ENV = "FORESPARK_STS"  # per-position confidence temperatures (calibrate_sts.py fit)
STS_COLLECT_ENV = "FORESPARK_STS_COLLECT"  # directory for calibrate_sts.py collect
CPU_RANKING_ENV = (
    "FORESPARK_CPU_RANKING"  # host-side verification ranking (models.yaml cpu_ranking)
)
DECODE_BARRIER_ENV = "FORESPARK_DECODE_BARRIER_DIR"  # exact decode-start barrier
TP_CONSISTENT_VERIFY_ENV = (
    "FORESPARK_TP_CONSISTENT_VERIFY"  # rank 0's confidences and drafts on every TP rank
)
POOL_ENV = "FORESPARK_POOL"  # prompt pool dataset for speculators data generation
STEP_TIMING_FILE_ENV = "VLLM_STEP_TIMING_FILE"  # vLLM's sampled step timer (patches/vllm.patch) writes here
GC_THRESHOLD_ENV = (
    "FORESPARK_GC_THRESHOLD"  # "g0,g1,g2": thresholds in every vLLM process
)


def _optional_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


@dataclass(frozen=True)
class ForeDConfig:
    """Switches for Fore-D, set by the launcher."""

    enabled: bool = False
    fore_d_calibration: Path | None = None  # calibrate_fore_d.py sidecar
    costs: Path | None = None  # calibrate_costs.py table: measured step costs
    fixed_width: int | None = (
        None  # calibration: every request drafts this many, all verified
    )
    log: Path | None = (
        None  # rank-0 Fore-D/confidence/acceptance log (for calibrate_fore_d.py)
    )

    ENV = {
        "enabled": "FORESPARK_FORE_D",
        "fore_d_calibration": "FORESPARK_FORE_D_CALIBRATION",
        "costs": "FORESPARK_FORE_D_COSTS",
        "fixed_width": "FORESPARK_FORE_D_FIXED_WIDTH",
        "log": "FORESPARK_FORE_D_LOG",
    }

    @classmethod
    def from_env(cls) -> ForeDConfig:
        env = cls.ENV
        fixed = os.environ.get(env["fixed_width"])
        return cls(
            enabled=os.environ.get(env["enabled"], "0") == "1",
            fore_d_calibration=_optional_path(env["fore_d_calibration"]),
            costs=_optional_path(env["costs"]),
            fixed_width=int(fixed) if fixed else None,
            log=_optional_path(env["log"]),
        )

    def to_env(self) -> dict[str, str]:
        values = {
            "enabled": "1" if self.enabled else None,
            "fore_d_calibration": self.fore_d_calibration,
            "costs": self.costs,
            "fixed_width": self.fixed_width,
            "log": self.log,
        }
        return {self.ENV[k]: str(v) for k, v in values.items() if v is not None}

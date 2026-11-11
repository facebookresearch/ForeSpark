# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Teach vLLM to load ForeSpark's `dflash2_conf` speculators checkpoints.

Two public registries: the speculators-config translation (checkpoint
config.json -> the hf_config vLLM builds the drafter from) and the model
registry (architecture name -> serving class, resolved lazily in the worker).
"""

import os

from ..config import DFLASH2_CONF_ARCHITECTURE, ForeDConfig

SPECULATORS_MODEL_TYPE = "dflash2_conf"


def _translate(config_dict: dict, pre_trained_config: dict) -> None:
    """DFlash2's translation, plus our architecture and heads."""
    from vllm.transformers_utils.configs.speculators.algos import update_dflash2

    update_dflash2(config_dict, pre_trained_config)
    pre_trained_config["architectures"] = [DFLASH2_CONF_ARCHITECTURE]
    causal = not config_dict.get("sliding_window_non_causal", True)
    # vLLM reads the top-level flag first; keep both locations consistent.
    pre_trained_config["is_causal"] = causal
    draft = pre_trained_config["dflash_config"]
    draft["causal"] = causal
    draft["enable_confidence_head"] = bool(
        config_dict.get("enable_confidence_head", True)
    )
    # The Fore-D head (one row per width: the drafter is non-causal, so a draft
    # changes with its width). Only Fore-D reads it. Fixed-k serving
    # (STS collection) leaves it out: it may serve fewer speculative tokens than
    # the head was trained for, which it would not fit.
    fore_d = bool(config_dict.get("enable_fore_d_predictor", False))
    draft["enable_fore_d_predictor"] = (
        fore_d and os.environ.get(ForeDConfig.ENV["enabled"]) == "1"
    )
    draft["fore_d_rank"] = int(config_dict.get("fore_d_rank", 256))


def register_dflash2_conf() -> None:
    from vllm import ModelRegistry
    from vllm.transformers_utils.configs.speculators.algos import (
        register_speculator,
        SUPPORTED_SPECULATORS_TYPES,
    )

    if SPECULATORS_MODEL_TYPE not in SUPPORTED_SPECULATORS_TYPES:
        register_speculator(SPECULATORS_MODEL_TYPE)(_translate)
    ModelRegistry.register_model(
        DFLASH2_CONF_ARCHITECTURE,
        "forespark.models.vllm_model:DFlash2ConfForCausalLM",
    )

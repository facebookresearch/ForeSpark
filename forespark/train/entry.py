# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Run a speculators script (the trainer, the hidden-state launcher) with our registrations.

    torchrun ... -m forespark.train.entry <speculators/scripts/train.py> <trainer args>
    python -m forespark.train.entry <speculators/scripts/launch_vllm.py> <launcher args>

The trainer resolves --speculator-type against a registry that only the
speculators package's own imports fill, so a model defined here is invisible
to it until something imports it first. The same holds for target configs the
pinned transformers lacks (speculators caps it below the release that added
GLM-5.3-Flash's `glm5_next`): vLLM ships them, so register those too.
"""

import runpy
import sys


def register_target_configs() -> None:
    from transformers import AutoConfig
    from vllm.transformers_utils.configs.glm5_next import (
        Glm5NextConfig,
        Glm5NextTextConfig,
        Glm5NextVisionConfig,
    )

    for config in (Glm5NextConfig, Glm5NextTextConfig, Glm5NextVisionConfig):
        try:
            AutoConfig.register(config.model_type, config)
        except ValueError:  # a transformers that already knows it
            pass


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} <speculators script> [args...]")
    from ..models import draft_model  # noqa: F401  registers dflash2_conf

    register_target_configs()

    script = sys.argv[1]
    sys.argv = [script, *sys.argv[2:]]  # what the trainer would see launched directly
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()

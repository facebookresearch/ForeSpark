# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""ForeSpark: a DFlash2 drafter with confidence and Fore-D heads, Fore-D scheduling, and benchmarks.

`register` is the `vllm.general_plugins` entry point, which vLLM calls in every
process it starts (API server, engine core, workers).
"""

import gc
import os

_registered = False


def apply_gc_threshold() -> None:
    """Collector thresholds from the environment. The engine core's default full
    collections (~200 ms at 512 requests) stall every step behind them."""
    from .config import GC_THRESHOLD_ENV

    if threshold := os.environ.get(GC_THRESHOLD_ENV):
        gc.set_threshold(*(int(x) for x in threshold.split(",")))


def register() -> None:
    """Teach this vLLM process about `dflash2_conf` and install opt-in hooks.

    Import-light: model classes are registered by name and resolved in the
    worker that builds the drafter.
    """
    global _registered
    if _registered:
        return
    _registered = True

    from .config import (
        CPU_RANKING_ENV,
        DECODE_BARRIER_ENV,
        STS_COLLECT_ENV,
        STS_ENV,
        TP_CONSISTENT_VERIFY_ENV,
    )  # fmt: skip
    from .models.registry import register_dflash2_conf

    apply_gc_threshold()
    if os.environ.get(STS_ENV) and os.environ.get(STS_COLLECT_ENV):
        raise ValueError(
            f"{STS_ENV} and {STS_COLLECT_ENV} are exclusive: collection needs "
            "uncalibrated confidence on untrimmed blocks."
        )
    register_dflash2_conf()
    if os.environ.get(STS_ENV) or os.environ.get(STS_COLLECT_ENV):
        from .serving import sts

        sts.install()
    if os.environ.get(CPU_RANKING_ENV):
        from .serving import cpu_ranking

        cpu_ranking.install()
    if os.environ.get(DECODE_BARRIER_ENV):
        from .serving import decode_barrier

        decode_barrier.install()
    if os.environ.get(TP_CONSISTENT_VERIFY_ENV):
        from .serving import tp_consistency

        tp_consistency.install()

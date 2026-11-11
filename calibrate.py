#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Calibrations, one subcommand each (scripts/calibration/calibrate.sh runs them all).

    ./calibrate.py sts     prompts|collect|fit ...          # confidence temperatures (train/calibrate_sts.py)
    ./calibrate.py fore-d LOG_DIR ... --out fore_d.json   # Fore-D calibration (train/calibrate_fore_d.py)
    ./calibrate.py costs   RUN.steps.bin ... --out costs.json  # the planner's measured step costs

fore-d reads Fore-D's step logs and costs their step timer:
eval.py ... --method forespark --drafter-width D --fore-d-log LOG_DIR --step-timing.
"""

import importlib
import sys

COMMANDS = {
    "sts": "forespark.train.calibrate_sts",
    "fore-d": "forespark.train.calibrate_fore_d",
    "costs": "forespark.train.calibrate_costs",
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(__doc__)
    module = importlib.import_module(COMMANDS[sys.argv[1]])
    sys.argv = [f"calibrate.py {sys.argv[1]}", *sys.argv[2:]]
    sys.exit(module.main())

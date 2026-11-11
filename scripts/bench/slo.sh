#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# ShareGPT under an SLO (scripts/bench/common.sh): open loop, Poisson arrivals at each request rate (scripts/env.sh),
# every measured request scored against MODEL's TTFT and TPOT objectives.
#   scripts/bench/slo.sh qwen14
set -uo pipefail
# shellcheck disable=SC1091  # common.sh sits beside this script
source "$(dirname "$0")/common.sh" "${1:-}"
run_cells sharegpt-rate --rates "${RATES[@]}" "${SLO[@]}"

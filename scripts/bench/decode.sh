#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# ShareGPT-Decode (scripts/bench/common.sh): fixed batches released together once every request has its first
# token, each request timed over the same 512-token span of its answer, on the screened prompts, at MODEL's decode
# batch sizes (scripts/env.sh).
#   scripts/bench/decode.sh qwen14
set -uo pipefail
# shellcheck disable=SC1091  # common.sh sits beside this script
source "$(dirname "$0")/common.sh" "${1:-}"
[ -s "$CAL/decode-prompts.json" ] || { echo "missing $CAL/decode-prompts.json: run scripts/calibration/calibrate.sh $MODEL" >&2; exit 1; }
run_cells sharegpt-decode --concurrency "${DECODE_CONCURRENCY[@]}"

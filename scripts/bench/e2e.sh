#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# ShareGPT end to end (scripts/bench/common.sh): closed loop at each concurrency (scripts/env.sh), prefill and
# decode sharing batches, the prefix cache on; a 120 s window of at least 128 requests, drained under load.
#   scripts/bench/e2e.sh qwen14
set -uo pipefail
# shellcheck disable=SC1091  # common.sh sits beside this script
source "$(dirname "$0")/common.sh" "${1:-}"
run_cells sharegpt --concurrency "${CONCURRENCY[@]}" --min-window-requests 128

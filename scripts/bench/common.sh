# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# shellcheck shell=bash
# Sourced by decode.sh, e2e.sh and slo.sh: one TP8 server per cell (arm x phase), one arm at a time, every cell
# with the step timer's draft/verify breakdown (--step-timing; its overhead measured negligible), greedy, into
# <results>/MODEL/bench/PHASE-ARM.json. Inputs only from scripts/calibration/calibrate.sh. A result that exists
# is skipped, so a rerun resumes.
#   ARMS="nospec forespark" scripts/bench/e2e.sh qwen14   # a subset of nospec dspark forespark
#   DRY=1 scripts/bench/decode.sh glm53                   # print the commands, run nothing
# BENCH_ARGS adds eval.py flags after the script's, so they win: BENCH_ARGS="--concurrency 8 64" runs two points.
# shellcheck disable=SC1091  # env.sh sits one level up
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh" "${1:-}"
[ "${#SERVE_ENV[@]}" -eq 0 ] || export "${SERVE_ENV[@]}"
read -ra ARM_LIST <<< "${ARMS:-nospec dspark forespark}"
read -ra BENCH_ARGS <<< "${BENCH_ARGS:-}"
OUT=$RES/bench
mkdir -p "$OUT/logs"
check_inputs() {  # the calibration files the arms need
    local needed=()
    for a in "${ARM_LIST[@]}"; do
        case $a in
            dspark) needed+=("$CAL/sts-dspark/sts.json") ;;
            forespark) needed+=("$CAL/sts-forespark/sts.json" "$CAL/fore_d/fore_d.json" "$COSTS") ;;
        esac
    done
    for f in "${needed[@]}"; do
        [ -s "$f" ] || { echo "missing $f: run scripts/calibration/calibrate.sh $MODEL" >&2; exit 1; }
    done
}
run_cells() {  # run_cells PHASE FLAG...: the phase for every arm, with these bench flags
    local phase=$1 arm_args file; shift
    check_inputs
    echo "start $phase $(date +%T)"
    for a in "${ARM_LIST[@]}"; do
        file=$OUT/$phase-$a.json
        [ -s "$file" ] && { echo "=== $phase $a done already"; continue; }
        mapfile -t arm_args < <(arm "$a")
        local cmd=("$PY" eval.py "$phase" "$MODEL" "${arm_args[@]}" "$@" "${SERVE[@]}" --step-timing
                   --arm-label "$a" --round 1 --gpus "$TP8" --port 8750 --out "$file" "${BENCH_ARGS[@]}")
        [ "${DRY:-0}" = 1 ] && { echo "${cmd[*]}"; continue; }
        "${cmd[@]}" > "$OUT/logs/$phase-$a.log" 2>&1
        echo "=== $phase $a exit $? $(date +%T)"
    done
    [ "${DRY:-0}" = 1 ] || "$PY" eval.py report --model "$MODEL" --results-root "$OUT" --phase "$phase"
}

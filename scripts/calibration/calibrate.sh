#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Every calibration MODEL's benchmarks read (scripts/env.sh), into <results>/MODEL/calibration, all on held-out
# Alpaca prompts (in neither the drafter corpus nor any benchmark):
#   sts-{dspark,forespark}/sts.json   confidence temperatures, from fixed-k serving with every draft verified
#                                     (collect.json beside each: the drafter's acceptance length at fixed k)
#   decode-prompts.json               ShareGPT prompts whose answer outruns the decode span (the target alone)
#   fore_d/fore_d.json                the Fore-D head's calibration, from forced-width logs (widths 1..K, every
#                                     draft verified: a draft changes with its width, so each width is its own run)
#   fore_d/costs.json                 Fore-D's measured verify costs, from the same runs' step timer
#   fore_d/costs-ext.json             qwen14: the same past the 256 requests those reach, from decode-only runs
# One server at a time with the benchmark's settings ($SERVE). A step whose output exists is skipped, so a rerun
# resumes.
#   scripts/calibration/calibrate.sh glm53flash
# STS_ARGS adds `calibrate.py sts collect` flags, BENCH_ARGS the forced-width runs' eval.py flags, and WIDTHS picks
# the forced widths (default 1..K); a smoke test of the pipeline in a scratch tree:
#   FORESPARK_RESULTS=/scratch/results STS_ARGS="--num-prompts 64" WIDTHS="1 8 15" \
#       BENCH_ARGS="--concurrency 8 64 --warmup-seconds 10 --measure-seconds 20 --drain-seconds 10" \
#       scripts/calibration/calibrate.sh qwen14
set -uo pipefail
# shellcheck disable=SC1091  # env.sh sits one level up
source "$(dirname "$0")/../env.sh" "${1:-}"
[ "${#SERVE_ENV[@]}" -eq 0 ] || export "${SERVE_ENV[@]}"
read -ra STS_ARGS <<< "${STS_ARGS:-}"
read -ra BENCH_ARGS <<< "${BENCH_ARGS:-}"
read -ra WIDTH_LIST <<< "${WIDTHS:-$(seq -s ' ' 1 "$K")}"
mkdir -p "$CAL/fore_d" "$CAL/logs"
ALPACA=$CAL/alpaca-4000.jsonl
have() { [ -s "$1" ]; }
log() { echo "=== $1 exit $2 $(date +%T)"; }
sts() {  # sts NAME DRAFT [GPU]: collect at fixed k (eager), then fit; TP8 without a GPU
    have "$CAL/sts-$1/sts.json" && return
    local gpus=$TP8 port=8669
    [ -n "${3:-}" ] && gpus=$3 port=$((8660 + $3))
    $PY calibrate.py sts collect "$MODEL" --draft "$2" --method "$1" --out "$CAL/sts-$1" --gpus "$gpus" \
        --port "$port" "${DRAFT_ATTN[@]}" "${STS_ARGS[@]}" \
        && $PY calibrate.py sts fit --shards "$CAL/sts-$1/raw/sts.*.npz" --out "$CAL/sts-$1/sts.json"
}
echo "start $(date +%T)"
have "$ALPACA" || $PY calibrate.py sts prompts --out "$ALPACA" --num-prompts 4000
if [ "${#STS_GPUS[@]}" -gt 0 ]; then
    { sts dspark "$DSPARK" "${STS_GPUS[0]}" > "$CAL/logs/sts-dspark.log" 2>&1; log sts-dspark $?; } &
    { sts forespark "$FORESPARK" "${STS_GPUS[1]}" > "$CAL/logs/sts-forespark.log" 2>&1; log sts-forespark $?; } &
    wait
else
    sts dspark "$DSPARK" > "$CAL/logs/sts-dspark.log" 2>&1; log sts-dspark $?
    sts forespark "$FORESPARK" > "$CAL/logs/sts-forespark.log" 2>&1; log sts-forespark $?
fi
if ! have "$CAL/decode-prompts.json"; then
    $PY eval.py decode-prompts "$MODEL" --gpus "$TP8" --port 8670 > "$CAL/logs/decode-prompts.log" 2>&1
    log decode-prompts $?
fi

# ForeSpark's forced widths (every request drafts w, every draft verified), then the Fore-D and cost fits.
for w in "${WIDTH_LIST[@]}"; do
    have "$CAL/fore_d/alpaca-log-w$w.json" && continue
    $PY eval.py sharegpt "$MODEL" --prompts "$ALPACA" --prefix-caching off --draft "$FORESPARK" --method forespark \
        --drafter-width "$w" --fore-d-log "$CAL/fore_d/log/w$w" --arm-label "alpaca-log-w$w" "${SERVE[@]}" \
        --concurrency "${CONCURRENCY[@]}" --warmup-seconds 30 --measure-seconds 60 --drain-seconds 60 \
        --step-timing --gpus "$TP8" --port 8660 --out "$CAL/fore_d/alpaca-log-w$w.json" "${BENCH_ARGS[@]}" \
        > "$CAL/logs/fore_d-w$w.log" 2>&1
    log "forced width $w" $?
done
if ! have "$CAL/fore_d/fore_d.json"; then
    $PY calibrate.py fore-d "$CAL"/fore_d/log/w* --out "$CAL/fore_d/fore_d.json" > "$CAL/logs/fore-d.log" 2>&1
    log fore-d $?
fi
if ! have "$CAL/fore_d/costs.json"; then
    $PY calibrate.py costs "$CAL"/fore_d/alpaca-log-w*.steps.bin --out "$CAL/fore_d/costs.json" > "$CAL/logs/costs.log" 2>&1
    log costs $?
fi

if [ "$EXTEND_COSTS" = 1 ]; then
    # ShareGPT-Decode's batches start at once and stay whole, so every step after their prefill is decode-only at
    # n = C: forced widths at C = 384 and 512, then their levels appended above the table's last. Each run logs its
    # steps: only a log run verifies every forced draft.
    for w in ${WIDTHS:-1 2 3 4 5 6 7 8 10 12 15}; do
        [ "$w" -le "$K" ] || continue
        have "$CAL/fore_d/ext/decode-w$w.json" && continue
        $PY eval.py sharegpt-decode "$MODEL" --draft "$FORESPARK" --method forespark --drafter-width "$w" \
            --fore-d-log "$CAL/fore_d/ext/log/w$w" --arm-label "decode-w$w" "${SERVE[@]}" --concurrency 384 512 \
            --step-timing --gpus "$TP8" --port 8665 --out "$CAL/fore_d/ext/decode-w$w.json" \
            > "$CAL/logs/costs-ext-w$w.log" 2>&1
        log "decode-only forced width $w" $?
    done
    if ! have "$CAL/fore_d/costs-ext.json"; then
        $PY calibrate.py costs "$CAL"/fore_d/ext/decode-w*.steps.bin --extend "$CAL/fore_d/costs.json" \
            --out "$CAL/fore_d/costs-ext.json" > "$CAL/logs/costs-ext.log" 2>&1
        log costs-ext $?
    fi
fi
echo "=== done $(date +%T)"

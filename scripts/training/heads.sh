#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# ForeSpark's two estimators fitted on a trained drafter, the drafter and the target frozen (heads.py), on one 8-GPU
# node:
#   1. extraction  the drafter, teacher-forced over corpus batches, saves both heads' inputs and labels: 500 held-out
#                  batches to score on, 16,000 training batches to fit on. The target's hidden states come from a
#                  server on SERVE_GPUS; extraction runs one process per GPU of TRAIN_GPUS
#                                                                  -> <runs>/MODEL-forespark-heads-k15/{eval,fit}/
#   2. fit         both heads on the training records, scored beside the drafter's own on the held-out ones
#                                                                  -> <runs>/MODEL-forespark-heads-k15/heads/report.md
#   3. merge       the drafter with the fitted heads                -> <runs>/MODEL-forespark-heads-k15/checkpoint/
# The drafter is FORESPARK (scripts/env.sh), the corpus train.sh's (steps 1-4). Extraction resumes from the parts it
# saved, on the same GPUs. The merged drafter calibrates and benchmarks as any other, in a results tree of its own:
#   scripts/training/heads.sh glm53flash
#   FORESPARK=<runs>/glm53flash-forespark-heads-k15/checkpoint FORESPARK_RESULTS=/scratch/results \
#       scripts/calibration/calibrate.sh glm53flash
# EXTRACT_ARGS adds `heads.py extract` flags and FIT_ARGS `heads.py fit` flags; a smoke test in scratch storage:
#   FORESPARK_RUNS=/scratch/runs EXTRACT_ARGS="--batches 0:8" FIT_ARGS="--steps 20" scripts/training/heads.sh qwen14
set -euo pipefail
# shellcheck disable=SC1091  # env.sh sits one level up
source "$(dirname "$0")/../env.sh" "${1:-}"
[ "${#SERVE_ENV[@]}" -eq 0 ] || export "${SERVE_ENV[@]}"
read -ra EXTRACT_ARGS <<< "${EXTRACT_ARGS:-}"
read -ra FIT_ARGS <<< "${FIT_ARGS:-}"
IFS=, read -ra GPUS <<< "$TRAIN_GPUS"
OUT=$RUNS/$MODEL-forespark-heads-k$K
PORT=$("$PY" -c "from forespark.config import MODELS; print(MODELS['$MODEL']['port'] + 100)")
step() { echo "=== $MODEL $1 $(date +%T)"; }
extract() {  # extract OUT FLAGS...: one shard per GPU
    local out=$1 pids=() i
    shift
    for i in "${!GPUS[@]}"; do
        CUDA_VISIBLE_DEVICES=${GPUS[$i]} "$PY" heads.py extract "$MODEL" "$FORESPARK" --out "$out" \
            --shard "$i/${#GPUS[@]}" "$@" "${EXTRACT_ARGS[@]}" &
        pids+=("$!")
    done
    for i in "${pids[@]}"; do wait "$i"; done
}

step "1. extraction"
"$PY" serve.py "$MODEL" --gpus "$SERVE_GPUS" &
server=$!
trap 'kill "$server" 2>/dev/null || true' EXIT  # under set -e a failed kill would become the exit status
until curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do
    kill -0 "$server" 2>/dev/null || { echo "the hidden-state server exited" >&2; exit 1; }
    sleep 10
done
extract "$OUT/eval" --split val --batches 0:500
extract "$OUT/fit" --split train --batches 0:16000 --anchors 160 --confidence-anchors 16
kill "$server"
wait "$server" || true

step "2. fit"
CUDA_VISIBLE_DEVICES=${GPUS[0]} "$PY" heads.py fit "$MODEL" "$FORESPARK" "$OUT/fit" --eval "$OUT/eval" \
    --out "$OUT/heads" "${FIT_ARGS[@]}"
step "3. merge"
"$PY" heads.py merge "$FORESPARK" "$OUT/heads/heads-fit@all.safetensors" --out "$OUT/checkpoint"
step finished

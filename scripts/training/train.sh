#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# MODEL's whole training procedure (scripts/env.sh) on one 8-GPU node, from the venv alone:
#   1. benchmark prompts  the ShareGPT stream (eval.py prepare), kept out of the corpus  -> <data>/bench/
#   2. prompt pool        Magpie + UltraChat prompts, the benchmark's removed            -> <data>/pool/train.jsonl
#   3. regeneration       every pool prompt answered by the target itself, at TP8        -> <data>/regenerated/MODEL.jsonl
#   4. tokenization       the regenerated conversations, tokenized for the trainer       -> <data>/prepared/MODEL/
#   5. drafters           DSpark, then ForeSpark, each fetching the target's hidden states from a server on SERVE_GPUS
#                                                                                         -> <runs>/MODEL-{dspark,forespark}-k15/
# Models and datasets download from Hugging Face. Every step resumes: steps 1, 2 and 4 reuse what they wrote,
# regeneration skips the prompts already answered, and a trainer resumes from its last checkpoint.
#   scripts/training/train.sh qwen14
#   METHODS=forespark scripts/training/train.sh glm53flash     # one drafter
# DATA_ARGS adds data.py flags and TRAIN_ARGS train.py flags; a smoke test of the procedure in scratch storage
# (configs/paths.yaml), a 2,000-prompt pool (its 5% validation split still feeds every trainer rank) and 20 steps
# per drafter:
#   FORESPARK_DATA=/scratch/data FORESPARK_RUNS=/scratch/runs DATA_ARGS="--limit 2000" TRAIN_ARGS="--max-steps 20" \
#       scripts/training/train.sh qwen14
# Follow a run with ./curves.py <runs>/MODEL-forespark-k15 --watch.
set -euo pipefail
# shellcheck disable=SC1091  # env.sh sits one level up
source "$(dirname "$0")/../env.sh" "${1:-}"
[ "${#SERVE_ENV[@]}" -eq 0 ] || export "${SERVE_ENV[@]}"
read -ra METHOD_LIST <<< "${METHODS:-dspark forespark}"
read -ra DATA_ARGS <<< "${DATA_ARGS:-}"
PORT=$("$PY" -c "from forespark.config import MODELS; print(MODELS['$MODEL']['port'] + 100)")
step() { echo "=== $MODEL $1 $(date +%T)"; }

step "1. benchmark prompts"
"$PY" eval.py prepare
step "2. prompt pool"
"$PY" data.py "$MODEL" --stage pool "${DATA_ARGS[@]}"
step "3. regeneration"
"$PY" data.py "$MODEL" --stage regenerate --gpus "$TP8" "${DATA_ARGS[@]}"
step "4. tokenization"
"$PY" data.py "$MODEL" --stage prepare "${DATA_ARGS[@]}"

for method in "${METHOD_LIST[@]}"; do
    step "5. $method"
    "$PY" serve.py "$MODEL" --gpus "$SERVE_GPUS" &
    server=$!
    trap 'kill "$server" 2>/dev/null || true' EXIT  # under set -e a failed kill would become the exit status
    until curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; do
        kill -0 "$server" 2>/dev/null || { echo "the hidden-state server exited" >&2; exit 1; }
        sleep 10
    done
    env "${TRAIN_ENV[@]}" "$PY" train.py "$MODEL" "$method" --gpus "$TRAIN_GPUS" "${TRAIN_ARGS[@]}"
    kill "$server"
    wait "$server" || true
done
step finished

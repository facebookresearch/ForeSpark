# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# shellcheck shell=bash
# shellcheck disable=SC2034  # read by the scripts that source this file
#
# Per-model settings for scripts/training, scripts/calibration and scripts/bench, sourced as
# `source scripts/env.sh MODEL`, MODEL one of:
#   qwen14       Qwen3-14B on 8x H100
#   glm53flash   GLM-5.3-Flash (NVFP4 experts) on 8x B200
#   glm53        GLM-5.3 (NVFP4 experts) on 8x B200
# Each model's entry in configs/models.yaml is both the training target and the benchmark target. Results go to one
# tree per model, <results>/MODEL/{calibration,bench} (configs/paths.yaml). The benchmark grids and SLOs are the
# paper's. Every drafter path can be overridden from the environment, and TRAIN_ARGS adds train.py flags.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
MODEL=${1:?usage: $0 qwen14|glm53flash|glm53}
PY=.venv/bin/python
TP8=0,1,2,3,4,5,6,7
# Every server's settings beyond eval.py's defaults, the same for every arm (AR included); DRAFT_ATTN also goes to
# the STS collection servers.
DRAFT_ATTN=()
SERVE=()
# Extra environment for every server: regeneration, hidden states, calibration and benchmarks.
SERVE_ENV=()
read -ra TRAIN_ARGS <<< "${TRAIN_ARGS:-}"
TRAIN_ENV=()
case $MODEL in
    qwen14)
        # Hidden-state server on 2 GPUs (two replicas), trainer on 6: the trainer is drafter-bound.
        SERVE_GPUS=0,1 TRAIN_GPUS=2,3,4,5,6,7
        # STS collects at TP1, one drafter per GPU (acceptance statistics, which TP does not move).
        STS_GPUS=(6 7)
        # models.yaml's 512 cap and the budget and graphs that follow from it: 3 x 512 x 16 = 24,576 tokens.
        CONCURRENCY=(1 8 32 64 128 256 512)
        DECODE_CONCURRENCY=(1 8 32 64 128 256 512)
        RATES=(4 8 13 18 23 28 33 38 43 48 53)
        SLO=(--slo-ttft-ms 1000 --slo-tpot-ms 50)
        # On held-out Alpaca's short answers nearly every step at C = 512 takes new requests' prefill, so the
        # measured cost table stops at 256 requests: decode-only forced widths extend it (calibrate.sh).
        EXTEND_COSTS=1
        ;;
    glm53flash|glm53)
        # Hidden-state server at TP4 + EP, trainer on the other four GPUs.
        SERVE_GPUS=0,1,2,3 TRAIN_GPUS=4,5,6,7
        TRAIN_ARGS=(--num-workers 8 "${TRAIN_ARGS[@]}")
        TRAIN_ENV=(PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)
        STS_GPUS=()  # the target needs every GPU: STS collects at TP8, one drafter at a time
        # The all-reduce settings every GLM measurement ran with: NVLink multicast hung on one of our nodes.
        SERVE_ENV=(VLLM_ALLREDUCE_USE_SYMM_MEM=0 TORCH_SYMM_MEM_DISABLE_MULTICAST=1 VLLM_ALLREDUCE_USE_FLASHINFER=0
                   VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm)
        # The drafter runs TRITON_ATTN: its B200 default, FA4, rejects the FP8 KV cache's scales, and Fore-D's
        # packed draft rows need varlen CUDA graphs. At most 256 requests: GLM-5.3-Flash keeps 1 + K linear-attention
        # states per speculative request, so no arm holds more than ~230 at K = 15, and wider graphs only take KV
        # cache. The token budget stays the 512 cap's, 3 x 512 x (K + 1): halved, vLLM's slot-mapping kernel overran
        # its buffer. GLM-5.3 serves at 0.85 of memory: at 0.95 FlashInfer's MoE workspace ran out.
        DRAFT_ATTN=(--draft-attention-backend TRITON_ATTN)
        SERVE=("${DRAFT_ATTN[@]}" --max-num-seqs 256 --max-num-batched-tokens 24576)
        CONCURRENCY=(1 8 32 64 128 256)
        if [ "$MODEL" = glm53flash ]; then
            SERVE+=(--gpu-memory-utilization 0.95)
            # A decode batch grows all at once: wider than 192, a batch preempted before its span ended.
            DECODE_CONCURRENCY=(1 8 32 64 128 192)
            RATES=(4 8 10.5 13 15.5 18 20.5 23 25.5 28 30.5)
            SLO=(--slo-ttft-ms 1000 --slo-tpot-ms 30)
        else
            SERVE+=(--gpu-memory-utilization 0.85)
            DECODE_CONCURRENCY=(1 8 32 64 128 256)
            RATES=(2.5 5 7 9 11 13 15 16 17 19 21)
            # TTFT 1 s; TPOT 5.5x AR's median at C = 1, 7.23 ms -> 40 ms
            SLO=(--slo-ttft-ms 1000 --slo-tpot-ms 40)
        fi
        EXTEND_COSTS=0
        ;;
    *) echo "unknown MODEL $MODEL: qwen14, glm53flash or glm53" >&2; exit 1 ;;
esac
RESULTS_ROOT=$($PY -c "from forespark.paths import RESULTS; print(RESULTS)")
RUNS=$($PY -c "from forespark.paths import RUNS; print(RUNS)")
K=$($PY -c "from forespark.config import MODELS; print(MODELS['$MODEL']['k'])")
# Not overridable alone: eval.py keeps the decode-prompt screening under <results>/MODEL/calibration. A scratch
# tree moves the whole root instead (FORESPARK_RESULTS, configs/paths.yaml).
RES=$RESULTS_ROOT/$MODEL
CAL=$RES/calibration
DSPARK=${DSPARK:-$RUNS/$MODEL-dspark-k$K/checkpoints/checkpoint_best}
FORESPARK=${FORESPARK:-$RUNS/$MODEL-forespark-k$K/checkpoints/checkpoint_best}
COSTS=$CAL/fore_d/costs.json
[ "$EXTEND_COSTS" = 1 ] && COSTS=$CAL/fore_d/costs-ext.json
arm() {  # arm NAME: its eval.py flags, one per line
    case $1 in
        nospec) ;;  # the target alone (AR)
        dspark) printf '%s\n' --draft "$DSPARK" --method dspark --adaptive-verification --sts "$CAL/sts-dspark/sts.json" ;;
        forespark) printf '%s\n' --draft "$FORESPARK" --method forespark --sts "$CAL/sts-forespark/sts.json" \
                       --fore-d-calibration "$CAL/fore_d/fore_d.json" --fore-d-costs "$COSTS" ;;
        *) echo "unknown arm $1: nospec, dspark or forespark" >&2; return 1 ;;
    esac
}

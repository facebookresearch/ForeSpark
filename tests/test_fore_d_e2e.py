# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""End to end on a real target (slow; opt in with FORESPARK_E2E_DRAFT=<a qwen14 ForeSpark checkpoint>).

Greedy decoding under Fore-D must reproduce the target's own
greedy output: speculation only changes how many tokens a step emits, never
which. Each arm runs in its own process (one qwen14 at a time on the GPU), in
the venv setup.sh builds (forespark installed: vLLM loads it as a plugin).
"""

import json
import os
import subprocess
import sys

import pytest

DRAFT = os.environ.get("FORESPARK_E2E_DRAFT")
pytestmark = pytest.mark.skipif(
    not DRAFT,
    reason="set FORESPARK_E2E_DRAFT to a qwen14 ForeSpark checkpoint (dflash2_conf with its Fore-D head)",
)

PROMPTS = [
    "Explain how a hash map handles collisions.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "What causes the seasons on Earth?",
    "List three differences between TCP and UDP.",
]
ARM = r"""
import json, os, sys
arm, draft = sys.argv[1], sys.argv[2]
k = json.load(open(os.path.join(draft, "config.json")))["speculators_config"]["proposal_methods"][0]["speculative_tokens"]
spec = None
if arm != "target":
    os.environ["FORESPARK_FORE_D"] = "1"
    if arm == "forespark-k":  # every request drafts the full width, every draft verified
        os.environ["FORESPARK_FORE_D_FIXED_WIDTH"] = str(k)
    spec = {"method": "dflash", "model": draft, "num_speculative_tokens": k, "enable_adaptive_verification": True}
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen3-14B", revision="40c069824f4251a91eefaf281ebe4c544efd3e18", speculative_config=spec,
          max_model_len=4096, gpu_memory_utilization=0.85, max_num_seqs=64, enable_prefix_caching=False)
prompts = json.loads(sys.argv[3])
outs = llm.chat([[{"role": "user", "content": p}] for p in prompts],
                SamplingParams(temperature=0.0, max_tokens=96), use_tqdm=False,
                chat_template_kwargs={"enable_thinking": False})
print("RESULT" + json.dumps([o.outputs[0].token_ids for o in outs]))
"""


def run(arm: str) -> list[list[int]]:
    env = {**os.environ, "VLLM_ENABLE_V1_MULTIPROCESSING": "0"}
    out = subprocess.run([sys.executable, "-c", ARM, arm, DRAFT, json.dumps(PROMPTS)], env=env,
                         capture_output=True, text=True, timeout=3600)  # fmt: skip
    line = next(
        (row for row in out.stdout.splitlines() if row.startswith("RESULT")), None
    )
    assert line, f"{arm} failed:\n{out.stderr[-4000:]}"
    return json.loads(line[len("RESULT") :])


def agreement(a: list[list[int]], b: list[list[int]]) -> float:
    same = total = 0
    for x, y in zip(a, b, strict=True):
        n = min(len(x), len(y))
        same += next((i for i in range(n) if x[i] != y[i]), n)
        total += n
    return same / total


def test_forespark_greedy_output_matches_the_target():
    target, forespark, forespark_k = run("target"), run("forespark"), run("forespark-k")
    # bf16 kernels differ across batch shapes, so a late near-tie may flip; the
    # common prefix must cover nearly all of every output.
    assert agreement(target, forespark) > 0.9
    assert agreement(target, forespark_k) > 0.9

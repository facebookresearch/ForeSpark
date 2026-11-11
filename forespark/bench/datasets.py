# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark prompts: the ShareGPT stream, and the decode phase's prompt screening.

prepare_sharegpt   multi-turn ShareGPT conversations, one request per user
                   turn carrying its history (that resend is where prefix
                   reuse comes from), interleaved so a turn returns a few
                   hundred requests after its predecessor
"""

import hashlib
import json
import random
from pathlib import Path

from ..paths import BENCH_DATA, RESULTS

SHAREGPT_FILE = BENCH_DATA / "sharegpt.jsonl"
SHAREGPT_SOURCE = (
    "anon8231489123/ShareGPT_Vicuna_unfiltered",
    "ShareGPT_V3_unfiltered_cleaned_split.json",
)
REUSE_GAP = 512  # mean requests between a conversation's consecutive turns


def load_prompts(path: Path) -> list[dict]:
    """Rows from a JSONL file."""
    if not path.is_file():
        raise SystemExit(f"missing {path}; run `./eval.py prepare` first")
    rows = [json.loads(line) for line in path.open() if line.strip()]
    if not rows:
        raise SystemExit(f"no prompts in {path}")
    return rows


def prompt_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def answer_floor(span: int) -> int:
    """Answer tokens a prompt needs for a decode span of `span` tokens: a quarter past it."""
    return span + span // 4 + 1


def screening_file(model: str) -> Path:
    """A per-target input, kept with the target's calibrations."""
    return RESULTS / model / "calibration" / "decode-prompts.json"


def load_screening(
    model: str, revision: str | None, prompts: Path, template_kwargs: dict, span: int
) -> list[int] | None:
    """Prompt indices `eval.py decode-prompts` kept for this target, or None if
    unscreened; a screening of other prompts, weights or template is refused. A
    screening that kept every answer's length serves any span up to its own,
    each taking the answers that outrun it."""
    path = screening_file(model)
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    same = (
        data["prompts_sha256"],
        data["target_revision"],
        data["chat_template_kwargs"],
    )
    if same != (prompt_digest(prompts), revision, template_kwargs):
        raise SystemExit(
            f"{path} screened other prompts, weights or chat template: rerun `eval.py decode-prompts {model}`"
        )
    if span > data["decode_tokens"]:
        raise SystemExit(
            f"{path} screened for {data['decode_tokens']}-token spans: rerun with --decode-tokens {span}"
        )
    floor = answer_floor(span)
    return [index for index, length in zip(data["order"], data["lengths"], strict=True)
            if length is not None and length >= floor]  # fmt: skip


def stop_token_ids(cfg: dict) -> list[int]:
    """The target's end-of-turn ids (its generation config): where an answer ends."""
    from transformers import GenerationConfig

    try:
        eos = GenerationConfig.from_pretrained(
            cfg["repo"], revision=cfg.get("revision")
        ).eos_token_id
    except OSError:
        return []
    return sorted({eos} if isinstance(eos, int) else set(eos or []))


def _turns(entry: dict) -> list[dict]:
    """One conversation as alternating user/assistant turns, or [] if unusable."""
    roles = {"human": "user", "gpt": "assistant"}
    turns: list[dict] = []
    for message in entry.get("conversations", []):
        role, text = (
            roles.get(message.get("from", "")),
            (message.get("value") or "").strip(),
        )
        if not role or not text or (turns and turns[-1]["role"] == role):
            return []
        turns.append({"role": role, "content": text})
    return turns


def prepare_sharegpt(
    conversations: int, min_turns: int, max_context: int, seed: int, force: bool
) -> None:
    if SHAREGPT_FILE.is_file() and not force:
        print(f"{SHAREGPT_FILE} exists; --force to rebuild")
        return
    from huggingface_hub import hf_hub_download

    raw = json.loads(
        Path(hf_hub_download(*SHAREGPT_SOURCE, repo_type="dataset")).read_text()
    )
    selected = []
    for entry in raw:
        turns = _turns(entry)
        if len(turns) < min_turns or turns[0]["role"] != "user":
            continue
        histories = []
        for i, turn in enumerate(turns):
            if turn["role"] != "user":
                continue
            if sum(len(m["content"]) for m in turns[: i + 1]) // 4 > max_context:
                break
            histories.append(turns[: i + 1])
        if histories:
            selected.append(histories)
        if len(selected) == conversations:
            break
    # Stagger conversation starts and jitter the gap between turns, so the
    # stream mixes short openers with long late turns and the reuse distance
    # is a distribution (a few hundred requests), not a constant.
    rng = random.Random(seed)
    total = sum(len(h) for h in selected)
    scheduled = []
    for index, histories in enumerate(selected):
        at = rng.uniform(-REUSE_GAP * 10, total)
        for history in histories:
            if at >= 0:
                scheduled.append((at, index, history))
            at += REUSE_GAP * rng.lognormvariate(0.0, 0.6)
    scheduled.sort(key=lambda entry: entry[0])
    SHAREGPT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SHAREGPT_FILE.open("w") as handle:
        for _, index, history in scheduled:
            handle.write(
                json.dumps(
                    {"category": "sharegpt", "conversation": index, "messages": history}
                )
                + "\n"
            )
    print(
        f"wrote {len(scheduled)} requests from {len(selected)} conversations -> {SHAREGPT_FILE}"
    )

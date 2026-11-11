# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Fit the Fore-D and confidence heads on a frozen drafter: extract once, then fit.

Neither head shapes the drafter here (Fore-D reads detached inputs in training
too), so training them needs no drafter backward pass and only a slice of the
corpus. The target still supplies the hidden states and the labels:

    ./serve.py glm53flash --gpus 0,1,2,3                                   # target hidden states
    ./heads.py extract glm53flash CKPT --split val --batches 0:500 --out X/eval
    ./heads.py extract glm53flash CKPT --split train --batches 0:16000 --out X/fit
    ./heads.py fit glm53flash CKPT X/fit --eval X/eval --out X/heads
    ./heads.py merge CKPT X/heads/heads-fit@all.safetensors --out X/checkpoint

extract  runs the drafter teacher-forced on packed corpus batches, as training
         does, and saves Fore-D's inputs for each sampled anchor with labels for
         every draft width, and the confidence head's inputs and labels for the
         draft slots of a subset of those anchors (models/heads.py defines both).
fit      trains both heads on those files and reports them beside the
         checkpoint's jointly trained heads on the --eval files, raw and after a
         per-slot calibration fitted on a quarter of the --eval batches.
merge    copies the drafter with fitted heads in place of its own, to calibrate
         and serve as any other (scripts/training/heads.sh runs them all).

--batches slices one fixed packing of the split, so slices are disjoint;
--shard i/n splits a slice across processes, one GPU each.
"""

import argparse
import json
import math
import shutil
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from ..config import model_config, target_snapshot
from ..models.heads import ForeDHead, overlap_label
from ..paths import DATA
from .launch import gamma_for

TRAIN_DATA_RATIO = 0.95  # launch.py's split: the tail 5% is validation
BATCH_KEYS = ("hidden_states", "input_ids", "loss_mask", "verifier_last_hidden_states", "document_ids",
              "position_ids")  # fmt: skip
REGRET_COSTS = (
    0.05,
    0.1,
    0.2,
    0.3,
)  # accepted tokens one drafted token costs, for the width-choice regret
RANKING_BUDGETS = (
    0.1,
    0.25,
    0.5,
)  # fractions of draft slots verified, for the ranking metric


def load_drafter(checkpoint: Path, verifier: str):
    from ..models.draft_model import DFlash2ConfDraftModel, DFlash2ConfSpeculatorConfig
    from .entry import register_target_configs

    register_target_configs()
    config = DFlash2ConfSpeculatorConfig.from_pretrained(checkpoint)
    # Embeddings, LM head and final norm come from the target served here, not
    # from the one the checkpoint names.
    config.speculators_config.verifier.name_or_path = verifier
    config.transformer_layer_config._attn_implementation = "simple_flex_attention"
    model = DFlash2ConfDraftModel.from_pretrained(checkpoint, config=config)
    return model.to("cuda", torch.bfloat16).eval().requires_grad_(False)


def _full_width(model, hs, ids, lm, vlh, doc, pos, anchors, targets, normalizer):
    """Every anchor's full-width draft: confidence inputs, slot labels, Fore-D inputs."""
    out = model._backbone_forward(
        hs, ids, lm, vlh, doc, pos, max_anchors=anchors[0].numel(), return_pre_draft_features=True,
        anchors=anchors, target_logits=targets,
    )  # fmt: skip
    hidden, unary, targets, mask, indices = out[:5]
    predecessors = model._predecessor_ids(ids, indices)
    candidates = unary.topk(model.candidate_selector.top_k, dim=-1).indices
    scores = model.candidate_selector.score_candidates(
        unary, hidden, predecessors, candidates
    )
    label = overlap_label(targets, candidates, scores, normalizer)
    selector = model.candidate_selector.context(hidden, predecessors)
    return hidden, selector, label, mask, *out[5][:2]


@torch.no_grad()
def extract_batch(
    model, full_width, batch, num_anchors: int, num_confidence: int, seed: int
) -> dict:
    hs, ids, lm, vlh, doc, pos = (
        batch[key].cuda(non_blocking=True) for key in BATCH_KEYS
    )
    k, block = model.block_size - 1, model.block_size
    torch.manual_seed(seed)
    anchors = model._select_anchors(lm, num_anchors)
    targets, normalizer = model._block_targets_and_normalizer(vlh, anchors[0])
    hidden, selector, label, mask, context, anchor_embeddings = full_width(
        model, hs, ids, lm, vlh, doc, pos, anchors, targets, normalizer
    )
    label, mask = label.view(-1, block), mask.view(-1, block).bool()
    rows = {k: (label[:, 1:], mask[:, 1:])}
    for width in range(1, k):
        # The drafter attends bidirectionally inside its block: each width drafts anew.
        masks = model._build_attention_mask(
            lm, num_anchors, doc, hs.device, width + 1, anchors
        )
        rows[width] = model._narrow_labels(
            width,
            1,
            targets,
            normalizer,
            anchors,
            masks[:2],
            hs,
            ids,
            lm,
            vlh,
            doc,
            pos,
        )
    labels = label.new_zeros(num_anchors, k * (k + 1) // 2)
    valid = torch.zeros_like(labels, dtype=torch.bool)
    for width, (row, row_valid) in rows.items():
        start = width * (width - 1) // 2
        labels[:, start : start + width] = row[:, :width]
        valid[:, start : start + width] = row_valid[:, :width]
    picked = (anchors[1].bool() & valid.any(-1)).nonzero().squeeze(1)
    confidence = picked[:num_confidence]

    def slots(t):
        return t[0].view(num_anchors, block, -1)[confidence, 1:]

    return {
        "fore_d_context": context[picked], "fore_d_anchor": anchor_embeddings[picked],
        "fore_d_label": labels[picked].half(), "fore_d_valid": valid[picked],
        "conf_hidden": slots(hidden), "conf_selector": slots(selector),
        "conf_label": label[confidence, 1:].half(), "conf_valid": mask[confidence, 1:],
    }  # fmt: skip


def extract(args) -> int:
    from hs_connectors import FileTransfer
    from speculators.train.data import ArrowDataset, CollateFn
    from speculators.train.distributed_batch_sampler import (
        MultipackDistributedBatchSamplerV2,
    )
    from torch.utils.data import DataLoader

    cfg = model_config(args.model)
    verifier = target_snapshot(cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    data_path = args.data_path or DATA / "prepared" / args.model
    dataset = ArrowDataset(
        max_len=args.seq_len, datapath=str(data_path),
        transfer=FileTransfer(args.out / "hidden_states"),  # a cache it never writes: on_generate="delete"
        vllm_endpoint=args.endpoint or f"http://127.0.0.1:{cfg['port'] + 100}/v1",
        on_missing="generate", on_generate="delete", train_ratio=TRAIN_DATA_RATIO, split=args.split, model=verifier,
    )  # fmt: skip
    packing = list(
        MultipackDistributedBatchSamplerV2(
            batch_max_length=args.seq_len,
            lengths=dataset.approx_lengths,
            num_replicas=1,
            rank=0,
        )
    )
    shard, shards = args.shard
    ids = list(range(args.batches[0], min(args.batches[1], len(packing))))[
        shard::shards
    ]
    done = len(list(args.out.glob(f"part-{shard:02d}-*.pt")))
    ids = ids[done * args.part_batches :]  # resume after the parts already written
    if not ids:
        print(f"{args.out} shard {shard}/{shards}: nothing left")
        return 0
    model = load_drafter(args.checkpoint, verifier)
    full_width = torch.compile(_full_width, dynamic=False)
    loader = DataLoader(
        dataset, batch_sampler=[packing[i].tolist() for i in ids], num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers else None, pin_memory=True,
        collate_fn=CollateFn(args.seq_len, model.config.transformer_layer_config.hidden_size,
                             len(model.target_layer_ids)),
        multiprocessing_context="spawn" if args.num_workers else None,
    )  # fmt: skip
    meta = {
        "model": args.model, "checkpoint": str(args.checkpoint), "verifier": verifier, "data_path": str(data_path),
        "split": args.split, "batches": args.batches, "packed_batches": len(packing), "anchors": args.anchors,
        "confidence_anchors": args.confidence_anchors, "seed": args.seed, "k": model.block_size - 1,
    }  # fmt: skip
    (args.out / f"meta-{shard:02d}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"{args.out} shard {shard}/{shards}: {len(ids)} batches of {len(packing)} ({args.split})",
        flush=True,
    )

    records, part, waited, start = defaultdict(list), done, 0.0, time.time()
    ready = time.time()
    for n, (batch_id, batch) in enumerate(zip(ids, loader, strict=True), 1):
        waited += time.time() - ready
        got = extract_batch(model, full_width, batch, args.anchors, args.confidence_anchors,
                            args.seed * 1_000_003 + batch_id)  # fmt: skip
        for key, value in got.items():
            records[key].append(value.cpu())
        for prefix in ("fore_d", "conf"):
            count = len(got[f"{prefix}_label"])
            records[f"{prefix}_batch"].append(
                torch.full((count,), batch_id, dtype=torch.int32)
            )
        if n % args.part_batches == 0 or n == len(ids):
            path = args.out / f"part-{shard:02d}-{part:04d}.pt"
            # Renamed once whole: resuming counts the parts present.
            partial = path.with_suffix(".partial")
            torch.save(
                {key: torch.cat(values) for key, values in records.items()}, partial
            )
            partial.rename(path)
            records.clear()
            part += 1
        if n % 10 == 0 or n == len(ids):
            elapsed = time.time() - start
            print(f"  {n}/{len(ids)} batches  {elapsed / n:.2f} s/batch ({waited / n:.2f} waiting for hidden states)",
                  flush=True)  # fmt: skip
        ready = time.time()
    return 0


def load_records(
    directories: list[Path], prefix: str, batches: int | None = None
) -> dict[str, torch.Tensor]:
    """One head's records from extract directories, from their first `batches` batches."""
    parts = sorted(
        part for directory in directories for part in directory.glob("part-*.pt")
    )
    if not parts:
        raise SystemExit(f"{directories}: no part-*.pt (run extract first)")
    loaded = [
        torch.load(part, map_location="cpu", mmap=True) for part in parts
    ]  # reads only this head's tensors
    data = {
        key: torch.cat([p[key] for p in loaded])
        for key in loaded[0]
        if key.startswith(prefix)
    }
    if batches is not None:
        first = data[f"{prefix}_batch"].unique(sorted=True)[:batches]
        keep = torch.isin(data[f"{prefix}_batch"], first)
        data = {key: value[keep] for key, value in data.items()}
    return data


def _cosine(step: int, steps: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))


def _train(head, loss_of, size: int, args) -> None:
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=0.0, foreach=True
    )
    warmup = max(1, args.steps // 30)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _cosine(s, args.steps, warmup)
    )
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    for _ in range(args.steps):
        idx = torch.randint(
            size, (args.batch_size,), device="cuda", generator=generator
        )
        loss = loss_of(idx)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        schedule.step()


def _as_shipped(head):
    """The head at the checkpoint's precision (bf16), computed in fp32 as the server does."""
    return head.to(torch.bfloat16).float().eval().requires_grad_(False)


def confidence_logits(head, data, idx=None) -> torch.Tensor:
    hidden, selector = data["conf_hidden"], data["conf_selector"]
    if idx is not None:
        hidden, selector = hidden[idx], selector[idx]
    return head(torch.cat([hidden, selector], -1).float())


def confidence_loss(logits, labels, valid, decay) -> torch.Tensor:
    """The trainer's confidence objective: decayed soft BCE over the valid slots."""
    per_slot = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (per_slot * valid * decay).sum() / valid.sum().clamp_min(1.0)


def fore_d_loss_per_width(logits, labels, valid, width_of) -> torch.Tensor:
    """The trainer's Fore-D objective: each step fits one width's row, every width in turn."""
    per_entry = (
        F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * valid
    )
    k = int(width_of.max()) + 1
    sums = logits.new_zeros(k).index_add_(0, width_of, per_entry.sum(0))
    counts = logits.new_zeros(k).index_add_(0, width_of, valid.sum(0))
    present = counts > 0
    return (sums[present] / counts[present]).mean()


def _to_cuda(data: dict) -> dict:
    return {key: value.cuda() for key, value in data.items()}


def _float_labels(data: dict, prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
    return data[f"{prefix}_label"].float(), data[f"{prefix}_valid"].float()


def fit_confidence(data: dict, k: int, gamma: float, args):
    from speculators.models.dspark.model_definitions import ConfidenceHead

    labels, valid = _float_labels(data, "conf")
    decay = torch.exp(
        -torch.arange(k, device="cuda") / gamma
    )  # slot j >= 1 weighs exp(-(j - 1) / gamma)
    torch.manual_seed(args.seed)
    head = ConfidenceHead(
        data["conf_hidden"].shape[-1] + data["conf_selector"].shape[-1]
    ).cuda()
    _train(head, lambda i: confidence_loss(confidence_logits(head, data, i), labels[i], valid[i], decay),
           len(labels), args)  # fmt: skip
    return _as_shipped(head)


def fit_fore_d(data: dict, width_of: torch.Tensor, rank: int, args):
    labels, valid = _float_labels(data, "fore_d")
    context, anchor = data["fore_d_context"], data["fore_d_anchor"]
    torch.manual_seed(args.seed)
    head = ForeDHead(context.shape[-1], rank, labels.shape[-1]).cuda()
    _train(head, lambda i: fore_d_loss_per_width(head(context[i].float(), anchor[i].float()), labels[i], valid[i],
                                                 width_of), len(labels), args)  # fmt: skip
    return _as_shipped(head)


def _temperatures(logits, labels, valid, per: str) -> tuple[torch.Tensor, torch.Tensor]:
    """A scale (and, per entry, a bias) per column, fitted by soft BCE: STS-like for
    the confidence head (per='slot', scale only), Fore-D's calibration (per='entry')."""
    columns = logits.shape[-1]
    scale = torch.ones(columns, device=logits.device, requires_grad=True)
    bias = torch.zeros(columns, device=logits.device, requires_grad=per == "entry")
    optimizer = torch.optim.LBFGS([scale, bias] if per == "entry" else [scale], max_iter=200,
                                  line_search_fn="strong_wolfe")  # fmt: skip

    def closure():
        optimizer.zero_grad()
        loss = (
            F.binary_cross_entropy_with_logits(
                logits * scale + bias, labels, reduction="none"
            )
            * valid
        ).sum()
        loss = loss / valid.sum().clamp_min(1.0)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scale.detach(), bias.detach()


def _ranking(predicted, observed, valid) -> dict:
    """Of the acceptance the best `budget` share of draft slots holds, the share the
    predicted ranking verifies (adaptive verification spends its budget this way)."""
    keep = valid.bool()
    predicted, observed = predicted[keep], observed[keep]
    order = predicted.argsort(descending=True)
    best = observed.sort(descending=True).values
    out = {}
    for budget in RANKING_BUDGETS:
        top = max(1, int(budget * len(order)))
        out[f"ranking@{budget:g}"] = float(
            observed[order[:top]].sum() / best[:top].sum().clamp_min(1e-9)
        )
    return out


def confidence_metrics(logits, labels, valid, decay) -> dict:
    p = logits.sigmoid()
    survival_pred = torch.cumprod(p * valid, -1)  # an invalid slot ends the block
    survival_true = torch.cumprod(labels * valid, -1)
    rows = valid.any(-1)
    accept_pred, accept_true = (
        1 + survival_pred.sum(-1)[rows],
        1 + survival_true.sum(-1)[rows],
    )
    count = valid.sum().clamp_min(1.0)
    return {
        "objective": float(confidence_loss(logits, labels, valid, decay)),
        "bce": float((F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * valid).sum() / count),
        "brier": float(((p - labels) ** 2 * valid).sum() / count),
        "accept_len_pred": float(accept_pred.mean()), "accept_len_true": float(accept_true.mean()),
        "accept_len_mae": float((accept_pred - accept_true).abs().mean()),
        **_ranking(survival_pred, survival_true, valid),
    }  # fmt: skip


def _width_rows(k: int) -> list[slice]:
    return [slice(w * (w - 1) // 2, w * (w + 1) // 2) for w in range(1, k + 1)]


def fore_d_metrics(logits, labels, valid, width_of) -> dict:
    k = int(width_of.max()) + 1
    out = {"objective": float(fore_d_loss_per_width(logits, labels, valid, width_of))}
    full = valid.bool().all(
        -1
    )  # anchors whose every width is labeled (not cut by a document end)
    p, y = logits[full].sigmoid(), labels[full]
    expected_pred = torch.stack(
        [torch.cumprod(p[:, r], -1).sum(-1) for r in _width_rows(k)], -1
    )  # [anchors, K]
    expected_true = torch.stack(
        [torch.cumprod(y[:, r], -1).sum(-1) for r in _width_rows(k)], -1
    )
    error = (expected_pred - expected_true).abs()
    out |= {
        "anchors_full": int(full.sum()),
        "accepted_true@K": float(expected_true[:, -1].mean()), "accepted_pred@K": float(expected_pred[:, -1].mean()),
        "accepted_mae_mean": float(error.mean()),
        **{f"accepted_mae@{w}": float(error[:, w - 1].mean()) for w in sorted({1, 4, 8, k} & set(range(1, k + 1)))},
    }  # fmt: skip
    widths = torch.arange(1, k + 1, device=logits.device, dtype=torch.float32)
    for cost in REGRET_COSTS:
        utility_true = expected_true - cost * widths
        chosen = (expected_pred - cost * widths).argmax(-1)
        regret = utility_true.max(-1).values - utility_true.gather(
            -1, chosen[:, None]
        ).squeeze(-1)
        out[f"regret@{cost:g}"] = float(regret.mean())
    return out


def evaluate(
    heads: dict, data: dict, k: int, gamma: float, width_of: torch.Tensor
) -> dict:
    """Each head set on the eval records: raw, and calibrated on the first quarter of
    the eval batches (scored on the rest)."""
    decay = torch.exp(-torch.arange(k, device="cuda") / gamma)
    conf_labels, conf_valid = _float_labels(data["conf"], "conf")
    fd_labels, fd_valid = _float_labels(data["fore_d"], "fore_d")
    report = {}
    for name, (confidence, fore_d) in heads.items():
        with torch.no_grad():
            conf_logits = torch.cat(
                [
                    confidence_logits(confidence, data["conf"], i)
                    for i in torch.arange(len(conf_labels), device="cuda").split(4096)
                ]
            )
            fd_logits = fore_d(
                data["fore_d"]["fore_d_context"].float(),
                data["fore_d"]["fore_d_anchor"].float(),
            )
        entry = {"confidence": confidence_metrics(conf_logits, conf_labels, conf_valid, decay),
                 "fore_d": fore_d_metrics(fd_logits, fd_labels, fd_valid, width_of)}  # fmt: skip
        for head, prefix, logits, labels, valid in (
            ("confidence", "conf", conf_logits, conf_labels, conf_valid),
            ("fore_d", "fore_d", fd_logits, fd_labels, fd_valid),
        ):
            batch = data[prefix][f"{prefix}_batch"]
            calibration = batch < torch.quantile(batch.unique().float(), 0.25)
            held = ~calibration
            per = "slot" if head == "confidence" else "entry"
            scale, bias = _temperatures(
                logits[calibration], labels[calibration], valid[calibration], per
            )
            calibrated = logits[held] * scale + bias
            entry[f"{head}_calibrated"] = (
                confidence_metrics(calibrated, labels[held], valid[held], decay) if head == "confidence"
                else fore_d_metrics(calibrated, labels[held], valid[held], width_of)
            )  # fmt: skip
        report[name] = entry
    return report


def joint_heads(
    checkpoint: Path, hidden: int, selector_rank: int, rank: int, columns: int
):
    from safetensors.torch import load_file
    from speculators.models.dspark.model_definitions import ConfidenceHead

    state = load_file(str(checkpoint / "model.safetensors"))
    confidence = ConfidenceHead(hidden + selector_rank)
    fore_d = ForeDHead(hidden, rank, columns)
    for head, prefix in (
        (confidence, "confidence_head."),
        (fore_d, "fore_d_predictor."),
    ):
        head.load_state_dict(
            {
                key[len(prefix) :]: value
                for key, value in state.items()
                if key.startswith(prefix)
            }
        )
    return confidence.cuda().float().eval(), fore_d.cuda().float().eval()


def _markdown(report: dict) -> str:
    columns = {
        "confidence": (
            "objective",
            "brier",
            "accept_len_mae",
            "ranking@0.25",
            "ranking@0.5",
        ),
        "fore_d": (
            "objective",
            "accepted_mae_mean",
            "accepted_mae@15",
            "regret@0.1",
            "regret@0.2",
        ),
    }
    lines = []
    for section in (
        "confidence",
        "confidence_calibrated",
        "fore_d",
        "fore_d_calibrated",
    ):
        keys = columns[section.removesuffix("_calibrated")]
        lines += [
            f"### {section}",
            "",
            "| heads | " + " | ".join(keys) + " |",
            "|---" * (len(keys) + 1) + "|",
        ]
        for name, entry in report.items():
            lines.append(f"| {name} | " + " | ".join(f"{entry[section].get(key, float('nan')):.4f}" for key in keys)
                         + " |")  # fmt: skip
        lines.append("")
    return "\n".join(lines)


def fit(args) -> int:
    from safetensors.torch import save_file

    config = json.loads((args.checkpoint / "config.json").read_text())
    k = config["block_size"] - 1
    gamma = float(gamma_for(config["block_size"]))
    width_of = torch.cat([torch.full((w,), w - 1) for w in range(1, k + 1)]).cuda()
    hidden = config["transformer_layer_config"]["hidden_size"]
    joint = joint_heads(
        args.checkpoint,
        hidden,
        config["selector_rank"],
        config["fore_d_rank"],
        len(width_of),
    )
    evaluation = {
        prefix: _to_cuda(load_records([args.eval], prefix))
        for prefix in ("conf", "fore_d")
    }
    args.out.mkdir(parents=True, exist_ok=True)
    heads = {"joint": joint}
    for entry in args.fit:
        directories = [Path(d) for d in entry.split("+")]
        for size in args.sizes or [None]:
            name = f"{'+'.join(d.name for d in directories)}@{size or 'all'}"
            started = time.time()
            conf = _to_cuda(load_records(directories, "conf", size))
            confidence = fit_confidence(conf, k, gamma, args)
            del conf
            fore_d_data = _to_cuda(load_records(directories, "fore_d", size))
            fore_d = fit_fore_d(fore_d_data, width_of, config["fore_d_rank"], args)
            anchors = len(fore_d_data["fore_d_label"])
            del fore_d_data
            heads[name] = (confidence, fore_d)
            save_file({**{f"confidence_head.{key}": value.to(torch.bfloat16).contiguous()
                          for key, value in confidence.state_dict().items()},
                       **{f"fore_d_predictor.{key}": value.to(torch.bfloat16).contiguous()
                          for key, value in fore_d.state_dict().items()}},
                      str(args.out / f"heads-{name}.safetensors"))  # fmt: skip
            print(
                f"fitted {name}: {anchors} Fore-D anchors in {time.time() - started:.0f} s",
                flush=True,
            )
            torch.cuda.empty_cache()
    report = evaluate(heads, evaluation, k, gamma, width_of)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "report.md").write_text(_markdown(report) + "\n")
    print(_markdown(report))
    return 0


def merge(args) -> int:
    """The drafter's config, code and weights, with the fitted heads in place of its own."""
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    weights = args.checkpoint / "model.safetensors"
    with safe_open(str(weights), "pt") as f:
        metadata = f.metadata()
    state = load_file(str(weights))
    heads = load_file(str(args.heads))
    for key, value in heads.items():
        if key not in state or state[key].shape != value.shape:
            raise SystemExit(
                f"{args.heads}: {key} {tuple(value.shape)} is not a head of {weights}"
            )
        state[key] = value.to(state[key].dtype)
    args.out.mkdir(parents=True, exist_ok=True)
    # Not the trainer's optimizer and scheduler state: they belong to the joint run.
    for path in [args.checkpoint / "config.json", *args.checkpoint.glob("*.py")]:
        shutil.copy2(path, args.out / path.name)
    save_file(state, str(args.out / "model.safetensors"), metadata=metadata)
    print(f"{args.out}: {args.checkpoint} with {len(heads)} tensors from {args.heads}")
    return 0


def _span(text: str) -> tuple[int, int]:
    lo, _, hi = text.partition(":")
    return int(lo or 0), int(hi) if hi else 1 << 62


def _shard(text: str) -> tuple[int, int]:
    i, _, n = text.partition("/")
    return int(i), int(n or 1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser(
        "extract", help="head inputs and labels from packed corpus batches"
    )
    e.add_argument("model")
    e.add_argument("checkpoint", type=Path)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--split", choices=["train", "val"], default="val")
    e.add_argument(
        "--batches",
        type=_span,
        default=(0, 500),
        help="slice of the split's packed batches, LO:HI",
    )
    e.add_argument(
        "--shard",
        type=_shard,
        default=(0, 1),
        help="i/n: this process's share of the slice",
    )
    e.add_argument(
        "--anchors", type=int, default=256, help="anchors per batch (Fore-D records)"
    )
    e.add_argument(
        "--confidence-anchors",
        type=int,
        default=64,
        help="of those, the ones with confidence records",
    )
    e.add_argument(
        "--data-path", type=Path, help="prepared corpus (default: data/prepared/MODEL)"
    )
    e.add_argument(
        "--endpoint",
        help="hidden-state server (default: serve.py's port for the target)",
    )
    e.add_argument("--seq-len", type=int, default=8192)
    e.add_argument("--num-workers", type=int, default=8)
    e.add_argument(
        "--part-batches", type=int, default=50, help="batches per saved part"
    )
    e.add_argument("--seed", type=int, default=0)
    f = sub.add_parser(
        "fit", help="train both heads on extracted records and report them"
    )
    f.add_argument("model")
    f.add_argument(
        "checkpoint",
        type=Path,
        help="the drafter, for its jointly trained heads and shapes",
    )
    f.add_argument(
        "fit",
        nargs="+",
        help="extract directories to fit on, each separately; A+B fits on their union",
    )
    f.add_argument(
        "--eval", type=Path, required=True, help="extract directory to report on"
    )
    f.add_argument("--out", type=Path, required=True)
    f.add_argument(
        "--sizes",
        type=int,
        nargs="*",
        help="first-N-batch subsets to fit (learning curve)",
    )
    f.add_argument("--steps", type=int, default=3000)
    f.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="anchors (Fore-D) or blocks (confidence) per step",
    )
    f.add_argument("--lr", type=float, default=1e-3)
    f.add_argument("--seed", type=int, default=0)
    m = sub.add_parser(
        "merge", help="the drafter with fitted heads in place of its own"
    )
    m.add_argument("checkpoint", type=Path, help="the drafter the heads were fitted on")
    m.add_argument("heads", type=Path, help="a heads-*.safetensors that fit wrote")
    m.add_argument("--out", type=Path, required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return {"extract": extract, "fit": fit, "merge": merge}[args.command](args)

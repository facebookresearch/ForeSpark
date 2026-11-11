# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Portions adapted from speculators' DFlash and DFlash2 models (https://github.com/vllm-project/speculators:
# src/speculators/models/dflash/core.py, src/speculators/models/dflash2/core.py), Apache License 2.0.

"""Training-side dflash2_conf: DFlash2 + confidence head + width-conditioned Fore-D head.

    L = L_dflash2 (ce, tv, selector) + a_conf * L_conf + a_fore_d * L_fore_d

Both heads fit one label: the acceptance probability of each draft slot under
DFlash2's actual proposal (the softmax over the selector-corrected serving
top-k; heads.overlap_label), as DSpark fits its confidence head (eq. 12) in the
same run as the drafter.

  confidence  reads the drafted slot and its realized predecessor (post-draft;
              adaptive verification ranks it)
  Fore-D     reads only what exists before drafting -- the target state before
              the anchor and the anchor token -- and predicts every slot of
              every draft width at once (Fore-D's draft widths). The
              drafter attends bidirectionally inside its block, so each width
              has its own row, labelled by a narrower re-draft (forward()).

Registered with speculators as `dflash2_conf` (train/entry.py imports it).
"""

import gc
from functools import partial
from typing import Any, ClassVar, Literal

import torch
from pydantic import Field
from speculators import reload_schemas, SpeculatorModelConfig
from speculators.losses import (
    dflash_loss_decay,
    dpace_loss_decay,
    kl_div_loss,
    loss_function,
    tv_loss,
)
from speculators.model import SpeculatorModel
from speculators.models.dflash.utils import (
    get_base_indices_for_anchored_blocks,
    select_anchors,
)
from speculators.models.dflash2 import DFlash2DraftModel, DFlash2SpeculatorConfig
from speculators.models.dflash2.metrics import (
    compute_metrics,
    selector_training_candidates,
)
from speculators.models.dspark.model_definitions import ConfidenceHead
from speculators.models.utils import conditional_torch_compile
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.functional import binary_cross_entropy_with_logits

from .heads import (
    confidence_features,
    confidence_input_dim,
    fore_d_loss,
    ForeDHead,
    log_normalizer,
    overlap_label,
)
from .layers import DFlash2ConfDecoderLayer
from .topk import topk_indices

_DEFAULT_LOSS_CONFIG = {"kl_div": (kl_div_loss, 1.0)}  # DFlash2's own fallback


@SpeculatorModelConfig.register("dflash2_conf")
class DFlash2ConfSpeculatorConfig(DFlash2SpeculatorConfig):
    speculators_model_type: Literal["dflash2_conf"] = "dflash2_conf"  # type: ignore[assignment]
    architectures: list[str] = Field(default_factory=lambda: ["DFlash2ConfDraftModel"])
    enable_confidence_head: bool = Field(
        default=True, description="post-draft acceptance head"
    )
    confidence_head_alpha: float = Field(
        default=1.0, ge=0.0, description="weight of its BCE (DSpark eq. 12: 1.0)"
    )
    enable_fore_d_predictor: bool = Field(
        default=False, description="pre-draft acceptance head"
    )
    fore_d_rank: int = Field(
        default=256, ge=1, description="hidden rank of the Fore-D head"
    )
    # A non-causal drafter's draft changes with its width: one row per width
    # d = 1..K (K(K+1)/2 logits, width d's at offset d(d-1)/2).
    fore_d_output_semantics: Literal["width_conditioned_survival"] = (
        "width_conditioned_survival"
    )


# Configs registered outside the speculators package need their pydantic schema
# rebuilt with torch in scope before they can be constructed.
reload_schemas()


@SpeculatorModel.register("dflash2_conf")
class DFlash2ConfDraftModel(DFlash2DraftModel):
    config_class: ClassVar[type[DFlash2ConfSpeculatorConfig]] = (
        DFlash2ConfSpeculatorConfig  # type: ignore[misc,assignment]
    )

    def __init__(self, config: DFlash2ConfSpeculatorConfig) -> None:
        if config.enable_fore_d_predictor and not config.sliding_window_non_causal:
            raise ValueError(
                "the width-conditioned Fore-D head is for a non-causal drafter (--sliding-window-non-causal)"
            )
        super().__init__(config=config)
        hidden = config.transformer_layer_config.hidden_size
        self.confidence_head = (
            ConfidenceHead(confidence_input_dim(hidden, config.selector_rank))
            if config.enable_confidence_head
            else None
        )
        k = self.block_size - 1
        self.width_conditioned = config.enable_fore_d_predictor
        self.fore_d_predictor = (
            ForeDHead(hidden, config.fore_d_rank, k * (k + 1) // 2)
            if config.enable_fore_d_predictor
            else None
        )
        self._train_steps = 0  # which width the width-conditioned head learns next
        if self.width_conditioned:
            # A static graph and block mask per narrow width: dynamo's defaults stop
            # compiling after 8 and generalize the shapes, which builds the masks ~14x slower.
            torch._dynamo.config.recompile_limit = max(
                torch._dynamo.config.recompile_limit, 4 * self.block_size
            )
            if self._attn_impl == "simple_flex_attention":
                self._create_mask_fn = torch.compile(create_block_mask, dynamic=False)

    def _make_decoder_layer(self, config, layer_idx: int) -> DFlash2ConfDecoderLayer:
        return DFlash2ConfDecoderLayer(
            config.transformer_layer_config, layer_idx, block_size=config.block_size,
            conv_kernel_size=config.conv_kernel_size, conv_group_size=config.conv_group_size,
        )  # fmt: skip

    @classmethod
    def from_training_args(
        cls, verifier_config, t2d=None, d2t=None, **kwargs
    ) -> "DFlash2ConfDraftModel":
        # "dflash2" selects the architecture defaults; flags still override them.
        base = cls._build_base_config_kwargs("dflash2", verifier_config, **kwargs)
        base["speculators_config"].algorithm = "dflash2_conf"
        config = DFlash2ConfSpeculatorConfig(
            **base,
            conv_kernel_size=kwargs.get("conv_kernel_size", 2),
            conv_group_size=kwargs.get("conv_group_size", 16),
            selector_rank=kwargs.get("selector_rank", 256),
            selector_top_k=kwargs.get("selector_top_k", 16),
            enable_confidence_head=kwargs.get("enable_confidence_head", True),
            confidence_head_alpha=kwargs.get("confidence_head_alpha", 1.0),
            enable_fore_d_predictor=kwargs.get("enable_fore_d_predictor", False),
            fore_d_rank=kwargs.get("fore_d_rank", 256),
        )
        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        train, val = DFlash2DraftModel.get_trainer_kwargs(**kwargs)
        for call in (train, val):
            call["confidence_head_alpha"] = kwargs.get("confidence_head_alpha", 1.0)
            call["fore_d_loss_alpha"] = kwargs.get("fore_d_loss_alpha", 1.0)
        train["fore_d_narrow_stride"] = kwargs.get("fore_d_narrow_stride", 4)
        return train, val

    def gradient_clip_groups(self) -> list[list[torch.nn.Parameter]]:
        """The Fore-D head reads detached inputs; clip it on its own norm so the
        drafter's gradient norm cannot scale it to nothing (trainer patch)."""
        if self.fore_d_predictor is None:
            return [list(self.parameters())]
        head = {id(p) for p in self.fore_d_predictor.parameters()}
        return [
            [p for p in self.parameters() if id(p) not in head],
            list(self.fore_d_predictor.parameters()),
        ]

    def _predecessor_ids(
        self,
        input_ids: torch.Tensor,
        block_indices: torch.Tensor,
        block: int | None = None,
    ) -> torch.Tensor:
        """Ground-truth token before each block slot (the anchor for slot 0)."""
        tokens = input_ids[0, block_indices].view(-1, block or self.block_size)
        return torch.cat([tokens[:, :1], tokens[:, :-1]], dim=1).reshape(1, -1)

    def forward(  # noqa: PLR0917
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        verifier_last_hidden_states: torch.Tensor,
        document_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[None, torch.Tensor, dict[str, Any]]:
        """A width-conditioned Fore-D head learns width w's row: K, 1, ..., K-1 in
        turn over training steps (identical on every rank), K in evaluation. Below
        K, the narrow re-draft runs before the step, from the step's anchors and
        block targets, so its memory never stacks on the step's saved activations.

        Every training step calls the step with the same arguments (row K's labels
        come from its own drafts, chosen by a flag), so it compiles once, on the
        first step, before any narrow re-draft. Under FSDP each decoder layer is
        its own frame, shared with the narrow re-draft; a step compiled after one
        would generalize the block size and fail in inductor."""
        batch = (
            hidden_states,
            input_ids,
            loss_mask,
            verifier_last_hidden_states,
            document_ids,
            position_ids,
        )
        stride = kwargs.pop("fore_d_narrow_stride", 4)
        if not self.width_conditioned:
            return self._step(*batch, **kwargs)
        k = self.block_size - 1
        width = (self._train_steps - 1) % k + 1 if self.training else k
        start = width * (width - 1) // 2
        columns = torch.arange(start, start + k, device=hidden_states.device)
        if not self.training:
            return self._step(*batch, fore_d_columns=columns, **kwargs)
        if self._train_steps <= k:
            # Each width's first step compiles, and dynamo's compile state is cyclic garbage
            # that holds that step's inputs (the block targets) until a full collection.
            gc.collect()
        self._train_steps += 1
        max_anchors = kwargs["max_anchors"]
        anchors = self._select_anchors(loss_mask, max_anchors)
        targets, normalizer = self._block_targets_and_normalizer(
            verifier_last_hidden_states, anchors[0]
        )
        if width < k:
            # Every stride-th anchor: a draft block sees only its own block and the
            # context, so those anchors' labels are the ones all anchors would get.
            narrow = tuple(a[::stride] for a in anchors)
            masks = self._build_attention_mask(
                loss_mask,
                narrow[0].numel(),
                document_ids,
                hidden_states.device,
                width + 1,
                narrow,
            )
            rows = self._narrow_labels(
                width, stride, targets, normalizer, narrow, masks[:2], *batch
            )
        else:
            shape, device = (anchors[0].numel(), k), hidden_states.device
            rows = (
                torch.zeros(shape, device=device),
                torch.zeros(shape, dtype=torch.bool, device=device),
            )
        # A device fill, not torch.tensor: a host-to-device copy would sync mid-step.
        own = torch.full((), width == k, dtype=torch.bool, device=hidden_states.device)
        return self._step(
            *batch, anchors=anchors, target_logits=targets, normalizer=normalizer, fore_d_rows=rows,
            fore_d_own=own, fore_d_columns=columns, **kwargs,
        )  # fmt: skip

    @torch.compiler.disable
    def _select_anchors(
        self, loss_mask: torch.Tensor, max_anchors: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The step's anchors (its one random draw), shared by the narrow re-draft."""
        return select_anchors(loss_mask, max_anchors, self.block_size)

    @conditional_torch_compile(dynamic=False)
    @torch.no_grad()
    def _block_targets_and_normalizer(
        self, verifier_last_hidden_states, anchor_positions
    ):
        indices = get_base_indices_for_anchored_blocks(
            anchor_positions, self.block_size
        )
        targets = self._block_targets(verifier_last_hidden_states, indices)
        return targets, log_normalizer(targets)

    @conditional_torch_compile
    def _step(  # noqa: PLR0917
        self,
        hidden_states: torch.Tensor,  # [1, T, layers * hidden] packed
        input_ids: torch.Tensor,  # [1, T]
        loss_mask: torch.Tensor,  # [1, T]
        verifier_last_hidden_states: torch.Tensor,  # [1, T, hidden]
        document_ids: torch.Tensor,  # [1, T]
        position_ids: torch.Tensor | None = None,
        loss_config=None,
        tv_loss_fn=tv_loss,
        gamma: float = 4.0,
        max_anchors: int = 512,
        selector_loss_alpha: float = 1.0,
        confidence_head_alpha: float = 1.0,
        fore_d_loss_alpha: float = 1.0,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        anchors: tuple[torch.Tensor, torch.Tensor] | None = None,
        target_logits: torch.Tensor | None = None,
        normalizer: torch.Tensor | None = None,
        fore_d_rows: tuple[torch.Tensor, torch.Tensor] | None = None,
        fore_d_own: torch.Tensor | None = None,
        fore_d_columns: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[None, torch.Tensor, dict[str, Any]]:
        with_fore_d = self.fore_d_predictor is not None
        output = self._backbone_forward(
            hidden_states, input_ids, loss_mask, verifier_last_hidden_states, document_ids, position_ids,
            max_anchors=max_anchors, return_pre_draft_features=with_fore_d, anchors=anchors, target_logits=target_logits,
            **kwargs,
        )  # fmt: skip
        hidden, unary_logits, targets, mask, block_indices = output[:5]
        predecessor_ids = self._predecessor_ids(input_ids, block_indices)
        top_k = self.candidate_selector.top_k
        candidate_ids = unary_logits.topk(top_k, dim=-1).indices
        training_ids, target_positions, contains_target = selector_training_candidates(
            candidate_ids, targets.argmax(dim=-1)
        )
        loss, metrics = compute_metrics(
            unary_logits=unary_logits,
            targets=targets,
            training_candidate_ids=training_ids,
            candidate_logits=self.candidate_selector.score_candidates(
                unary_logits, hidden, predecessor_ids, training_ids
            ),
            target_positions=target_positions,
            contains_target=contains_target,
            loss_mask=mask,
            block_size=self.block_size,
            top_k=top_k,
            sample_from_anchor=self.config.sample_from_anchor,
            loss_config=loss_config or _DEFAULT_LOSS_CONFIG,
            tv_loss_fn=tv_loss_fn,
            gamma=gamma,
            selector_loss_alpha=selector_loss_alpha,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
        )
        if self.confidence_head is None and not with_fore_d:
            return None, loss, metrics

        with torch.no_grad():
            serving_scores = self.candidate_selector.score_candidates(
                unary_logits, hidden, predecessor_ids, candidate_ids
            )
            label = overlap_label(
                targets,
                candidate_ids,
                serving_scores,
                log_normalizer(targets) if normalizer is None else normalizer,
            )

        if self.confidence_head is not None:
            confidence_loss, confidence_metrics = self._confidence_loss(
                hidden,
                predecessor_ids,
                label,
                mask,
                gamma,
                per_position_loss_weight,
                dpace_alpha,
            )
            loss = loss + confidence_head_alpha * confidence_loss
            metrics.update(confidence_metrics)
        if with_fore_d:
            context, anchor_embeddings, _, _ = output[5]
            logits = self.fore_d_predictor(context, anchor_embeddings)
            # Width-conditioned: width w's row, the K columns from its start.
            if fore_d_columns is not None:
                logits = logits.index_select(-1, fore_d_columns)
            # Slot 0 of each block is the anchor; narrow rows arrive padded to K, masked.
            rows = (
                label.view(-1, self.block_size)[:, 1:],
                mask.bool().view(-1, self.block_size)[:, 1:],
            )
            if fore_d_rows is not None:
                rows = tuple(
                    torch.where(fore_d_own, own, narrow)
                    for own, narrow in zip(rows, fore_d_rows, strict=True)
                )
            head_loss, head_metrics = fore_d_loss(logits, *rows)
            loss = loss + fore_d_loss_alpha * head_loss
            metrics.update(head_metrics)
        metrics["loss_sum"] = loss.detach().clone()
        return None, loss, metrics

    @conditional_torch_compile(dynamic=False)
    @torch.no_grad()
    def _narrow_labels(self, width, stride, targets, normalizer, anchors, masks, hidden_states, input_ids, loss_mask,
                       verifier_last_hidden_states, document_ids, position_ids):  # fmt: skip
        """Acceptance labels of width-`width` drafts from every stride-th anchor of the
        step, in the step's [anchors, K] rows (the rest masked): a non-causal drafter
        drafts them anew, in (width + 1)-row blocks. Only the drafted rows are scored,
        against the full blocks' targets and normalizer (no second verifier head).
        The masks come built, so this graph has no break."""
        block = width + 1
        hidden, _, _, mask, indices = self._backbone_forward(
            hidden_states, input_ids, loss_mask, verifier_last_hidden_states, document_ids, position_ids,
            max_anchors=anchors[0].numel(), block_size=block, anchors=anchors, target_logits=targets,
            attention_masks=masks, return_logits=False,
        )[:5]  # fmt: skip
        # Contiguous: on a strided view inductor may lower the vocabulary projection to a
        # bmm over anchors that rereads the whole weight per anchor (0.7 s under FSDP).
        draft = hidden.view(-1, block, hidden.shape[-1])[:, 1:].contiguous()
        unary = self.lm_head(draft)
        candidates = topk_indices(unary, self.candidate_selector.top_k)
        predecessors = self._predecessor_ids(input_ids, indices, block)
        predecessors = predecessors.view(-1, block)[:, 1:]
        scores = self.candidate_selector.score_candidates(
            unary, draft, predecessors, candidates
        )
        full = (-1, self.block_size)
        rows = targets.view(*full, targets.shape[-1])[::stride, 1:block]
        label = overlap_label(
            rows, candidates, scores, normalizer.view(full)[::stride, 1:block]
        )
        labels = label.new_zeros(normalizer.view(full).shape[0], self.block_size - 1)
        valid = torch.zeros_like(labels, dtype=torch.bool)
        labels[::stride, :width] = label
        valid[::stride, :width] = mask.bool().view(-1, block)[:, 1:]
        return labels, valid

    def _confidence_loss(
        self,
        hidden,
        predecessor_ids,
        label,
        mask,
        gamma,
        per_position_loss_weight,
        dpace_alpha,
    ):
        context = self.candidate_selector.context(hidden, predecessor_ids)
        logits = self.confidence_head(confidence_features(hidden, context))  # [1, T]
        label = label.to(logits.dtype)
        position = (
            torch.arange(logits.shape[1], device=logits.device) % self.block_size
        ).unsqueeze(0)
        # The same position weighting as the drafter's terms; it zeroes slot 0.
        if per_position_loss_weight == "dpace":
            decay = partial(
                dpace_loss_decay,
                loss_mask=mask,
                block_size=self.block_size,
                dpace_alpha=dpace_alpha,
            )
        else:
            decay = partial(
                dflash_loss_decay,
                gamma=gamma,
                sample_from_anchor=self.config.sample_from_anchor,
            )
        loss = loss_function(
            logits, label, mask, position,
            loss_fn=partial(binary_cross_entropy_with_logits, reduction="none"), decay_fn=decay,
        )  # fmt: skip
        with torch.no_grad():
            blocks = (-1, self.block_size)
            draft = mask.float().view(blocks)[:, 1:]
            predicted = (logits.float().sigmoid().view(blocks)[:, 1:] * draft).cumprod(
                -1
            )
            observed = (label.float().view(blocks)[:, 1:] * draft).cumprod(-1)
            valid = (draft.sum(-1) > 0).float()
            count = valid.sum().clamp_min(1.0)
            one = torch.ones((), device=logits.device)
            metrics = {
                "confidence_loss_sum": loss.detach().clone(),
                "confidence_loss_total": one,
                "confidence_accept_len_sum": ((predicted.sum(-1) + 1) * valid).sum(),
                "confidence_accept_len_total": count,
                "proposal_accept_len_sum": ((observed.sum(-1) + 1) * valid).sum(),
                "proposal_accept_len_total": count.clone(),
            }
        return loss, metrics

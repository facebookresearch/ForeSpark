# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""What training and serving must agree on: head inputs and the labels they fit.

The heads exist twice -- `nn.Linear` in the trainer, vLLM's `ReplicatedLinear`
in the server -- but their inputs are defined once, here. A mismatch fails
silently: the checkpoint loads and Fore-D ranks the wrong number.

Two heads:
  confidence  post-draft P(draft j accepted | drafts < j accepted), from the
              draft hidden state and the selector's view of the realized
              predecessor token. Drives adaptive verification.
  Fore-D     pre-draft P(draft j accepted | drafts < j accepted) for every
              draft width d = 1..K and position j < d, from the target context
              and the anchor token. Drives Fore-D's draft widths. One
              row per width: the drafter attends bidirectionally inside its
              block, so a width-d draft is not a prefix of the width-K one.
"""

import torch
import torch.nn.functional as F
from torch import nn

# The label needs a full-vocabulary log-normalizer in fp32 (bf16 logsumexp puts
# up to 7.5% error on it), and the whole fp32 tensor does not fit beside a
# training step, so it is computed a slice of positions at a time.
_NORMALIZER_BUDGET_BYTES = 512 << 20


class ForeDHead(nn.Module):
    """Training-side Fore-D head: conditional acceptance logits per anchor."""

    def __init__(self, hidden_size: int, rank: int, num_positions: int) -> None:
        super().__init__()
        self.context_projection = nn.Linear(hidden_size, rank, bias=False)
        self.anchor_projection = nn.Linear(hidden_size, rank, bias=False)
        self.fusion_bias = nn.Parameter(torch.zeros(rank))
        self.output_projection = nn.Linear(rank, num_positions)

    def forward(
        self, context: torch.Tensor, anchor_embeddings: torch.Tensor
    ) -> torch.Tensor:
        # Detached: predicting acceptance must not reshape the drafter.
        fused = (
            self.context_projection(context.detach())
            + self.anchor_projection(anchor_embeddings.detach())
            + self.fusion_bias
        )
        return self.output_projection(F.silu(fused))


def fore_d_loss(
    logits: torch.Tensor,  # [anchors, K]
    labels: torch.Tensor,  # [anchors, K] detached per-position acceptance
    mask: torch.Tensor,  # [anchors, K] bool
) -> tuple[torch.Tensor, dict]:
    """Soft BCE per position, plus survival calibration metrics."""
    labels = labels.detach().to(logits.dtype)
    weight = mask.to(logits.dtype)
    total = weight.sum().clamp_min(1.0)
    loss = (
        F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * weight
    ).sum() / total

    with torch.no_grad():
        predicted = (
            torch.cumprod(torch.where(mask, logits.float().sigmoid(), 1.0), -1) * weight
        )
        observed = torch.cumprod(torch.where(mask, labels.float(), 1.0), -1) * weight
        anchors = (mask.any(-1)).sum().clamp_min(1).float()
        metrics = {
            "fore_d_loss_sum": loss.detach().clone(),
            "fore_d_loss_total": torch.ones((), device=logits.device),
            "fore_d_brier_sum": ((predicted - observed) ** 2 * weight).sum(),
            "fore_d_brier_total": total,
            "fore_d_pred_accept_len_sum": predicted.sum(),
            "fore_d_pred_accept_len_total": anchors,
            "fore_d_target_accept_len_sum": observed.sum(),
            "fore_d_target_accept_len_total": anchors.clone(),
        }
    return loss, metrics


def confidence_input_dim(hidden_size: int, selector_rank: int) -> int:
    return hidden_size + selector_rank


def selector_context(
    projected_hidden: torch.Tensor, predecessor_embedding: torch.Tensor
) -> torch.Tensor:
    """s_j = (W_hidden h_j) * E_pred[x_{j-1}]: the vector the selector scores with.

    DFlash2's analogue of the previous-token embedding DSpark feeds its
    confidence head. Serving rebuilds it after the selector walk, because
    `_score_edges` contracts all top-k predecessors away.
    """
    return projected_hidden * predecessor_embedding.to(projected_hidden.dtype)


def confidence_features(hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    """phi_j = [h_j ; s_j]. The order is part of the checkpoint format."""
    return torch.cat([hidden, context.to(hidden.dtype)], dim=-1)


def overlap_label(
    target_logits: torch.Tensor,  # [..., V]
    candidate_ids: torch.Tensor,  # [..., C] serving top-k
    candidate_scores: torch.Tensor,  # [..., C] selector-corrected scores
    target_log_normalizer: torch.Tensor | None = None,
) -> torch.Tensor:
    """alpha_j = sum_c min(p_target(c), q_draft(c)): the acceptance the heads predict.

    DFlash2's proposal is the softmax over the top-k selector-corrected
    candidates, not its unary distribution, so the overlap is taken over those.
    `candidate_ids` must be the *serving* top-k (not the training set with the
    target swapped in), or the label credits support that does not exist.
    """
    q = torch.softmax(candidate_scores.float(), dim=-1)
    if target_log_normalizer is None:
        target_log_normalizer = log_normalizer(target_logits)
    p = torch.exp(
        target_logits.gather(-1, candidate_ids).float()
        - target_log_normalizer.unsqueeze(-1)
    )
    return torch.minimum(p, q).sum(dim=-1)


def log_normalizer(target_logits: torch.Tensor) -> torch.Tensor:
    """fp32 logsumexp over the vocabulary, a slice of positions at a time."""
    flat = target_logits.reshape(-1, target_logits.shape[-1])
    rows = max(1, _NORMALIZER_BUDGET_BYTES // (4 * flat.shape[-1]))
    parts = [
        torch.logsumexp(flat[i : i + rows].float(), dim=-1)
        for i in range(0, flat.shape[0], rows)
    ]
    return torch.cat(parts).view(target_logits.shape[:-1])

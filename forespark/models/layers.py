# MIT License
#
# Copyright (c) 2026 Z Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Portions adapted from speculators' DFlash2 layers (src/speculators/models/dflash2/model_definitions.py), which
# adapt Z Lab's DFlash2 (https://github.com/z-lab/dflash) under the notice above.

"""A DFlash2 decoder layer whose draft block can be narrower than trained.

The width-conditioned Fore-D head of a non-causal drafter needs labels for
drafts of every width, and a non-causal draft of width d is not a prefix of
the width-K one: training drafts again with (d + 1)-row blocks. The layer's
two grouped convolutions find block boundaries from the row index, so the
block size arrives per forward (`effective_block_size`) rather than by
mutating module state a compiled graph has captured.
"""

import torch
from speculators.models.dflash2.model_definitions import (
    grouped_dynamic_conv,
    GroupedDynamicCausalConv,
    Qwen3DFlash2DecoderLayer,
)


class BlockConv(GroupedDynamicCausalConv):
    def prepare(
        self, hidden_states: torch.Tensor, block_size: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_size = block_size or self.block_size
        kernels = self.kernel_projection(hidden_states).view(
            *hidden_states.shape[:-1], 2, self.kernel_size, self.num_groups
        )
        prepared = grouped_dynamic_conv(
            hidden_states,
            kernels[..., 0, :, :],
            self.base_kernel[0],
            block_size=block_size,
            group_size=self.group_size,
        )
        return prepared, kernels[..., 1, :, :]

    def finish(
        self,
        hidden_states: torch.Tensor,
        delta_kernel: torch.Tensor,
        block_size: int | None = None,
    ) -> torch.Tensor:
        return grouped_dynamic_conv(
            hidden_states, delta_kernel, self.base_kernel[1], block_size=block_size or self.block_size,
            group_size=self.group_size,
        )  # fmt: skip


class DFlash2ConfDecoderLayer(Qwen3DFlash2DecoderLayer):
    def __init__(
        self,
        config,
        layer_idx: int,
        *,
        block_size: int,
        conv_kernel_size: int,
        conv_group_size: int,
    ) -> None:
        super().__init__(
            config,
            layer_idx,
            block_size=block_size,
            conv_kernel_size=conv_kernel_size,
            conv_group_size=conv_group_size,
        )
        conv = {
            "block_size": block_size,
            "kernel_size": conv_kernel_size,
            "group_size": conv_group_size,
        }
        self.attention_conv = BlockConv(config.hidden_size, **conv)
        self.mlp_conv = BlockConv(config.hidden_size, **conv)
        self.block_size = block_size

    def forward(  # noqa: PLR0917
        self,
        target_hidden: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value=None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        effective_block_size: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        block = effective_block_size or self.block_size
        residual = hidden_states
        hidden_states, kernel = self.attention_conv.prepare(
            self.input_layernorm(hidden_states), block
        )
        hidden_states = self.self_attn(
            hidden_states=hidden_states, target_hidden=target_hidden, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_value, output_attentions=output_attentions,
            use_cache=use_cache, cache_position=cache_position, position_embeddings=position_embeddings, **kwargs,
        )[0]  # fmt: skip
        hidden_states = residual + self.attention_conv.finish(
            hidden_states, kernel, block
        )
        residual = hidden_states
        hidden_states, kernel = self.mlp_conv.prepare(
            self.post_attention_layernorm(hidden_states), block
        )
        return residual + self.mlp_conv.finish(self.mlp(hidden_states), kernel, block)

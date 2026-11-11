# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""ForeSpark's grouped conv over Fore-D's packed draft rows, as one kernel per call.

The eager form (upstream's `_grouped_conv` in qwen3_dflash2.py, which
PackedGroupedConv follows) is about eight elementwise kernels per call, four
calls per decoder layer. Under torch.compile Inductor fuses them; uncompiled
(vLLM serves GLM targets with compilation off) they cost ~0.5 ms of GLM-5.3's
six-layer draft step at C = 1, mostly launch gaps. Each product and sum here
is rounded to the input dtype exactly where the eager form rounds it, so the
output matches it bit for bit.
"""

import torch
from vllm.triton_utils import tl, triton

BLOCK_H = 1024


@triton.jit
def _grouped_conv_kernel(
    x_ptr, x_stride, delta_ptr, delta_stride, delta_tap_stride, base_ptr, base_stride,
    position_ptr, out_ptr, out_stride, hidden,
    GROUP: tl.constexpr, TAPS: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    t = tl.program_id(0)
    row = t.to(tl.int64)
    h = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = h < hidden
    dtype = out_ptr.dtype.element_ty
    position = tl.load(position_ptr + t)
    delta = delta_ptr + row * delta_stride + h // GROUP
    x = tl.load(x_ptr + row * x_stride + h, mask=live, other=0.0).to(tl.float32)
    base = tl.load(base_ptr + h, mask=live, other=0.0).to(tl.float32)
    coef = (
        (base + tl.load(delta, mask=live, other=0.0).to(tl.float32))
        .to(dtype)
        .to(tl.float32)
    )
    out = (coef * x).to(dtype).to(tl.float32)
    for tap in tl.static_range(1, TAPS):
        base = tl.load(base_ptr + tap * base_stride + h, mask=live, other=0.0).to(
            tl.float32
        )
        coef = base + tl.load(delta + tap * delta_tap_stride, mask=live, other=0.0).to(
            tl.float32
        )
        coef = coef.to(dtype).to(tl.float32)
        # The eager form's zero padding above row 0, then its in-request mask.
        shifted = tl.load(
            x_ptr + (row - tap) * x_stride + h, mask=live & (t >= tap), other=0.0
        ).to(tl.float32)
        term = (coef * shifted).to(dtype).to(tl.float32)
        term = (term * (position >= tap).to(tl.float32)).to(dtype).to(tl.float32)
        out = (out + term).to(dtype).to(tl.float32)
    tl.store(out_ptr + row * out_stride + h, out.to(dtype), mask=live)


def grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    group_size: int,
    positions: torch.Tensor,
) -> torch.Tensor:
    """`hidden_states` [T, H], `delta` [T, taps, H // group_size] (a strided view
    of the coefficients is fine), `base` [taps, H], `positions` [>= T]: each
    row's position in its request."""
    rows, hidden = hidden_states.shape
    if hidden_states.stride(1) != 1 or delta.stride(2) != 1 or base.stride(1) != 1:
        raise ValueError("grouped_conv needs unit stride along the hidden dimension")
    out = torch.empty(
        rows, hidden, dtype=hidden_states.dtype, device=hidden_states.device
    )
    if rows == 0:
        return out
    _grouped_conv_kernel[(rows, triton.cdiv(hidden, BLOCK_H))](
        hidden_states, hidden_states.stride(0), delta, delta.stride(0), delta.stride(1), base, base.stride(0),
        positions, out, out.stride(0), hidden, GROUP=group_size, TAPS=base.shape[0], BLOCK=BLOCK_H,
        # With fusion Triton contracts products into FMAs past the casts that round them.
        enable_fp_fusion=False,
    )  # fmt: skip
    return out

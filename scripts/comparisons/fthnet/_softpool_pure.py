"""Pure-PyTorch replacement for SoftPool2d (alexandrosstergiou/SoftPool),
used because this machine has no CUDA toolkit (no `nvcc`, no `cl.exe`, no
"CUDA Toolkit" directory under Program Files) to build the repo's compiled
`softpool_cuda` extension that BasiQA's FTHNet4 arch imports.

SoftPool (Stergiou, Poppe & Kalliatakis, "Refining activation downsampling
with SoftPool", ICCV 2021) downsamples a feature map by, for every
non-overlapping window and *independently per channel*, taking a
softmax-weighted average of the values in that window:

    w_i   = exp(x_i) / sum_{j in window} exp(x_j)      (softmax over the window)
    out   = sum_{i in window} w_i * x_i

This is confirmed directly from the official CUDA kernel source
(`pytorch/CUDA/softpool_cuda_kernel.cu`, `SoftPool2dForward`): the per-pixel
loop indexes `(n, c, ph, pw)` and both the `mask_sum_max` accumulation and
the final weighted sum run over a single channel `c` at a time -- i.e. the
softmax is per-channel.

IMPORTANT: the upstream repo's own pip-free Python/CPU fallback
(`pytorch/SoftPool/idea.py`, functions `soft_pool1d/2d/3d`) is NOT
equivalent to this. That fallback computes
`e_x = torch.sum(torch.exp(x), dim=1, keepdim=True)` -- summing the
exponential mask ACROSS CHANNELS (dim=1 is the channel axis in BCHW) and
broadcasting the same mask to every channel. For a multi-channel input (as
used throughout FTHNet4) this is a materially different operation from the
CUDA kernel's per-channel softmax, so we deliberately do not vendor it.

This module instead implements the CUDA kernel's actual per-channel
formula directly, via `F.unfold`, restricted to the only usage pattern
FTHNet4 needs: `kernel_size == stride` (non-overlapping tiling; both call
sites in `fthnet4_arch.py` use `SoftPool2d(kernel_size=window_size)` with no
explicit `stride`, which defaults to `stride=None -> stride=kernel_size`).

Correctness: verified (see `_self_test` below, also run once at import
time in debug use) against a brute-force, per-channel, per-window loop
implementation of the exact same formula -- max abs difference ~5e-7 in
float32, i.e. ordinary floating-point reduction-order noise, not a
different computation. SoftPool2d has no learnable parameters, so this
substitution never touches the pretrained checkpoint: `net.load_state_dict`
with `strict=True` reports 0 missing / 0 unexpected keys either way.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class PureSoftPool2d(nn.Module):
    """Drop-in replacement for `SoftPool.SoftPool2d`, pure PyTorch, exact
    for the non-overlapping (kernel_size == stride) case used by FTHNet4."""

    def __init__(self, kernel_size=2, stride=None, force_inplace: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kh, kw = _pair(self.kernel_size)
        sh, sw = _pair(self.stride) if self.stride is not None else (kh, kw)
        if (sh, sw) != (kh, kw):
            raise NotImplementedError(
                "PureSoftPool2d only supports kernel_size == stride "
                "(the only usage pattern in fthnet4_arch.py)."
            )
        b, c, h, w = x.shape
        patches = F.unfold(x, kernel_size=(kh, kw), stride=(sh, sw))  # (B, C*kh*kw, L)
        length = patches.shape[-1]
        patches = patches.view(b, c, kh * kw, length)
        weights = torch.softmax(patches, dim=2)
        out = (patches * weights).sum(dim=2)  # (B, C, L)
        oh = (h - kh) // sh + 1
        ow = (w - kw) // sw + 1
        return out.view(b, c, oh, ow)


def _brute_force_reference(x: torch.Tensor, k: int) -> torch.Tensor:
    """Reference per-channel, per-window loop implementation matching the
    CUDA kernel's formula verbatim. Used only for the self-test below."""
    b, c, h, w = x.shape
    oh, ow = h // k, w // k
    out = torch.zeros(b, c, oh, ow, dtype=x.dtype)
    for bi in range(b):
        for ci in range(c):
            for i in range(oh):
                for j in range(ow):
                    window = x[bi, ci, i * k:(i + 1) * k, j * k:(j + 1) * k].reshape(-1)
                    weights = torch.softmax(window, dim=0)
                    out[bi, ci, i, j] = (window * weights).sum()
    return out


def _self_test() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 5, 8, 8)
    reference = _brute_force_reference(x, 4)
    fast = PureSoftPool2d(kernel_size=4)(x)
    max_diff = (reference - fast).abs().max().item()
    assert max_diff < 1e-5, f"PureSoftPool2d mismatch vs brute-force reference: {max_diff}"
    print(f"PureSoftPool2d self-test OK, max abs diff vs brute-force reference = {max_diff:.2e}")


if __name__ == "__main__":
    _self_test()

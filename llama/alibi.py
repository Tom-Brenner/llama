from ._base import ModelConfig, MHABase, BlockBase, GPTBase, FFN, RMSNorm
from torch import nn, Tensor
from torch.nn import functional as F

import torch
import math


def get_alibi_slopes(cfg: ModelConfig) -> Tensor:
    """
    Canonical ALiBi slopes.

    Reference construction (commonly used in ALiBi implementations):
    - If n_heads is a power of two: geometric progression with a start determined by n_heads.
    - Otherwise: slopes for the closest lower power of two, then append every other slope from
      the 2x power-of-two set (interleave/extrapolate) until reaching n_heads.
    """

    def _get_slopes_power_of_2(n: int) -> Tensor:
        # start = 2^(-2^(-(log2(n) - 3))) from reference implementations
        start = 2.0 ** (-(2.0 ** (-(math.log2(n) - 3.0))))
        ratio = start
        i = torch.arange(n, dtype=torch.float32)
        start_t = torch.tensor(start, dtype=torch.float32)
        ratio_t = torch.tensor(ratio, dtype=torch.float32)
        return start_t * torch.pow(ratio_t, i)  # (n,) float32

    n_heads = int(cfg.n_heads)
    if n_heads <= 0:
        raise ValueError(f"n_heads must be positive, got {n_heads}.")

    # Power-of-two fast path.
    if (n_heads & (n_heads - 1)) == 0:
        return _get_slopes_power_of_2(n_heads)

    closest_power_of_2 = 2 ** int(math.floor(math.log2(n_heads)))
    base = _get_slopes_power_of_2(closest_power_of_2)  # (closest_power_of_2,)
    # Recurse on 2x power-of-two and take even indices (0,2,4,...) per canonical method.
    extra_all = _get_slopes_power_of_2(2 * closest_power_of_2)  # (2*closest_power_of_2,)
    extra = extra_all[0::2][: (n_heads - closest_power_of_2)]
    return torch.cat([base, extra], dim=0).to(torch.float32)


def get_linear_bias(cfg: ModelConfig) -> Tensor:
    slopes = get_alibi_slopes(cfg).view(cfg.n_heads, 1, 1)  # n_heads,1,1 (float32)
    pos = torch.arange(cfg.ctx_size, dtype=torch.float32)
    distances = pos[None, :] - pos[:, None]  # ctx_size, ctx_size (float32)
    distances = torch.where(distances > 0, torch.zeros_like(distances), distances)
    distances = distances.unsqueeze(0)  # 1, ctx_size, ctx_size
    linear_bias = distances * slopes  # n_heads, ctx_size, ctx_size (float32)
    return linear_bias.unsqueeze(0).to(torch.float32)  # 1, n_heads, ctx_size, ctx_size


class MHA(MHABase):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        linear_bias = get_linear_bias(cfg)
        # Keep reusable ALiBi bias in float32 for stability under AMP.
        self.register_buffer("linear_bias", linear_bias, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        # x -> bsz, seq_len, n_embd
        bsz, seq_len, n_embd = x.shape
        linear_bias = self.linear_bias[:, :, :seq_len, :seq_len]
        qkv: Tensor = self.QKV(x)
        q, k, v = qkv.split(n_embd, 2)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        attn = attn + linear_bias + self.mask[:, :, :seq_len, :seq_len]
        attn = F.softmax(attn, dim=-1)  # bsz, n_heads, seq_len, seq_len
        y = attn @ v  # bsz, n_heads, seq_len, head_dim
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.O(y)


class Block(BlockBase):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.mha = MHA(cfg)  # Override the base mha


class GPT(GPTBase):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        x = self.tok_emb(x)  # bsz, seq_len, n_embd
        for block in self.blocks:
            x = block(x)
        
        return self.lm_head(self.norm(x))
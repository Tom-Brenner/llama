from ._base import ModelConfig, MHABase, BlockBase, GPTBase, FFN, RMSNorm
from torch import nn, Tensor
from torch.nn import functional as F

import torch
import math


def get_freqs_cis(cfg: ModelConfig) -> Tensor:
    head_dim = cfg.n_embd // cfg.n_heads
    if head_dim % 2 != 0:
        raise ValueError(
            f"RoPE requires an even head_dim, but got head_dim={head_dim} "
            f"(n_embd={cfg.n_embd}, n_heads={cfg.n_heads})."
        )
    if cfg.theta is None:
        raise ValueError("RoPE requires cfg.theta to be set (e.g. 10000).")

    half = head_dim // 2
    # Explicit float32 construction to avoid float64/complex128 buffers.
    i = torch.arange(half, dtype=torch.float32)
    theta = torch.tensor(float(cfg.theta), dtype=torch.float32)
    exponent = (-2.0 * i) / float(head_dim)
    thetas = torch.pow(theta, exponent)  # half
    pos = torch.arange(cfg.ctx_size, dtype=torch.float32)  # ctx_size
    freqs = torch.outer(pos, thetas)  # ctx_size, half
    real = torch.cos(freqs)
    imag = torch.sin(freqs)
    return torch.complex(real, imag).to(torch.complex64)


def apply_rot_emb(x: Tensor, freqs: Tensor, pos_offset: int = 0) -> Tensor:
    # x -> bsz, n_heads, seq_len, head_dim; freqs -> ctx_size, head_dim // 2
    bsz, n_heads, seq_len, head_dim = x.shape
    half = head_dim // 2
    f = freqs[pos_offset : pos_offset + seq_len]  # seq_len, half

    # torch.view_as_complex requires float32/float64 real inputs.
    orig_dtype = x.dtype
    x_pairs = x.reshape(bsz, n_heads, seq_len, half, 2).contiguous()
    if x_pairs.dtype not in (torch.float32, torch.float64):
        x_pairs = x_pairs.to(torch.float32)

    x_rot = torch.view_as_complex(x_pairs) * f.view(1, 1, seq_len, half)  # bsz,n_heads,seq_len,half
    x_real = torch.view_as_real(x_rot)  # bsz,n_heads,seq_len,half,2 (float32)
    x_out = x_real.reshape(bsz, n_heads, seq_len, head_dim)
    return x_out.to(orig_dtype)


class MHA(MHABase):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        
        freqs = get_freqs_cis(cfg)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, x: Tensor, pos_offset: int = 0) -> Tensor:
        # x -> bsz, seq_len, n_embd
        bsz, seq_len, n_embd = x.shape
        qkv: Tensor = self.QKV(x)
        q, k, v = qkv.split(n_embd, 2)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rot_emb(q, self.freqs, pos_offset=pos_offset)
        k = apply_rot_emb(k, self.freqs, pos_offset=pos_offset)

        attn = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        attn = attn + self.mask[:, :, :seq_len, :seq_len]
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
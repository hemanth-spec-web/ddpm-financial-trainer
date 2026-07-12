"""
1D U-Net Denoiser for DDPM
============================
Takes a noisy sequence x_t and timestep t, predicts the noise ε
that was added. This is the neural network DDPM trains.

Architecture: encoder-decoder with skip connections, GroupNorm
(not BatchNorm — unstable for diffusion with small batches),
sinusoidal time embeddings injected into every residual block.
"""

import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Encodes timestep t as a sinusoidal embedding, like positional
    encoding in Transformers. Lets the network know 'how noisy' the
    input currently is."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=device) / half_dim
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([args.sin(), args.cos()], dim=-1)
        return embedding


class ResBlock1D(nn.Module):
    """Residual block with GroupNorm and time-embedding injection."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)

        self.act = nn.SiLU()

        self.skip = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # Inject time information — broadcasts over sequence length
        time_bias = self.time_mlp(time_emb)[:, :, None]
        h = h + time_bias

        h = self.act(self.norm2(h))
        h = self.conv2(h)

        return h + self.skip(x)


class SelfAttention1D(nn.Module):
    """Self-attention over the sequence dimension. Used only at the
    bottleneck where the sequence is short enough to be tractable."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) -> (B, L, C) for attention -> back to (B, C, L)
        b, c, l = x.shape
        h = self.norm(x)
        h = h.transpose(1, 2)  # (B, L, C)
        h, _ = self.attn(h, h, h)
        h = h.transpose(1, 2)  # (B, C, L)
        return x + h


class Down(nn.Module):
    """Downsampling block: ResBlock + strided conv to halve sequence length."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.res = ResBlock1D(in_ch, out_ch, time_emb_dim)
        self.downsample = nn.Conv1d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor):
        h = self.res(x, time_emb)
        return self.downsample(h), h  # return skip connection too


class Up(nn.Module):
    """Upsampling block: transposed conv to double sequence length,
    concatenate skip connection, then ResBlock."""

    def __init__(self, in_ch: int, out_ch: int, skip_ch: int, time_emb_dim: int):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.res = ResBlock1D(out_ch + skip_ch, out_ch, time_emb_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor):
        h = self.upsample(x)
        if h.shape[-1] != skip.shape[-1]:
            h = nn.functional.interpolate(h, size=skip.shape[-1])
        h = torch.cat([h, skip], dim=1)
        return self.res(h, time_emb)


class UNet1D(nn.Module):
    """
    Full 1D U-Net for DDPM noise prediction.

    Input:  x_t (B, 1, L) — noisy sequence at timestep t
            t   (B,)      — timestep indices
    Output: predicted noise ε_θ(x_t, t), shape (B, 1, L)
    """

    def __init__(self, d_model: int = 64, time_emb_dim: int = 128):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(d_model),
            nn.Linear(d_model, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Input projection: 1 channel (raw signal) -> d_model channels
        self.input_proj = nn.Conv1d(1, d_model, kernel_size=3, padding=1)

        # Encoder (downsampling path)
        self.down1 = Down(d_model, d_model * 2, time_emb_dim)
        self.down2 = Down(d_model * 2, d_model * 4, time_emb_dim)

        # Bottleneck with self-attention
        self.bottleneck_res1 = ResBlock1D(d_model * 4, d_model * 4, time_emb_dim)
        self.bottleneck_attn = SelfAttention1D(d_model * 4)
        self.bottleneck_res2 = ResBlock1D(d_model * 4, d_model * 4, time_emb_dim)

        # Decoder (upsampling path)
        # up1: bottleneck (d_model*4) -> d_model*2, skip connection has d_model*4 channels (from down2)
        self.up1 = Up(d_model * 4, d_model * 2, skip_ch=d_model * 4, time_emb_dim=time_emb_dim)
        # up2: d_model*2 -> d_model, skip connection has d_model*2 channels (from down1)
        self.up2 = Up(d_model * 2, d_model, skip_ch=d_model * 2, time_emb_dim=time_emb_dim)
        # Output projection: d_model channels -> 1 channel (predicted noise)
        self.output_norm = nn.GroupNorm(min(8, d_model), d_model)
        self.output_proj = nn.Conv1d(d_model, 1, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x expected shape: (B, L) -> add channel dim -> (B, 1, L)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        time_emb = self.time_embedding(t)

        h = self.input_proj(x)

        h, skip1 = self.down1(h, time_emb)
        h, skip2 = self.down2(h, time_emb)

        h = self.bottleneck_res1(h, time_emb)
        h = self.bottleneck_attn(h)
        h = self.bottleneck_res2(h, time_emb)

        h = self.up1(h, skip2, time_emb)
        h = self.up2(h, skip1, time_emb)

        h = self.act(self.output_norm(h))
        out = self.output_proj(h)

        return out.squeeze(1)  # back to (B, L)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
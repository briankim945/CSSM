"""
CSSM-SHViT: SHViT with CSSM replacing single-head attention.

Maintains the hierarchical structure with:
- ConvBlocks in early stages (unchanged)
- CSSM blocks replacing attention in later stages
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from typing import Sequence, Optional

from .cssm import StandardCSSM, GatedCSSM, GatedOpponentCSSM
from .shvit import DropPath, ConvBN, PatchEmbed, Downsample, Mlp, ConvBlock


class CSSMSHViTBlock(nn.Module):
    """
    SHViT block with CSSM replacing single-head attention.

    Matches SHViT architecture with optional VideoRoPE-style position encoding.

    Supports three CSSM types:
    - 'standard': Fixed kernel, no input-dependent gating
    - 'gated': Mamba-style input-dependent gating (recommended)
    - 'opponent': Coupled oscillator with excitation/inhibition
    """
    dim: int
    mlp_ratio: float = 4.0
    drop_path: float = 0.0
    cssm_type: str = 'gated'  # 'standard', 'gated', or 'opponent'
    dense_mixing: bool = False
    block_size: int = 32  # Block size for LMME channel mixing (only with gated + dense_mixing)
    mixing_rank: int = 0  # If > 0, use low-rank mixing (recommended: 4-16)
    gate_activation: str = 'softplus'
    num_timesteps: int = 8
    kernel_size: int = 15  # CSSM spectral kernel size
    spectral_rho: float = 0.999  # Maximum spectral magnitude for stability
    rope_mode: str = 'none'  # 'spatiotemporal', 'temporal', or 'none'
    use_dwconv: bool = True  # DWConv in MLP (matches SHViT)
    output_act: str = 'none'  # Output activation after CSSM: 'gelu', 'silu', or 'none'

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Forward pass supporting both image and video input.

        Args:
            x: Input tensor - either (B, H, W, C) for images or (B, T, H, W, C) for video
            deterministic: If True, disable dropout

        Returns:
            Output tensor with same dimensionality as input
        """
        # Handle both 4D (image) and 5D (video) input
        is_video = x.ndim == 5

        if is_video:
            B, T, H, W, C = x.shape
        else:
            B, H, W, C = x.shape
            T = self.num_timesteps

        # Select CSSM variant
        if self.cssm_type == 'opponent':
            CSSM = GatedOpponentCSSM
        elif self.cssm_type == 'gated':
            CSSM = GatedCSSM
        else:
            CSSM = StandardCSSM

        # CSSM path (replaces attention)
        if is_video:
            # Video input: process each frame through norm, then CSSM on full video
            residual = x  # (B, T, H, W, C)
            # Apply norm per-frame
            x_flat = x.reshape(B * T, H, W, C)
            x_flat = nn.LayerNorm(name='norm1')(x_flat)
            x_video = x_flat.reshape(B, T, H, W, C)
        else:
            # Image input: repeat to create temporal dimension
            residual = x  # (B, H, W, C)
            x = nn.LayerNorm(name='norm1')(x)
            x_video = jnp.repeat(x[:, jnp.newaxis, :, :, :], T, axis=1)

        # Apply CSSM (with optional RoPE inside)
        # Build kwargs - block_size/mixing_rank only for GatedCSSM
        cssm_kwargs = dict(
            channels=self.dim,
            dense_mixing=self.dense_mixing,
            gate_activation=self.gate_activation,
            kernel_size=self.kernel_size,
            spectral_rho=self.spectral_rho,
            rope_mode=self.rope_mode,
        )
        # Only GatedCSSM supports block_size and mixing_rank
        if self.cssm_type == 'gated':
            cssm_kwargs['block_size'] = self.block_size
            cssm_kwargs['mixing_rank'] = self.mixing_rank

        x_video = CSSM(**cssm_kwargs, name='cssm')(x_video)

        # Optional output activation after CSSM (adds nonlinearity like attention's softmax)
        if self.output_act == 'gelu':
            x_video = jax.nn.gelu(x_video)
        elif self.output_act == 'silu':
            x_video = jax.nn.silu(x_video)

        if is_video:
            # Video output: keep full temporal dimension
            x = x_video  # (B, T, H, W, C)
            x = DropPath(self.drop_path)(x, deterministic)
            x = residual + x
        else:
            # Image output: take last timestep
            x = x_video[:, -1]  # (B, H, W, C)
            x = DropPath(self.drop_path)(x, deterministic)
            x = residual + x

        # MLP path
        residual = x
        if is_video:
            # Video: apply norm and MLP per-frame
            B, T, H, W, C = x.shape
            x_flat = x.reshape(B * T, H, W, C)
            x_flat = nn.LayerNorm(name='norm2')(x_flat)
            x_flat = Mlp(
                hidden_dim=int(self.dim * self.mlp_ratio),
                out_dim=self.dim,
                use_dwconv=self.use_dwconv,
                name='mlp'
            )(x_flat, deterministic)
            x = x_flat.reshape(B, T, H, W, C)
        else:
            x = nn.LayerNorm(name='norm2')(x)
            x = Mlp(
                hidden_dim=int(self.dim * self.mlp_ratio),
                out_dim=self.dim,
                use_dwconv=self.use_dwconv,
                name='mlp'
            )(x, deterministic)
        x = DropPath(self.drop_path)(x, deterministic)
        x = residual + x

        return x


class CSSMSHViT(nn.Module):
    """
    CSSM-SHViT: SHViT with CSSM in attention stages.

    Hierarchical 4-stage architecture:
    - ConvBlocks in stages 0-1 (efficient local processing)
    - CSSM blocks in stages 2-3 (temporal recurrence)

    With optional VideoRoPE-style spatiotemporal position encoding.
    Reference: https://arxiv.org/abs/2502.05173
    """
    num_classes: int = 1000
    embed_dims: Sequence[int] = (128, 256, 384, 512)
    depths: Sequence[int] = (1, 2, 4, 1)
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    use_cssm_stages: Sequence[bool] = (False, False, True, True)
    cssm_type: str = 'gated'
    dense_mixing: bool = False
    block_size: int = 32  # Block size for LMME channel mixing (only with gated + dense_mixing)
    mixing_rank: int = 0  # If > 0, use low-rank mixing (recommended: 4-16)
    gate_activation: str = 'softplus'  # 'softplus' for gated, 'sigmoid' for opponent
    num_timesteps: int = 8
    kernel_sizes: Sequence[int] = (15, 15, 5, 3)  # Kernel size per stage (stages 2,3 use CSSM)
    spectral_rho: float = 0.999  # Maximum spectral magnitude for stability
    rope_mode: str = 'none'  # 'spatiotemporal', 'temporal', or 'none'
    use_dwconv: bool = True  # DWConv in MLP (matches SHViT, adds params)
    output_act: str = 'none'  # Output activation after CSSM: 'gelu', 'silu', or 'none'

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        deterministic = not training

        # Handle video input - extract T for variable timesteps, then use last frame
        if x.ndim == 5:
            num_timesteps = x.shape[1]  # Use T from input (enables variable timesteps)
            x = x[:, -1]  # Take last frame for processing (early stages are 4D)
        else:
            num_timesteps = self.num_timesteps

        B = x.shape[0]

        # Patch embedding (4x downsample)
        x = PatchEmbed(self.embed_dims[0], name='patch_embed')(x, deterministic)

        # Stochastic depth
        total_depth = sum(self.depths)
        dp_rates = np.linspace(0, self.drop_path_rate, total_depth)
        dp_idx = 0

        # 4 stages
        for stage_idx in range(4):
            # Downsample between stages
            if stage_idx > 0:
                x = Downsample(self.embed_dims[stage_idx], name=f'downsample{stage_idx}')(x, deterministic)

            # Blocks
            for block_idx in range(self.depths[stage_idx]):
                if self.use_cssm_stages[stage_idx]:
                    x = CSSMSHViTBlock(
                        dim=self.embed_dims[stage_idx],
                        mlp_ratio=self.mlp_ratio,
                        drop_path=float(dp_rates[dp_idx]),
                        cssm_type=self.cssm_type,
                        dense_mixing=self.dense_mixing,
                        block_size=self.block_size,
                        mixing_rank=self.mixing_rank,
                        gate_activation=self.gate_activation,
                        num_timesteps=num_timesteps,  # Use extracted T for variable timesteps
                        kernel_size=self.kernel_sizes[stage_idx],
                        spectral_rho=self.spectral_rho,
                        rope_mode=self.rope_mode,
                        use_dwconv=self.use_dwconv,
                        output_act=self.output_act,
                        name=f'stage{stage_idx}_block{block_idx}'
                    )(x, deterministic)
                else:
                    x = ConvBlock(
                        dim=self.embed_dims[stage_idx],
                        mlp_ratio=self.mlp_ratio,
                        drop_path=float(dp_rates[dp_idx]),
                        name=f'stage{stage_idx}_block{block_idx}'
                    )(x, deterministic)
                dp_idx += 1

        # Final norm
        x = nn.LayerNorm(name='norm')(x)

        # Global max pooling
        x = jnp.max(x, axis=(1, 2))  # (B, C)

        # Classification head
        if not deterministic and self.drop_rate > 0:
            x = nn.Dropout(self.drop_rate)(x, deterministic=False)
        x = nn.Dense(self.num_classes, name='head')(x)

        return x


# Factory functions

def cssm_shvit_s1(num_classes: int = 1000, num_timesteps: int = 8, cssm_type: str = 'gated', **kwargs):
    """CSSM-SHViT-S1 (~6M params + CSSM overhead)."""
    return CSSMSHViT(
        num_classes=num_classes,
        embed_dims=(64, 128, 256, 384),
        depths=(1, 2, 2, 1),
        num_timesteps=num_timesteps,
        cssm_type=cssm_type,
        **kwargs
    )


def cssm_shvit_s2(num_classes: int = 1000, num_timesteps: int = 8, cssm_type: str = 'gated', **kwargs):
    """CSSM-SHViT-S2 (~11M params + CSSM overhead)."""
    return CSSMSHViT(
        num_classes=num_classes,
        embed_dims=(96, 192, 320, 448),
        depths=(1, 2, 3, 1),
        num_timesteps=num_timesteps,
        cssm_type=cssm_type,
        **kwargs
    )


def cssm_shvit_s3(num_classes: int = 1000, num_timesteps: int = 8, cssm_type: str = 'gated', **kwargs):
    """CSSM-SHViT-S3 (~16M params + CSSM overhead)."""
    return CSSMSHViT(
        num_classes=num_classes,
        embed_dims=(112, 224, 352, 480),
        depths=(1, 2, 4, 1),
        num_timesteps=num_timesteps,
        cssm_type=cssm_type,
        **kwargs
    )


def cssm_shvit_s4(num_classes: int = 1000, num_timesteps: int = 8, cssm_type: str = 'gated', **kwargs):
    """CSSM-SHViT-S4 (~22M params + CSSM overhead)."""
    return CSSMSHViT(
        num_classes=num_classes,
        embed_dims=(128, 256, 384, 512),
        depths=(1, 2, 4, 1),
        num_timesteps=num_timesteps,
        cssm_type=cssm_type,
        **kwargs
    )

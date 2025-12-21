"""
CSSM-ViT: Vision Transformer architecture with CSSM replacing attention.

Key design principles:
- Maintains 2D spatial structure (no tokenization/unraveling)
- Pre-norm everywhere (ViT style)
- Separate residual streams for CSSM and MLP
- Optional 2D position embeddings
- Clean MLP with GELU activation

Architecture:
    Input (B, T, H, W, 3)
    → PatchEmbed (keep 2D structure)
    → + Position Embeddings
    → N × CSSMBlock (pre-norm style)
    → LayerNorm → Global Pool → Head
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from typing import Type, Optional, Tuple

from .cssm import StandardCSSM, GatedOpponentCSSM


class DropPath(nn.Module):
    """Stochastic depth (drop path) regularization."""
    drop_prob: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        if self.drop_prob == 0.0 or deterministic:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rng = self.make_rng('dropout')
        mask = jax.random.bernoulli(rng, keep_prob, shape)
        return x * mask / keep_prob


class PatchEmbed(nn.Module):
    """
    Patch embedding that maintains 2D spatial structure.

    Unlike ViT which flattens patches into tokens, this keeps
    the (H', W') spatial dimensions for CSSM to operate on.

    Attributes:
        embed_dim: Output embedding dimension
        patch_size: Size of each patch (square)
    """
    embed_dim: int = 384
    patch_size: int = 16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tensor (B, T, H, W, 3)
        Returns:
            Patches tensor (B, T, H', W', embed_dim) where H'=H/patch_size
        """
        # Patchify with strided conv, maintaining 2D structure
        x = nn.Conv(
            self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding='VALID',
            name='proj'
        )(x)
        return x


class Mlp(nn.Module):
    """
    MLP block using 1x1 convolutions to maintain spatial structure.

    Attributes:
        hidden_dim: Hidden layer dimension (typically 4x input)
        out_dim: Output dimension
    """
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # 1x1 conv expansion
        x = nn.Dense(self.hidden_dim, name='fc1')(x)
        x = nn.gelu(x)
        # 1x1 conv projection
        x = nn.Dense(self.out_dim, name='fc2')(x)
        return x


class CSSMBlock(nn.Module):
    """
    ViT-style block with CSSM instead of attention.

    Structure (pre-norm with layer scale):
        x → LayerNorm → CSSM → γ1 * → DropPath → + x
          → LayerNorm → MLP  → γ2 * → DropPath → + x

    Layer scale (γ initialized near 0) prevents early runaway activations
    and improves training stability, especially for deep networks.

    Attributes:
        dim: Feature dimension
        cssm_cls: CSSM class (StandardCSSM or GatedOpponentCSSM)
        mlp_ratio: MLP hidden dim ratio (typically 4)
        drop_path: Drop path probability
        dense_mixing: CSSM dense mixing flag
        concat_xy: CSSM concat_xy flag
        gate_activation: Gate activation type for GatedOpponentCSSM
        layer_scale_init: Initial value for layer scale (near 0 for stability)
    """
    dim: int
    cssm_cls: Type[nn.Module]
    mlp_ratio: float = 4.0
    drop_path: float = 0.0
    dense_mixing: bool = False
    concat_xy: bool = True
    gate_activation: str = 'sigmoid'
    layer_scale_init: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # CSSM path (pre-norm)
        residual = x
        x = nn.LayerNorm(name='norm1')(x)
        x = self.cssm_cls(
            channels=self.dim,
            dense_mixing=self.dense_mixing,
            concat_xy=self.concat_xy,
            gate_activation=self.gate_activation,
            name='cssm'
        )(x)

        # Layer scale for CSSM branch
        gamma1 = self.param(
            'gamma1',
            nn.initializers.constant(self.layer_scale_init),
            (self.dim,)
        )
        x = x * gamma1

        x = DropPath(self.drop_path)(x, deterministic=not training)
        x = residual + x

        # MLP path (pre-norm)
        residual = x
        x = nn.LayerNorm(name='norm2')(x)
        x = Mlp(
            hidden_dim=int(self.dim * self.mlp_ratio),
            out_dim=self.dim,
            name='mlp'
        )(x)

        # Layer scale for MLP branch
        gamma2 = self.param(
            'gamma2',
            nn.initializers.constant(self.layer_scale_init),
            (self.dim,)
        )
        x = x * gamma2

        x = DropPath(self.drop_path)(x, deterministic=not training)
        x = residual + x

        return x


class CSSMViT(nn.Module):
    """
    Vision Transformer with CSSM replacing self-attention.

    Maintains 2D spatial structure throughout (no tokenization).
    Uses pre-norm style blocks like modern ViTs.

    Attributes:
        num_classes: Number of output classes
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        patch_size: Patch size for stem
        mlp_ratio: MLP expansion ratio
        drop_path_rate: Maximum drop path rate (linear increase)
        cssm_type: 'standard' or 'opponent'
        dense_mixing: CSSM dense mixing flag
        concat_xy: CSSM concat_xy flag
        use_pos_embed: Whether to use position embeddings
    """
    num_classes: int = 10
    embed_dim: int = 384
    depth: int = 12
    patch_size: int = 16
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.1
    cssm_type: str = 'opponent'
    dense_mixing: bool = False
    concat_xy: bool = True
    gate_activation: str = 'sigmoid'
    use_pos_embed: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x: Input video tensor (B, T, H, W, 3)
            training: Training mode flag

        Returns:
            Logits tensor (B, num_classes)
        """
        B, T, H, W, C = x.shape

        # Select CSSM class
        if self.cssm_type == 'standard':
            CSSM = StandardCSSM
        else:
            CSSM = GatedOpponentCSSM

        # Patch embedding (maintains 2D structure)
        x = PatchEmbed(
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
            name='patch_embed'
        )(x)  # (B, T, H', W', embed_dim)

        _, _, H_p, W_p, _ = x.shape

        # Position embeddings (2D spatial, broadcast over time)
        if self.use_pos_embed:
            pos_embed = self.param(
                'pos_embed',
                nn.initializers.normal(0.02),
                (1, 1, H_p, W_p, self.embed_dim)
            )
            x = x + pos_embed  # Broadcast over B and T

        # Transformer blocks with linear drop path increase
        dp_rates = np.linspace(0, self.drop_path_rate, self.depth)

        for i in range(self.depth):
            x = CSSMBlock(
                dim=self.embed_dim,
                cssm_cls=CSSM,
                mlp_ratio=self.mlp_ratio,
                drop_path=float(dp_rates[i]),
                dense_mixing=self.dense_mixing,
                concat_xy=self.concat_xy,
                gate_activation=self.gate_activation,
                name=f'block{i}'
            )(x, training=training)

        # Final norm
        x = nn.LayerNorm(name='norm')(x)

        # Use last timestep (after full recurrence), then spatial max pool
        x = x[:, -1]  # (B, H', W', embed_dim) - final timestep
        x = jnp.max(x, axis=(1, 2))  # (B, embed_dim) - spatial max

        # Classification head
        x = nn.Dense(self.num_classes, name='head')(x)

        return x


# Model configurations (similar to ViT-S, ViT-B, ViT-L)
def cssm_vit_tiny(**kwargs):
    """CSSM-ViT-Tiny: ~6M params"""
    return CSSMViT(embed_dim=192, depth=12, **kwargs)

def cssm_vit_small(**kwargs):
    """CSSM-ViT-Small: ~22M params"""
    return CSSMViT(embed_dim=384, depth=12, **kwargs)

def cssm_vit_base(**kwargs):
    """CSSM-ViT-Base: ~86M params"""
    return CSSMViT(embed_dim=768, depth=12, **kwargs)

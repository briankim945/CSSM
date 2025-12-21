"""
Baseline ViT for comparison with CSSM-ViT.

Uses standard multi-head self-attention instead of CSSM, with the same:
- Patch embedding (maintains 2D structure)
- Pre-norm block structure
- Layer scale
- MLP design
- Position embeddings

Supports factorized space-time attention for video:
- Spatial attention: per-frame attention over H'*W' tokens
- Temporal attention: per-position attention over T frames
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from typing import Optional


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
    """Patch embedding that maintains 2D spatial structure."""
    embed_dim: int = 384
    patch_size: int = 16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tensor (B, T, H, W, 3)
        Returns:
            Patches tensor (B, T, H', W', embed_dim)
        """
        x = nn.Conv(
            self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding='VALID',
            name='proj'
        )(x)
        return x


class Mlp(nn.Module):
    """MLP block with GELU activation."""
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, name='fc1')(x)
        x = nn.gelu(x)
        x = nn.Dense(self.out_dim, name='fc2')(x)
        return x


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head self-attention.

    Attributes:
        dim: Input/output dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in QKV projection
    """
    dim: int
    num_heads: int = 8
    qkv_bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tensor (B, N, C) where N is sequence length
        Returns:
            Output tensor (B, N, C)
        """
        B, N, C = x.shape
        head_dim = C // self.num_heads
        scale = head_dim ** -0.5

        # QKV projection
        qkv = nn.Dense(3 * C, use_bias=self.qkv_bias, name='qkv')(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale  # (B, num_heads, N, N)
        attn = nn.softmax(attn, axis=-1)

        # Output
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = nn.Dense(C, name='proj')(x)
        return x


class SpatialAttentionBlock(nn.Module):
    """
    ViT-style block with spatial attention (per-frame).

    Structure (pre-norm with layer scale):
        x → LayerNorm → SpatialAttn → γ1 * → DropPath → + x
          → LayerNorm → MLP        → γ2 * → DropPath → + x

    Attributes:
        dim: Feature dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        drop_path: Drop path probability
        layer_scale_init: Initial value for layer scale
    """
    dim: int
    num_heads: int = 8
    mlp_ratio: float = 4.0
    drop_path: float = 0.0
    layer_scale_init: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Args:
            x: Input tensor (B, T, H, W, C)
        Returns:
            Output tensor (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape

        # Flatten spatial dims for attention: (B, T, H, W, C) -> (B*T, H*W, C)
        x_flat = x.reshape(B * T, H * W, C)

        # Spatial attention path (pre-norm)
        residual = x_flat
        x_flat = nn.LayerNorm(name='norm1')(x_flat)
        x_flat = MultiHeadAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            name='attn'
        )(x_flat)

        # Layer scale
        gamma1 = self.param(
            'gamma1',
            nn.initializers.constant(self.layer_scale_init),
            (self.dim,)
        )
        x_flat = x_flat * gamma1

        x_flat = DropPath(self.drop_path)(x_flat, deterministic=not training)
        x_flat = residual + x_flat

        # MLP path (pre-norm)
        residual = x_flat
        x_flat = nn.LayerNorm(name='norm2')(x_flat)
        x_flat = Mlp(
            hidden_dim=int(self.dim * self.mlp_ratio),
            out_dim=self.dim,
            name='mlp'
        )(x_flat)

        # Layer scale
        gamma2 = self.param(
            'gamma2',
            nn.initializers.constant(self.layer_scale_init),
            (self.dim,)
        )
        x_flat = x_flat * gamma2

        x_flat = DropPath(self.drop_path)(x_flat, deterministic=not training)
        x_flat = residual + x_flat

        # Reshape back to (B, T, H, W, C)
        return x_flat.reshape(B, T, H, W, C)


class TemporalAttentionBlock(nn.Module):
    """
    Temporal attention block (per-spatial-position).

    Applies attention over the temporal dimension at each spatial location.

    Attributes:
        dim: Feature dimension
        num_heads: Number of attention heads
        drop_path: Drop path probability
        layer_scale_init: Initial value for layer scale
    """
    dim: int
    num_heads: int = 8
    drop_path: float = 0.0
    layer_scale_init: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Args:
            x: Input tensor (B, T, H, W, C)
        Returns:
            Output tensor (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape

        # Reshape for temporal attention: (B, T, H, W, C) -> (B*H*W, T, C)
        x_temp = x.transpose(0, 2, 3, 1, 4).reshape(B * H * W, T, C)

        # Temporal attention (pre-norm)
        residual = x_temp
        x_temp = nn.LayerNorm(name='norm')(x_temp)
        x_temp = MultiHeadAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            name='attn'
        )(x_temp)

        # Layer scale
        gamma = self.param(
            'gamma',
            nn.initializers.constant(self.layer_scale_init),
            (self.dim,)
        )
        x_temp = x_temp * gamma

        x_temp = DropPath(self.drop_path)(x_temp, deterministic=not training)
        x_temp = residual + x_temp

        # Reshape back: (B*H*W, T, C) -> (B, T, H, W, C)
        return x_temp.reshape(B, H, W, T, C).transpose(0, 3, 1, 2, 4)


class BaselineViT(nn.Module):
    """
    Baseline Vision Transformer for video classification.

    Matches CSSM-ViT architecture but uses standard attention:
    - Spatial attention (per-frame) replaces CSSM spatial mixing
    - Optional temporal attention for temporal modeling

    Attributes:
        num_classes: Number of output classes
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        patch_size: Patch size for stem
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        drop_path_rate: Maximum drop path rate
        use_temporal_attn: Whether to use temporal attention blocks
        temporal_attn_every: Add temporal attention every N spatial blocks
        use_pos_embed: Whether to use position embeddings
    """
    num_classes: int = 10
    embed_dim: int = 384
    depth: int = 12
    patch_size: int = 16
    num_heads: int = 6
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.1
    use_temporal_attn: bool = True
    temporal_attn_every: int = 3
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

        # Patch embedding
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
            x = x + pos_embed

        # Transformer blocks with linear drop path increase
        dp_rates = np.linspace(0, self.drop_path_rate, self.depth)

        for i in range(self.depth):
            # Spatial attention block
            x = SpatialAttentionBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                drop_path=float(dp_rates[i]),
                name=f'block{i}'
            )(x, training=training)

            # Temporal attention (every N blocks)
            if self.use_temporal_attn and (i + 1) % self.temporal_attn_every == 0:
                x = TemporalAttentionBlock(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    drop_path=float(dp_rates[i]),
                    name=f'temporal_block{i}'
                )(x, training=training)

        # Final norm
        x = nn.LayerNorm(name='norm')(x)

        # Global average pooling over T, H', W'
        x = jnp.mean(x, axis=(1, 2, 3))  # (B, embed_dim)

        # Classification head
        x = nn.Dense(self.num_classes, name='head')(x)

        return x


# Model configurations matching CSSM-ViT variants
def baseline_vit_tiny(**kwargs):
    """Baseline-ViT-Tiny: comparable to CSSM-ViT-Tiny"""
    return BaselineViT(embed_dim=192, depth=12, num_heads=3, **kwargs)

def baseline_vit_small(**kwargs):
    """Baseline-ViT-Small: comparable to CSSM-ViT-Small"""
    return BaselineViT(embed_dim=384, depth=12, num_heads=6, **kwargs)

def baseline_vit_base(**kwargs):
    """Baseline-ViT-Base: comparable to CSSM-ViT-Base"""
    return BaselineViT(embed_dim=768, depth=12, num_heads=12, **kwargs)

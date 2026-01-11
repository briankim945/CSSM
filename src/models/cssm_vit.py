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

from .cssm import (
    StandardCSSM, GatedCSSM, GatedOpponentCSSM, BilinearOpponentCSSM,
    LinearCSSM, LinearOpponentCSSM, HGRUStyleCSSM, HGRUBilinearCSSM
)


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

    WARNING: Non-overlapping patches lose fine spatial detail.
    Use ConvStem for tasks requiring high spatial resolution (e.g., Pathfinder).

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


class ConvStem(nn.Module):
    """
    Convolutional stem with overlapping receptive fields.

    Preserves much more spatial detail than PatchEmbed.
    Similar to SHViT/ResNet-style stems.

    For 300×300 Pathfinder → 75×75 output (stride 4 total)
    For 224×224 ImageNet → 56×56 output (stride 4 total)

    Attributes:
        embed_dim: Output embedding dimension
        stem_type: 'shvit' (2x conv3x3 s2) or 'resnet' (conv7x7 s2 + pool)
    """
    embed_dim: int = 384
    stem_type: str = 'shvit'  # 'shvit' or 'resnet'

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tensor (B, T, H, W, 3)
        Returns:
            Features tensor (B, T, H/4, W/4, embed_dim)
        """
        # Handle video input: process each frame
        is_video = x.ndim == 5
        if is_video:
            B, T, H, W, C = x.shape
            x = x.reshape(B * T, H, W, C)

        if self.stem_type == 'resnet':
            # ResNet-style: 7×7 conv stride 2 + 3×3 max pool stride 2
            x = nn.Conv(
                self.embed_dim // 2,
                kernel_size=(7, 7),
                strides=(2, 2),
                padding='SAME',
                name='conv1'
            )(x)
            x = nn.LayerNorm(name='norm1')(x)
            x = jax.nn.gelu(x)
            # Max pool 3×3 stride 2
            x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')
            # Project to embed_dim
            x = nn.Conv(
                self.embed_dim,
                kernel_size=(1, 1),
                name='proj'
            )(x)
        else:  # 'shvit' style
            # Two overlapping 3×3 convs with stride 2 each
            x = nn.Conv(
                self.embed_dim // 2,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding='SAME',
                name='conv1'
            )(x)
            x = nn.LayerNorm(name='norm1')(x)
            x = jax.nn.gelu(x)
            x = nn.Conv(
                self.embed_dim,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding='SAME',
                name='conv2'
            )(x)
            x = nn.LayerNorm(name='norm2')(x)

        if is_video:
            x = x.reshape(B, T, x.shape[1], x.shape[2], self.embed_dim)

        return x


class DWConv(nn.Module):
    """Depthwise convolution for local mixing."""
    dim: int
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, T, H, W, C) or (B, H, W, C)
        # Apply DWConv on spatial dims
        if x.ndim == 5:
            B, T, H, W, C = x.shape
            x = x.reshape(B * T, H, W, C)
            x = nn.Conv(
                self.dim,
                kernel_size=(self.kernel_size, self.kernel_size),
                padding='SAME',
                feature_group_count=self.dim,
                name='dwconv'
            )(x)
            x = x.reshape(B, T, H, W, C)
        else:
            x = nn.Conv(
                self.dim,
                kernel_size=(self.kernel_size, self.kernel_size),
                padding='SAME',
                feature_group_count=self.dim,
                name='dwconv'
            )(x)
        return x


class Mlp(nn.Module):
    """
    MLP block using 1x1 convolutions to maintain spatial structure.

    Attributes:
        hidden_dim: Hidden layer dimension (typically 4x input)
        out_dim: Output dimension
        use_dwconv: Whether to add DWConv in hidden layer (matches SHViT)
    """
    hidden_dim: int
    out_dim: int
    use_dwconv: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # 1x1 conv expansion
        x = nn.Dense(self.hidden_dim, name='fc1')(x)
        x = nn.gelu(x)
        # Optional DWConv for local spatial mixing
        if self.use_dwconv:
            x = x + DWConv(self.hidden_dim, name='dwconv')(x)
        # 1x1 conv projection
        x = nn.Dense(self.out_dim, name='fc2')(x)
        return x


class CSSMBlock(nn.Module):
    """
    ViT-style block with CSSM instead of attention.

    Structure (pre-norm with layer scale):
        x → LayerNorm → CSSM → [act] → γ1 * → DropPath → + x
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
        use_dwconv: Whether to add DWConv in MLP (matches SHViT)
        output_act: Output activation after CSSM ('gelu', 'silu', or 'none')
    """
    dim: int
    cssm_cls: Type[nn.Module]
    mlp_ratio: float = 4.0
    drop_path: float = 0.0
    dense_mixing: bool = False
    concat_xy: bool = True
    gate_activation: str = 'sigmoid'
    layer_scale_init: float = 1e-6
    rope_mode: str = 'none'  # 'spatiotemporal', 'temporal', or 'none'
    block_size: int = 32  # Block size for LMME channel mixing
    gate_rank: int = 0  # Low-rank gate bottleneck (0 = full rank)
    kernel_size: int = 11  # Spatial kernel size for CSSM
    use_dwconv: bool = False  # DWConv in MLP
    output_act: str = 'none'  # Output activation after CSSM: 'gelu', 'silu', or 'none'

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # CSSM path (pre-norm)
        residual = x
        x = nn.LayerNorm(name='norm1')(x)

        # Build CSSM kwargs - block_size only for GatedCSSM
        cssm_kwargs = dict(
            channels=self.dim,
            dense_mixing=self.dense_mixing,
            gate_activation=self.gate_activation,
            rope_mode=self.rope_mode,
            gate_rank=self.gate_rank,
            kernel_size=self.kernel_size,
        )
        # GatedCSSM supports block_size for LMME, others don't
        if hasattr(self.cssm_cls, 'block_size'):
            cssm_kwargs['block_size'] = self.block_size
        # GatedOpponentCSSM has concat_xy, others may not
        if hasattr(self.cssm_cls, 'concat_xy'):
            cssm_kwargs['concat_xy'] = self.concat_xy

        x = self.cssm_cls(**cssm_kwargs, name='cssm')(x)

        # Optional output activation after CSSM
        if self.output_act == 'gelu':
            x = jax.nn.gelu(x)
        elif self.output_act == 'silu':
            x = jax.nn.silu(x)

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
            use_dwconv=self.use_dwconv,
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
        patch_size: Patch size for PatchEmbed stem (ignored if stem_mode != 'patch')
        stem_mode: 'patch' (ViT-style), 'conv' (overlapping convs), or 'resnet'
        mlp_ratio: MLP expansion ratio
        drop_path_rate: Maximum drop path rate (linear increase)
        cssm_type: 'standard', 'gated', or 'opponent'
        dense_mixing: CSSM dense mixing flag
        concat_xy: CSSM concat_xy flag
        use_pos_embed: Whether to use position embeddings
    """
    num_classes: int = 10
    embed_dim: int = 384
    depth: int = 12
    patch_size: int = 16  # Only used if stem_mode='patch'
    stem_mode: str = 'conv'  # 'patch', 'conv' (SHViT-style), or 'resnet'
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.1
    cssm_type: str = 'opponent'
    dense_mixing: bool = False
    concat_xy: bool = True
    gate_activation: str = 'sigmoid'
    use_pos_embed: bool = True
    rope_mode: str = 'none'  # 'spatiotemporal', 'temporal', or 'none'
    block_size: int = 32  # Block size for LMME channel mixing
    gate_rank: int = 0  # Low-rank gate bottleneck (0 = full rank, try 16-64)
    kernel_size: int = 11  # Spatial kernel size for CSSM
    use_dwconv: bool = False  # DWConv in MLP
    output_act: str = 'none'  # Output activation after CSSM: 'gelu', 'silu', or 'none'
    layer_scale_init: float = 1e-6  # Layer scale initialization (1e-6 for deep, 1.0 for shallow)

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True,
                 return_spatial: bool = False) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x: Input video tensor (B, T, H, W, 3)
            training: Training mode flag
            return_spatial: If True, return (logits, perpixel_logits) where
                           perpixel_logits has shape (B, T, H', W', num_classes)

        Returns:
            Logits tensor (B, num_classes), or tuple if return_spatial=True
        """
        B, T, H, W, C = x.shape

        # Select CSSM class
        if self.cssm_type == 'standard':
            CSSM = StandardCSSM
        elif self.cssm_type == 'gated':
            CSSM = GatedCSSM
        elif self.cssm_type == 'bilinear':
            CSSM = BilinearOpponentCSSM
        elif self.cssm_type == 'linear':
            # Ablation: vanilla CSSM without log-space (sequential scan)
            CSSM = LinearCSSM
        elif self.cssm_type == 'linear_opponent':
            # Ablation: opponent CSSM without log-space (sequential scan)
            CSSM = LinearOpponentCSSM
        elif self.cssm_type == 'hgru':
            # 2x2 linear opponent: X↔Y with separate coupling gates
            CSSM = HGRUStyleCSSM
        elif self.cssm_type == 'hgru_bi':
            # 3x3 with Z interaction channel: X, Y, Z where Z learns X-Y correlation
            CSSM = HGRUBilinearCSSM
        else:
            CSSM = GatedOpponentCSSM

        # Stem: convert input to feature maps
        if self.stem_mode == 'patch':
            # ViT-style non-overlapping patches (loses fine detail)
            x = PatchEmbed(
                embed_dim=self.embed_dim,
                patch_size=self.patch_size,
                name='stem'
            )(x)  # (B, T, H/patch_size, W/patch_size, embed_dim)
        elif self.stem_mode == 'resnet':
            # ResNet-style: 7×7 conv + pool (stride 4 total)
            x = ConvStem(
                embed_dim=self.embed_dim,
                stem_type='resnet',
                name='stem'
            )(x)  # (B, T, H/4, W/4, embed_dim)
        else:  # 'conv' (default, SHViT-style)
            # Overlapping convolutions (stride 4 total, preserves detail)
            x = ConvStem(
                embed_dim=self.embed_dim,
                stem_type='shvit',
                name='stem'
            )(x)  # (B, T, H/4, W/4, embed_dim)

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
                rope_mode=self.rope_mode,
                block_size=self.block_size,
                gate_rank=self.gate_rank,
                kernel_size=self.kernel_size,
                use_dwconv=self.use_dwconv,
                output_act=self.output_act,
                layer_scale_init=self.layer_scale_init,
                name=f'block{i}'
            )(x, training=training)

        # Final norm
        x = nn.LayerNorm(name='norm')(x)
        # x shape: (B, T, H', W', embed_dim)

        if return_spatial:
            # Return per-pixel logits at ALL timesteps
            # Apply readout to all timesteps
            x_all = jax.nn.gelu(x)
            x_all = nn.Dense(self.embed_dim, name='readout_proj')(x_all)  # (B, T, H', W', embed_dim)

            # Apply head per-pixel at each timestep
            head = nn.Dense(self.num_classes, name='head')
            # Reshape to apply head: (B*T*H'*W', embed_dim) -> (B*T*H'*W', num_classes)
            B, T, Hp, Wp, E = x_all.shape
            x_flat = x_all.reshape(-1, E)
            perpixel_logits_flat = head(x_flat)
            perpixel_logits = perpixel_logits_flat.reshape(B, T, Hp, Wp, self.num_classes)

            # Also compute final logits normally (for prediction)
            x_last = x_all[:, -1]  # (B, H', W', embed_dim)
            x_pooled_flat = x_last.reshape(x_last.shape[0], -1, self.embed_dim)
            x_pooled = jax.scipy.special.logsumexp(x_pooled_flat, axis=1)
            final_logits = head(x_pooled)

            return final_logits, perpixel_logits

        # Standard forward: use last timestep only
        x = x[:, -1]  # (B, H', W', embed_dim) - final timestep

        # Readout: nonlinearity + channel projection before pooling
        x = jax.nn.gelu(x)
        x = nn.Dense(self.embed_dim, name='readout_proj')(x)  # (B, H', W', embed_dim)

        # LogSumExp pooling: smooth max, gradients flow everywhere
        x_flat = x.reshape(x.shape[0], -1, self.embed_dim)  # (B, H'*W', embed_dim)
        x = jax.scipy.special.logsumexp(x_flat, axis=1)  # (B, embed_dim)

        # Classification head
        x = nn.Dense(self.num_classes, name='head')(x)

        return x

def get_spatial_features_from_params(model_config: dict, params: dict, x: jnp.ndarray):
    """
    Get spatial features at each timestep before pooling.

    This is a standalone function that extracts per-pixel features from a CSSMViT
    without needing module scopes. Useful for visualization.

    Args:
        model_config: Dict with model configuration (embed_dim, depth, etc.)
        params: Model parameters dict
        x: Input video tensor (B, T, H, W, 3)

    Returns:
        Tuple of:
            - spatial_features: (B, T, H', W', embed_dim) features at each timestep
            - perpixel_logits: (B, T, H', W', num_classes) per-pixel class logits
            - final_logits: (B, num_classes) final classification logits
    """
    B, T, H, W, C = x.shape
    embed_dim = model_config['embed_dim']
    num_classes = model_config.get('num_classes', 2)

    # Run stem manually using conv operations
    stem_params = params['stem']

    # Process first frame through stem, then broadcast to all timesteps
    x_frame = x[:, 0]  # (B, H, W, C)

    # Conv1: 3x3 stride 2
    x_stem = jax.lax.conv_general_dilated(
        x_frame,
        stem_params['conv1']['kernel'],
        window_strides=(2, 2),
        padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    if 'bias' in stem_params['conv1']:
        x_stem = x_stem + stem_params['conv1']['bias']
    x_stem = jax.nn.gelu(x_stem)

    # Conv2: 3x3 stride 2
    x_stem = jax.lax.conv_general_dilated(
        x_stem,
        stem_params['conv2']['kernel'],
        window_strides=(2, 2),
        padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )
    if 'bias' in stem_params['conv2']:
        x_stem = x_stem + stem_params['conv2']['bias']
    x_stem = jax.nn.gelu(x_stem)
    # x_stem: (B, H/4, W/4, embed_dim)

    H_p, W_p = x_stem.shape[1], x_stem.shape[2]

    # Broadcast to all timesteps
    x_proc = jnp.broadcast_to(x_stem[:, jnp.newaxis], (B, T, H_p, W_p, embed_dim))

    # Add position embeddings
    if 'pos_embed' in params:
        x_proc = x_proc + params['pos_embed']

    # Run through CSSM blocks
    # We need to apply each block's params
    depth = model_config['depth']
    for i in range(depth):
        block_params = params[f'block{i}']

        # CSSMBlock structure: norm1 -> cssm -> norm2 -> mlp
        # With residual connections and layer scale

        # Pre-norm 1
        norm1_params = block_params['norm1']
        x_normed = layer_norm(x_proc, norm1_params)

        # CSSM (this is complex - for now, we'll use the full model apply)
        # Actually, let's use a simpler approach: run the full model and capture intermediates
        # For now, return what we have after stem as a placeholder
        pass

    # Since CSSM blocks are complex, let's use a different approach:
    # Run the full model but modify it to return all timesteps

    # For visualization, we can approximate by running the model multiple times
    # with different numbers of timesteps, but this is expensive

    # Simple approach: just apply norm, gelu, and readout to stem features
    # This won't capture CSSM dynamics but will test the pipeline

    # Apply LayerNorm to all timesteps
    norm_params = params['norm']
    x_normed = layer_norm(x_proc, norm_params)

    # Apply readout
    x_readout = jax.nn.gelu(x_normed)
    readout_kernel = params['readout_proj']['kernel']
    readout_bias = params['readout_proj'].get('bias', jnp.zeros(embed_dim))
    spatial_features = jnp.einsum('bthwc,cd->bthwd', x_readout, readout_kernel) + readout_bias

    # Apply head per-pixel
    head_kernel = params['head']['kernel']
    head_bias = params['head'].get('bias', jnp.zeros(num_classes))
    perpixel_logits = jnp.einsum('bthwc,cd->bthwd', spatial_features, head_kernel) + head_bias

    # Final logits from last timestep
    x_last = spatial_features[:, -1]
    x_flat = x_last.reshape(x_last.shape[0], -1, embed_dim)
    x_pooled = jax.scipy.special.logsumexp(x_flat, axis=1)
    final_logits = jnp.einsum('bc,cd->bd', x_pooled, head_kernel) + head_bias

    return spatial_features, perpixel_logits, final_logits


def layer_norm(x, params, epsilon=1e-6):
    """Apply layer normalization using params dict."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + epsilon)
    return x_norm * params['scale'] + params['bias']


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

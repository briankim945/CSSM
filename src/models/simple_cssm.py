"""
SimpleCSSM: Clean architecture for plugging different CSSM variants.

Architecture:
    Conv -> act -> norm -> maxpool
    Conv -> act -> norm -> maxpool
    + Position Embeddings (spatiotemporal RoPE by default)
    CSSM block(s)
    Frame selection (last or all)
    Norm -> act -> spatial pool -> norm -> head
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional

from .cssm import GatedCSSM, HGRUBilinearCSSM, TransformerCSSM, apply_rope


# Registry of CSSM variants
CSSM_REGISTRY = {
    'gated': GatedCSSM,
    'hgru_bi': HGRUBilinearCSSM,          # Original 3x3 with E/I kernels
    'transformer': TransformerCSSM,       # Minimal Q/K/A
}


class SimpleCSSM(nn.Module):
    """
    Simple CSSM architecture with clean stem and readout.

    Attributes:
        num_classes: Number of output classes (2 for Pathfinder)
        embed_dim: Embedding dimension after stem
        depth: Number of CSSM blocks
        cssm_type: Which CSSM variant to use
        kernel_size: CSSM spatial kernel size
        frame_readout: 'last' (single frame) or 'all' (spatiotemporal pool)
        norm_type: 'layer' or 'batch'
        pos_embed: Position embedding type
        act_type: Nonlinearity type (softplus, gelu, relu)
        pool_type: Final pooling type (mean, max)
        seq_len: Number of temporal recurrence steps
    """
    num_classes: int = 2
    embed_dim: int = 32
    depth: int = 1
    cssm_type: str = 'hgru_bi'
    kernel_size: int = 15
    block_size: int = 1          # Channel mixing block size (1=depthwise, >1=block mixing)
    frame_readout: str = 'last'  # 'last' or 'all'
    norm_type: str = 'layer'     # 'layer' or 'batch'
    pos_embed: str = 'spatiotemporal'  # 'spatiotemporal', 'temporal', 'learnable', 'none'
    act_type: str = 'softplus'   # 'softplus', 'gelu', 'relu'
    pool_type: str = 'mean'      # 'mean' or 'max'
    seq_len: int = 8             # Temporal sequence length

    def _get_act(self):
        """Get activation function."""
        if self.act_type == 'softplus':
            return nn.softplus
        elif self.act_type == 'gelu':
            return jax.nn.gelu
        elif self.act_type == 'relu':
            return jax.nn.relu
        else:
            return nn.softplus

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x: Input (B, T, H, W, C) or (B, H, W, C)
        Returns:
            Logits (B, num_classes)
        """
        act = self._get_act()

        # Handle single image input - repeat to create temporal sequence
        if x.ndim == 4:
            x = jnp.repeat(x[:, None, ...], self.seq_len, axis=1)  # (B, T, H, W, C)

        B, T, H, W, C = x.shape

        # === STEM: 2x Conv -> act -> norm -> maxpool ===
        # Reshape for 2D convs: (B*T, H, W, C)
        x = x.reshape(B * T, H, W, C)

        # Conv block 1: Conv(embed_dim) -> act -> norm -> maxpool
        x = nn.Conv(self.embed_dim, kernel_size=(3, 3), strides=(1, 1), padding='SAME', name='conv1')(x)
        x = act(x)
        if self.norm_type == 'batch':
            x = nn.BatchNorm(use_running_average=not training, name='norm1')(x)
        else:
            x = nn.LayerNorm(name='norm1')(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')

        # Conv block 2: Conv(embed_dim) -> act -> norm -> maxpool
        x = nn.Conv(self.embed_dim, kernel_size=(3, 3), strides=(1, 1), padding='SAME', name='conv2')(x)
        x = act(x)
        if self.norm_type == 'batch':
            x = nn.BatchNorm(use_running_average=not training, name='norm2')(x)
        else:
            x = nn.LayerNorm(name='norm2')(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')

        # Reshape back: (B, T, H', W', embed_dim)
        _, H_new, W_new, _ = x.shape
        x = x.reshape(B, T, H_new, W_new, self.embed_dim)

        # === POSITION EMBEDDINGS (learnable 2D spatial, if enabled) ===
        if self.pos_embed == 'learnable':
            pos = self.param('pos_embed', nn.initializers.normal(0.02),
                           (1, 1, H_new, W_new, self.embed_dim))
            x = x + pos
        # Note: RoPE ('spatiotemporal', 'temporal') is applied INSIDE CSSM via rope_mode

        # === CSSM BLOCK(S) ===
        # Determine rope_mode to pass to CSSM
        rope_mode = self.pos_embed if self.pos_embed in ['spatiotemporal', 'temporal'] else 'none'

        CSSMClass = CSSM_REGISTRY.get(self.cssm_type, HGRUBilinearCSSM)
        for i in range(self.depth):
            cssm = CSSMClass(
                channels=self.embed_dim,
                kernel_size=self.kernel_size,
                block_size=self.block_size,
                rope_mode=rope_mode,  # Spatiotemporal RoPE inside CSSM
                name=f'cssm_{i}'
            )
            x = x + cssm(x)  # Residual connection

        # === READOUT ===
        # Frame selection
        if self.frame_readout == 'last':
            x = x[:, -1]  # (B, H', W', embed_dim)
        else:  # 'all' - keep temporal dimension for pooling
            pass  # x stays (B, T, H', W', embed_dim)

        # Norm -> act
        if self.norm_type == 'batch':
            x = nn.BatchNorm(use_running_average=not training, name='norm_pre')(x)
        else:
            x = nn.LayerNorm(name='norm_pre')(x)
        x = act(x)

        # Pool over space (and time if frame_readout='all')
        if self.frame_readout == 'last':
            # (B, H', W', embed_dim) -> (B, embed_dim)
            if self.pool_type == 'max':
                x = x.max(axis=(1, 2))
            else:
                x = x.mean(axis=(1, 2))
        else:
            # (B, T, H', W', embed_dim) -> (B, embed_dim)
            if self.pool_type == 'max':
                x = x.max(axis=(1, 2, 3))
            else:
                x = x.mean(axis=(1, 2, 3))

        # Final norm
        if self.norm_type == 'batch':
            x = nn.BatchNorm(use_running_average=not training, name='norm_post')(x)
        else:
            x = nn.LayerNorm(name='norm_post')(x)

        # Head: 1x1 -> num_classes (as Dense since we pooled)
        x = nn.Dense(self.num_classes, name='head')(x)

        return x

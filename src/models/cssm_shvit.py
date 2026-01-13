"""
SHViT (Single-Head Vision Transformer) in JAX/Flax.

Converted from PyTorch implementation.

NOTES:
- TODO: BatchNorm has been replaced with GroupNorm throughout. This should be fine for
  training from scratch, but pretrained PyTorch weights may not transfer perfectly.
  If you see unexpected behavior, this is the first place to investigate.
  The original uses BatchNorm which normalizes across the batch dimension and maintains
  running statistics; GroupNorm normalizes within each sample independently.

- SqueezeExcite implemented directly based on timm:
  https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/squeeze_excite.py

- fuse() methods for inference optimization are not implemented.
"""

from typing import Sequence, Literal, Type
import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np

from .cssm import StandardCSSM, GatedOpponentCSSM
from .cssm_vit import CSSMBlock


# -----------------------------------------------------------------------------
# Building Blocks
# -----------------------------------------------------------------------------


class SqueezeExcite(nn.Module):
    """
    Squeeze-and-Excitation block.
    
    Reference: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/squeeze_excite.py
    """
    rd_ratio: float = 0.25
    
    @nn.compact
    def __call__(self, x):
        # x: [B, H, W, C] (Flax uses channels-last)
        C = x.shape[-1]
        rd_channels = int(C * self.rd_ratio)
        
        # Squeeze: global average pooling
        se = x.mean(axis=(1, 2), keepdims=True)  # [B, 1, 1, C]
        
        # Excite: FC -> ReLU -> FC -> Sigmoid
        se = nn.Dense(rd_channels, use_bias=True)(se)
        se = nn.relu(se)
        se = nn.Dense(C, use_bias=True)(se)
        se = nn.sigmoid(se)
        
        return x * se


class GroupNorm(nn.Module):
    """Group Normalization with 1 group (equivalent to LayerNorm over channels)."""
    
    @nn.compact
    def __call__(self, x):
        # x: [B, H, W, C]
        return nn.GroupNorm(num_groups=1)(x)


class Conv2d_GN(nn.Module):
    """Conv2d followed by GroupNorm (replacing BatchNorm)."""
    features: int
    kernel_size: int = 1
    strides: int = 1
    padding: int = 0
    groups: int = 1
    gn_init_scale: float = 1.0
    
    @nn.compact
    def __call__(self, x):
        # x: [B, H, W, C]
        if self.padding > 0:
            x = jnp.pad(x, ((0, 0), (self.padding, self.padding), 
                           (self.padding, self.padding), (0, 0)))
        
        x = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(self.strides, self.strides),
            padding='VALID',
            feature_group_count=self.groups,
            use_bias=False,
        )(x)
        
        # GroupNorm with 1 group, with configurable initial scale
        x = nn.GroupNorm(
            num_groups=1,
            scale_init=nn.initializers.constant(self.gn_init_scale),
            bias_init=nn.initializers.zeros,
        )(x)
        
        return x


class GN_Linear(nn.Module):
    """GroupNorm followed by Linear (replacing BatchNorm + Linear)."""
    features: int
    use_bias: bool = True
    std: float = 0.02
    
    @nn.compact
    def __call__(self, x):
        # x: [B, C]
        # Need to add spatial dims for GroupNorm, then remove
        x = x[:, None, None, :]  # [B, 1, 1, C]
        x = nn.GroupNorm(num_groups=1)(x)
        x = x[:, 0, 0, :]  # [B, C]
        
        x = nn.Dense(
            features=self.features,
            use_bias=self.use_bias,
            kernel_init=nn.initializers.truncated_normal(stddev=self.std),
            bias_init=nn.initializers.zeros,
        )(x)
        
        return x


class FFN(nn.Module):
    """Feed-forward network with two pointwise convolutions."""
    dim: int
    hidden_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = Conv2d_GN(self.hidden_dim)(x)
        x = nn.relu(x)
        x = Conv2d_GN(self.dim, gn_init_scale=0.0)(x)
        return x


class Residual(nn.Module):
    """Residual connection with optional stochastic depth."""
    drop: float = 0.0
    
    @nn.compact
    def __call__(self, x, sublayer_output, *, deterministic: bool = True):
        if not deterministic and self.drop > 0:
            # Stochastic depth: randomly drop the sublayer
            keep_prob = 1.0 - self.drop
            rng = self.make_rng('dropout')
            mask = jax.random.bernoulli(rng, keep_prob, (x.shape[0], 1, 1, 1))
            return x + sublayer_output * mask / keep_prob
        else:
            return x + sublayer_output


class SHSA(nn.Module):
    """Single-Head Self-Attention."""
    dim: int
    qk_dim: int
    pdim: int  # partial dimension for attention
    
    @nn.compact
    def __call__(self, x):
        # x: [B, H, W, C]
        B, H, W, C = x.shape
        
        # Split channels into attention part and bypass part
        x1 = x[..., :self.pdim]           # [B, H, W, pdim]
        x2 = x[..., self.pdim:]           # [B, H, W, dim - pdim]
        
        # Pre-norm on attention part
        x1 = GroupNorm()(x1)
        
        # QKV projection
        qkv = Conv2d_GN(self.qk_dim * 2 + self.pdim)(x1)
        q = qkv[..., :self.qk_dim]                          # [B, H, W, qk_dim]
        k = qkv[..., self.qk_dim:self.qk_dim*2]             # [B, H, W, qk_dim]
        v = qkv[..., self.qk_dim*2:]                        # [B, H, W, pdim]
        
        # Flatten spatial dimensions
        q = q.reshape(B, H * W, self.qk_dim)  # [B, HW, qk_dim]
        k = k.reshape(B, H * W, self.qk_dim)  # [B, HW, qk_dim]
        v = v.reshape(B, H * W, self.pdim)    # [B, HW, pdim]
        
        # Attention: (HW, qk_dim) @ (qk_dim, HW) -> (HW, HW)
        scale = self.qk_dim ** -0.5
        attn = (q @ k.transpose(0, 2, 1)) * scale  # [B, HW, HW]
        attn = nn.softmax(attn, axis=-1)
        
        # Apply attention to values: (HW, HW) @ (HW, pdim) -> (HW, pdim)
        x1 = (attn @ v).reshape(B, H, W, self.pdim)  # [B, H, W, pdim]
        
        # Concatenate and project
        x_cat = jnp.concatenate([x1, x2], axis=-1)  # [B, H, W, dim]
        x_out = nn.relu(x_cat)
        x_out = Conv2d_GN(self.dim, gn_init_scale=0.0)(x_out)
        
        return x_out


class BasicBlock(nn.Module):
    """Basic SHViT block with conv, mixer (optional attention), and FFN."""
    dim: int
    qk_dim: int
    pdim: int
    block_type: Literal["s", "i"]  # "s" = with attention, "i" = identity mixer
    drop: float = 0.0
    
    @nn.compact
    def __call__(self, x, *, deterministic: bool = True):
        residual = Residual(drop=self.drop)
        
        # Depthwise conv
        conv_out = Conv2d_GN(self.dim, kernel_size=3, padding=1, 
                             groups=self.dim, gn_init_scale=0.0)(x)
        x = residual(x, conv_out, deterministic=deterministic)
        
        # Mixer (attention or identity)
        if self.block_type == "s":
            mixer_out = SHSA(self.dim, self.qk_dim, self.pdim)(x)
            x = residual(x, mixer_out, deterministic=deterministic)
        # else: identity, no change
        
        # FFN
        ffn_out = FFN(self.dim, self.dim * 2)(x)
        x = residual(x, ffn_out, deterministic=deterministic)
        
        return x


class PatchMerging(nn.Module):
    """Downsample spatial dimensions and increase channels."""
    dim: int
    out_dim: int
    
    @nn.compact
    def __call__(self, x):
        hid_dim = self.dim * 4
        
        x = Conv2d_GN(hid_dim)(x)
        x = nn.relu(x)
        x = Conv2d_GN(hid_dim, kernel_size=3, strides=2, padding=1, groups=hid_dim)(x)
        x = nn.relu(x)
        x = SqueezeExcite(rd_ratio=0.25)(x)
        x = Conv2d_GN(self.out_dim)(x)
        
        return x


class DownsampleBlock(nn.Module):
    """Pre/post blocks around PatchMerging."""
    in_dim: int
    out_dim: int
    
    @nn.compact
    def __call__(self, x, *, deterministic: bool = True):
        residual = Residual()
        
        # Pre-downsample block
        conv_out = Conv2d_GN(self.in_dim, kernel_size=3, padding=1, groups=self.in_dim)(x)
        x = residual(x, conv_out, deterministic=deterministic)
        ffn_out = FFN(self.in_dim, self.in_dim * 2)(x)
        x = residual(x, ffn_out, deterministic=deterministic)
        
        # Downsample
        x = PatchMerging(self.in_dim, self.out_dim)(x)
        
        # Post-downsample block
        conv_out = Conv2d_GN(self.out_dim, kernel_size=3, padding=1, groups=self.out_dim)(x)
        x = residual(x, conv_out, deterministic=deterministic)
        ffn_out = FFN(self.out_dim, self.out_dim * 2)(x)
        x = residual(x, ffn_out, deterministic=deterministic)
        
        return x


class Stage(nn.Module):
    """A stage containing multiple BasicBlocks."""
    dim: int
    qk_dim: int
    pdim: int
    depth: int
    block_type: Literal["s", "i"]
    drop: float = 0.0
    cssm_cls: Type[nn.Module]
    mlp_ratio: float = 4.0
    dense_mixing: bool = False
    concat_xy: bool = True
    gate_activation: str = 'sigmoid'
    
    @nn.compact
    def __call__(self, x, *, deterministic: bool = True):
        # Transformer blocks with linear drop path increase
        dp_rates = np.linspace(0, self.drop_path_rate, self.depth)

        for i in range(self.depth):
            x = CSSMBlock(
                dim=self.dim,
                cssm_cls=self.cssm_cls,
                mlp_ratio=self.mlp_ratio,
                drop_path=float(dp_rates[i]),
                dense_mixing=self.dense_mixing,
                concat_xy=self.concat_xy,
                gate_activation=self.gate_activation,
                name=f'block{i}'
            )(x, training=deterministic)
            # BasicBlock(
            #     dim=self.dim,
            #     qk_dim=self.qk_dim,
            #     pdim=self.pdim,
            #     block_type=self.block_type,
            #     drop=self.drop,
            # )(x, deterministic=deterministic)
        return x


# -----------------------------------------------------------------------------
# Main Model
# -----------------------------------------------------------------------------


class SHViT(nn.Module):
    """
    Single-Head Vision Transformer.
    
    Args:
        num_classes: Number of output classes. Set to 0 for feature extraction.
        in_chans: Number of input channels.
        embed_dim: Embedding dimensions for each stage.
        partial_dim: Partial dimensions for attention in each stage.
        qk_dim: Query/Key dimensions for each stage.
        depth: Number of blocks in each stage.
        types: Block types for each stage ("s" = with attention, "i" = identity).
        down_ops: Downsample operations between stages.
        distillation: Whether to use a distillation head.
        drop: Stochastic depth drop rate.
    """
    num_classes: int = 1000
    in_chans: int = 3
    embed_dim: Sequence[int] = (128, 256, 384)
    partial_dim: Sequence[int] = (32, 64, 96)
    qk_dim: Sequence[int] = (16, 16, 16)
    depth: Sequence[int] = (1, 2, 3)
    types: Sequence[Literal["s", "i"]] = ("s", "s", "s")
    down_ops: Sequence[Sequence[str]] = (("subsample", "2"), ("subsample", "2"), ("",))
    distillation: bool = False
    drop: float = 0.0
    cssm_type: str = 'opponent'
    mlp_ratio: float = 4.0
    dense_mixing: bool = False
    concat_xy: bool = True
    gate_activation: str = 'sigmoid'
    
    @nn.compact
    def __call__(self, x, *, deterministic: bool = True):
        """
        Args:
            x: Input tensor of shape [B, H, W, C] (channels-last).
            deterministic: If True, disable dropout/stochastic depth.
        
        Returns:
            If distillation=False: logits of shape [B, num_classes]
            If distillation=True and training: tuple of (logits, dist_logits)
            If distillation=True and inference: averaged logits
        """
        # Select CSSM class
        if self.cssm_type == 'standard':
            CSSM = StandardCSSM
        else:
            CSSM = GatedOpponentCSSM
        
        # Patch embedding: 4 conv layers with stride 2 each (total 16x downsample)
        x = Conv2d_GN(self.embed_dim[0] // 8, kernel_size=3, strides=2, padding=1)(x)
        x = nn.relu(x)
        x = Conv2d_GN(self.embed_dim[0] // 4, kernel_size=3, strides=2, padding=1)(x)
        x = nn.relu(x)
        x = Conv2d_GN(self.embed_dim[0] // 2, kernel_size=3, strides=2, padding=1)(x)
        x = nn.relu(x)
        x = Conv2d_GN(self.embed_dim[0], kernel_size=3, strides=2, padding=1)(x)
        
        # Stage 1
        x = Stage(
            dim=self.embed_dim[0],
            qk_dim=self.qk_dim[0],
            pdim=self.partial_dim[0],
            depth=self.depth[0],
            block_type=self.types[0],
            drop=self.drop,
            mlp_ratio=self.mlp_ratio,
            dense_mixing=self.dense_mixing,
            cssm_cls=CSSM,
            concat_xy=self.concat_xy,
            gate_activation=self.gate_activation
        )(x, deterministic=deterministic)
        
        # Downsample 1 -> 2 (if specified)
        if len(self.down_ops) > 0 and len(self.down_ops[0]) > 0 and self.down_ops[0][0] == "subsample":
            x = DownsampleBlock(self.embed_dim[0], self.embed_dim[1])(x, deterministic=deterministic)
        
        # Stage 2
        x = Stage(
            dim=self.embed_dim[1],
            qk_dim=self.qk_dim[1],
            pdim=self.partial_dim[1],
            depth=self.depth[1],
            block_type=self.types[1],
            drop=self.drop,
            mlp_ratio=self.mlp_ratio,
            dense_mixing=self.dense_mixing,
            cssm_cls=CSSM,
            concat_xy=self.concat_xy,
            gate_activation=self.gate_activation
        )(x, deterministic=deterministic)
        
        # Downsample 2 -> 3 (if specified)
        if len(self.down_ops) > 1 and len(self.down_ops[1]) > 0 and self.down_ops[1][0] == "subsample":
            x = DownsampleBlock(self.embed_dim[1], self.embed_dim[2])(x, deterministic=deterministic)
        
        # Stage 3
        x = Stage(
            dim=self.embed_dim[2],
            qk_dim=self.qk_dim[2],
            pdim=self.partial_dim[2],
            depth=self.depth[2],
            block_type=self.types[2],
            drop=self.drop,
            mlp_ratio=self.mlp_ratio,
            dense_mixing=self.dense_mixing,
            cssm_cls=CSSM,
            concat_xy=self.concat_xy,
            gate_activation=self.gate_activation
        )(x, deterministic=deterministic)
        
        # Global average pooling
        x = x.mean(axis=(1, 2))  # [B, C]
        
        # Classification head
        if self.num_classes > 0:
            if self.distillation:
                logits = GN_Linear(self.num_classes)(x)
                dist_logits = GN_Linear(self.num_classes)(x)
                
                if deterministic:
                    # Inference: average the two heads
                    return (logits + dist_logits) / 2
                else:
                    # Training: return both
                    return logits, dist_logits
            else:
                return GN_Linear(self.num_classes)(x)
        else:
            return x
        

# -----------------------------------------------------------------------------
# Model Variants (matching PyTorch defaults)
# -----------------------------------------------------------------------------
        

def shvit_s1(num_classes: int = 1000, **kwargs) -> SHViT:
    """SHViT-S1: Smallest variant."""
    return SHViT(
        num_classes=num_classes,
        embed_dim=(128, 256, 384),
        partial_dim=(32, 64, 96),
        qk_dim=(16, 16, 16),
        depth=(1, 2, 3),
        types=("i", "s", "s"),
        **kwargs
    )


def shvit_s2(num_classes: int = 1000, **kwargs) -> SHViT:
    """SHViT-S2: Small variant."""
    return SHViT(
        num_classes=num_classes,
        embed_dim=(128, 256, 384),
        partial_dim=(32, 64, 96),
        qk_dim=(16, 16, 16),
        depth=(2, 4, 6),
        types=("i", "s", "s"),
        **kwargs
    )


def shvit_s3(num_classes: int = 1000, **kwargs) -> SHViT:
    """SHViT-S3: Medium variant."""
    return SHViT(
        num_classes=num_classes,
        embed_dim=(128, 256, 512),
        partial_dim=(32, 64, 128),
        qk_dim=(16, 16, 16),
        depth=(2, 4, 6),
        types=("i", "s", "s"),
        **kwargs
    )


def shvit_s4(num_classes: int = 1000, **kwargs) -> SHViT:
    """SHViT-S4: Large variant."""
    return SHViT(
        num_classes=num_classes,
        embed_dim=(160, 320, 640),
        partial_dim=(40, 80, 160),
        qk_dim=(16, 16, 16),
        depth=(2, 4, 12),
        types=("i", "s", "s"),
        **kwargs
    )
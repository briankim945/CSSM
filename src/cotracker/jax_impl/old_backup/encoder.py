"""
JAX/Flax implementation of CoTracker's BasicEncoder.

The BasicEncoder is a ResNet-style CNN that extracts features from video frames.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Optional


class ResidualBlock(nn.Module):
    """Residual block with optional downsampling."""
    features: int
    stride: int = 1
    norm: str = 'instance'  # 'instance', 'batch', 'group', or 'none'

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        residual = x

        # First conv
        x = nn.Conv(self.features, (3, 3), strides=(self.stride, self.stride), padding='SAME')(x)
        x = self._norm(x, training)
        x = nn.relu(x)

        # Second conv
        x = nn.Conv(self.features, (3, 3), padding='SAME')(x)
        x = self._norm(x, training)

        # Shortcut
        if self.stride != 1 or residual.shape[-1] != self.features:
            residual = nn.Conv(self.features, (1, 1), strides=(self.stride, self.stride))(residual)
            residual = self._norm(residual, training)

        return nn.relu(x + residual)

    def _norm(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        if self.norm == 'instance':
            # Instance norm: normalize over H, W
            mean = x.mean(axis=(1, 2), keepdims=True)
            var = x.var(axis=(1, 2), keepdims=True)
            return (x - mean) / jnp.sqrt(var + 1e-5)
        elif self.norm == 'batch':
            return nn.BatchNorm(use_running_average=not training)(x)
        elif self.norm == 'group':
            return nn.GroupNorm(num_groups=8)(x)
        else:
            return x


class BasicEncoder(nn.Module):
    """
    Basic CNN encoder for video feature extraction.

    Similar to CoTracker's BasicEncoder, this uses a ResNet-style architecture
    with progressive downsampling to extract features.

    Attributes:
        output_dim: Output feature dimension
        stride: Total downsampling factor (4 or 8)
        norm: Normalization type
    """
    output_dim: int = 128
    stride: int = 4
    norm: str = 'instance'

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = True,
    ) -> jnp.ndarray:
        """
        Extract features from input images.

        Args:
            x: Input images (B, H, W, C) or video (B, T, H, W, C)
            training: Whether in training mode

        Returns:
            Features (B, H/stride, W/stride, output_dim) or
                     (B, T, H/stride, W/stride, output_dim)
        """
        # Handle video input
        is_video = x.ndim == 5
        if is_video:
            B, T, H, W, C = x.shape
            x = x.reshape(B * T, H, W, C)

        # Initial convolution with stride 2
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding='SAME')(x)
        x = self._norm(x, training, 64)
        x = nn.relu(x)

        # Residual layers
        x = self._make_layer(x, 64, 2, stride=1, training=training)
        x = self._make_layer(x, 96, 2, stride=2 if self.stride >= 4 else 1, training=training)

        if self.stride >= 8:
            x = self._make_layer(x, 128, 2, stride=2, training=training)
            x = self._make_layer(x, 128, 2, stride=1, training=training)

        # Output projection
        x = nn.Conv(self.output_dim, (1, 1))(x)

        # Reshape back to video if needed
        if is_video:
            _, H_out, W_out, D = x.shape
            x = x.reshape(B, T, H_out, W_out, D)

        return x

    def _make_layer(
        self,
        x: jnp.ndarray,
        features: int,
        num_blocks: int,
        stride: int,
        training: bool,
    ) -> jnp.ndarray:
        """Create a layer of residual blocks."""
        x = ResidualBlock(features, stride=stride, norm=self.norm)(x, training)
        for _ in range(1, num_blocks):
            x = ResidualBlock(features, stride=1, norm=self.norm)(x, training)
        return x

    def _norm(self, x: jnp.ndarray, training: bool, features: int) -> jnp.ndarray:
        if self.norm == 'instance':
            mean = x.mean(axis=(1, 2), keepdims=True)
            var = x.var(axis=(1, 2), keepdims=True)
            return (x - mean) / jnp.sqrt(var + 1e-5)
        elif self.norm == 'batch':
            return nn.BatchNorm(use_running_average=not training)(x)
        elif self.norm == 'group':
            return nn.GroupNorm(num_groups=8)(x)
        else:
            return x


class CSSMEncoder(nn.Module):
    """
    CSSM-based video encoder using existing CSSM-SHViT architecture.

    This wraps the existing CSSM-SHViT model to provide feature extraction
    for the CoTracker pipeline, leveraging pretrained JEPA weights.

    Attributes:
        output_dim: Output feature dimension
        model_size: CSSM-SHViT model size ('s1', 's2', 's3', 's4')
        pretrained_path: Path to pretrained JEPA checkpoint (optional)
    """
    output_dim: int = 128
    model_size: str = 's1'
    pretrained_path: Optional[str] = None

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = True,
    ) -> jnp.ndarray:
        """
        Extract features using CSSM-SHViT.

        Args:
            x: Input video (B, T, H, W, C)
            training: Whether in training mode

        Returns:
            Features (B, T, H/stride, W/stride, output_dim)
        """
        # Import here to avoid circular dependency
        from ...models.cssm_shvit import CSSMSHViT

        # Get model config based on size
        configs = {
            's1': {'embed_dims': [128, 256, 384], 'depths': [1, 2, 4]},
            's2': {'embed_dims': [128, 256, 384], 'depths': [1, 3, 8]},
            's3': {'embed_dims': [192, 384, 512], 'depths': [1, 3, 10]},
            's4': {'embed_dims': [256, 512, 768], 'depths': [1, 3, 11]},
        }
        config = configs.get(self.model_size, configs['s1'])

        # Create CSSM-SHViT model
        model = CSSMSHViT(
            embed_dims=config['embed_dims'],
            depths=config['depths'],
            num_classes=0,  # No classification head
        )

        # Forward pass
        features = model(x, training=training)

        # Project to output dimension if needed
        if features.shape[-1] != self.output_dim:
            features = nn.Dense(self.output_dim)(features)

        return features

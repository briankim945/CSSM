"""
JAX/Flax implementation of CSSM-CoTracker.

This module provides a complete point tracking model that uses CSSM
for temporal and spatial processing instead of transformer attention.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Optional, Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from .encoder import BasicEncoder, CSSMEncoder
from .correlation import EfficientCorrBlock, bilinear_sample


class CSSMUpdateBlock(nn.Module):
    """
    CSSM-based update block for iterative track refinement.

    Replaces CoTracker's EfficientUpdateFormer with CSSM-based processing.

    Attributes:
        hidden_dim: Hidden feature dimension
        cssm_type: Type of CSSM ('standard', 'opponent', 'hgru')
        num_virtual_tracks: Number of virtual tokens for spatial reasoning
        kernel_size: CSSM spatial kernel size
    """
    hidden_dim: int = 256
    cssm_type: str = 'opponent'
    num_virtual_tracks: int = 64
    kernel_size: int = 11

    @nn.compact
    def __call__(
        self,
        track_feats: jnp.ndarray,
        corr_feats: jnp.ndarray,
        flow_feats: jnp.ndarray,
        vis: jnp.ndarray,
        training: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Update track predictions using CSSM.

        Args:
            track_feats: Track features (B, T, N, D)
            corr_feats: Correlation features (B, T, N, D_corr)
            flow_feats: Flow/delta features (B, T, N, D_flow)
            vis: Visibility predictions (B, T, N, 1)
            training: Training mode flag

        Returns:
            delta: Coordinate updates (B, T, N, 2)
            vis_out: Updated visibility (B, T, N, 1)
            track_feats_out: Updated track features (B, T, N, D)
        """
        from ...models.cssm import GatedOpponentCSSM, StandardCSSM

        B, T, N, D = track_feats.shape

        # Concatenate all input features
        x = jnp.concatenate([track_feats, corr_feats, flow_feats, vis], axis=-1)

        # Project to hidden dimension
        x = nn.Dense(self.hidden_dim, name='input_proj')(x)
        x = nn.gelu(x)

        # Add virtual tokens for spatial reasoning
        virtual = self.param(
            'virtual_tokens',
            nn.initializers.normal(0.02),
            (1, 1, self.num_virtual_tracks, self.hidden_dim)
        )
        virtual = jnp.broadcast_to(virtual, (B, T, self.num_virtual_tracks, self.hidden_dim))
        x_with_virtual = jnp.concatenate([x, virtual], axis=2)  # (B, T, N+V, D)

        # Temporal CSSM processing
        # Reshape: (B, T, N+V, D) -> (B*(N+V), T, 1, 1, D)
        N_total = N + self.num_virtual_tracks
        x_temporal = x_with_virtual.transpose(0, 2, 1, 3)  # (B, N+V, T, D)
        x_temporal = x_temporal.reshape(B * N_total, T, 1, 1, self.hidden_dim)

        # Create CSSM layer
        if self.cssm_type == 'opponent':
            cssm = GatedOpponentCSSM(
                channels=self.hidden_dim,
                kernel_size=self.kernel_size,
            )
        else:
            cssm = StandardCSSM(
                channels=self.hidden_dim,
                kernel_size=self.kernel_size,
            )

        # Apply CSSM
        x_temporal = cssm(x_temporal)

        # Reshape back: (B*(N+V), T, 1, 1, D) -> (B, T, N+V, D)
        x_temporal = x_temporal.reshape(B, N_total, T, self.hidden_dim)
        x_temporal = x_temporal.transpose(0, 2, 1, 3)

        # Residual connection
        x_with_virtual = nn.LayerNorm(name='temporal_norm')(x_with_virtual + x_temporal)

        # Spatial MLP for point interactions
        x_spatial = nn.Dense(self.hidden_dim * 4, name='spatial_up')(x_with_virtual)
        x_spatial = nn.gelu(x_spatial)
        x_spatial = nn.Dense(self.hidden_dim, name='spatial_down')(x_spatial)
        x_with_virtual = nn.LayerNorm(name='spatial_norm')(x_with_virtual + x_spatial)

        # Remove virtual tokens
        x_out = x_with_virtual[:, :, :N, :]

        # Output heads
        delta = nn.Dense(2, name='delta_head')(x_out)
        vis_out = nn.Dense(1, name='vis_head')(x_out)

        # Update track features
        track_feats_out = nn.Dense(D, name='track_proj')(x_out)
        track_feats_out = track_feats + track_feats_out  # Residual

        return delta, vis_out, track_feats_out


class CSSMCoTracker(nn.Module):
    """
    Complete CSSM-based point tracker.

    This model replaces CoTracker's transformer-based architecture with
    CSSM for efficient temporal processing.

    Attributes:
        stride: Feature extraction stride
        hidden_dim: Hidden feature dimension
        latent_dim: Encoder output dimension
        corr_levels: Number of correlation pyramid levels
        corr_radius: Correlation sampling radius
        num_iters: Number of update iterations
        cssm_type: Type of CSSM to use
        use_cssm_encoder: Whether to use CSSM-SHViT encoder
        encoder_model_size: CSSM-SHViT model size (if using CSSM encoder)
    """
    stride: int = 4
    hidden_dim: int = 256
    latent_dim: int = 128
    corr_levels: int = 4
    corr_radius: int = 3
    num_iters: int = 4
    cssm_type: str = 'opponent'
    use_cssm_encoder: bool = False
    encoder_model_size: str = 's1'

    @nn.compact
    def __call__(
        self,
        video: jnp.ndarray,
        queries: jnp.ndarray,
        training: bool = True,
    ) -> Dict[str, jnp.ndarray]:
        """
        Track points through video.

        Args:
            video: Input video (B, T, H, W, 3)
            queries: Query points (B, N, 3) where each row is [frame_idx, x, y]
            training: Training mode flag

        Returns:
            Dictionary with:
            - coords: Predicted coordinates (B, T, N, 2)
            - vis: Visibility predictions (B, T, N, 1)
            - all_coords: Coordinates from all iterations (num_iters, B, T, N, 2)
            - all_vis: Visibility from all iterations (num_iters, B, T, N, 1)
        """
        B, T, H, W, C = video.shape
        N = queries.shape[1]

        # Extract features
        if self.use_cssm_encoder:
            encoder = CSSMEncoder(
                output_dim=self.latent_dim,
                model_size=self.encoder_model_size,
            )
        else:
            encoder = BasicEncoder(
                output_dim=self.latent_dim,
                stride=self.stride,
            )

        fmaps = encoder(video, training=training)  # (B, T, H', W', D)
        _, _, H_f, W_f, D = fmaps.shape

        # Initialize tracks from query positions
        query_frames = queries[:, :, 0].astype(jnp.int32)  # (B, N)
        query_coords = queries[:, :, 1:3]  # (B, N, 2) - x, y coordinates

        # Scale coordinates to feature map resolution
        scale = jnp.array([W_f / W, H_f / H])
        coords = query_coords * scale  # (B, N, 2)

        # Expand to all timesteps
        coords = jnp.broadcast_to(coords[:, None, :, :], (B, T, N, 2))
        coords = jnp.array(coords)  # Make mutable

        # Initialize visibility (1 for all)
        vis = jnp.ones((B, T, N, 1))

        # Sample initial track features from query frame
        track_feats = self._sample_track_features(fmaps, query_frames, query_coords, scale)
        track_feats = jnp.broadcast_to(track_feats[:, None, :, :], (B, T, N, self.latent_dim))
        track_feats = jnp.array(track_feats)

        # Create correlation block
        corr_block = EfficientCorrBlock(
            num_levels=self.corr_levels,
            radius=self.corr_radius,
        )

        # Create update block
        update_block = CSSMUpdateBlock(
            hidden_dim=self.hidden_dim,
            cssm_type=self.cssm_type,
        )

        # Iterative refinement
        all_coords = []
        all_vis = []

        for iter_idx in range(self.num_iters):
            # Compute correlation features
            corr_feats = self._compute_correlations(fmaps, coords, track_feats, corr_block)

            # Compute flow features (coordinate deltas)
            flow_feats = self._compute_flow_features(coords, query_coords, scale)

            # Update predictions
            delta, vis_update, track_feats = update_block(
                track_feats, corr_feats, flow_feats, vis, training=training
            )

            # Apply updates
            coords = coords + delta
            vis = nn.sigmoid(vis + vis_update)

            all_coords.append(coords)
            all_vis.append(vis)

        # Scale coordinates back to original resolution
        coords_out = coords / scale

        return {
            'coords': coords_out,
            'vis': vis,
            'all_coords': jnp.stack([c / scale for c in all_coords], axis=0),
            'all_vis': jnp.stack(all_vis, axis=0),
        }

    def _sample_track_features(
        self,
        fmaps: jnp.ndarray,
        query_frames: jnp.ndarray,
        query_coords: jnp.ndarray,
        scale: jnp.ndarray,
    ) -> jnp.ndarray:
        """Sample features at query locations."""
        B, T, H, W, D = fmaps.shape
        N = query_frames.shape[1]

        # Scale coordinates
        coords_scaled = query_coords * scale  # (B, N, 2)

        # Sample from each batch element's query frame
        features = []
        for b in range(B):
            batch_features = []
            for n in range(N):
                frame_idx = query_frames[b, n]
                fmap = fmaps[b, frame_idx]  # (H, W, D)
                coord = coords_scaled[b, n:n+1]  # (1, 2)
                feat = bilinear_sample(fmap[None], coord[None])  # (1, 1, D)
                batch_features.append(feat[0, 0])
            features.append(jnp.stack(batch_features, axis=0))

        return jnp.stack(features, axis=0)  # (B, N, D)

    def _compute_correlations(
        self,
        fmaps: jnp.ndarray,
        coords: jnp.ndarray,
        track_feats: jnp.ndarray,
        corr_block: nn.Module,
    ) -> jnp.ndarray:
        """Compute correlation features at current coordinates."""
        B, T, H, W, D = fmaps.shape
        N = coords.shape[2]

        # Compute correlations for each timestep pair
        corr_list = []
        for t in range(T):
            # Use track_feats as reference, correlate with fmaps at time t
            fmap_t = fmaps[:, t]  # (B, H, W, D)
            coords_t = coords[:, t]  # (B, N, 2)

            # For each point, compute correlation with neighborhood
            # Simplified: sample fmap_t at coords_t
            sampled = bilinear_sample(fmap_t, coords_t)  # (B, N, D)
            corr_list.append(sampled)

        return jnp.stack(corr_list, axis=1)  # (B, T, N, D)

    def _compute_flow_features(
        self,
        coords: jnp.ndarray,
        query_coords: jnp.ndarray,
        scale: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute flow features from coordinate deltas."""
        # Delta from query position
        query_scaled = query_coords * scale
        delta = coords - query_scaled[:, None, :, :]  # (B, T, N, 2)

        # Embed deltas
        # Use sinusoidal positional encoding for flow
        flow_dim = 32
        freqs = jnp.logspace(0, 4, flow_dim // 4, base=2.0)

        delta_flat = delta.reshape(-1, 2)
        enc_x = jnp.concatenate([
            jnp.sin(delta_flat[:, 0:1] * freqs),
            jnp.cos(delta_flat[:, 0:1] * freqs)
        ], axis=-1)
        enc_y = jnp.concatenate([
            jnp.sin(delta_flat[:, 1:2] * freqs),
            jnp.cos(delta_flat[:, 1:2] * freqs)
        ], axis=-1)

        flow_feats = jnp.concatenate([enc_x, enc_y], axis=-1)
        return flow_feats.reshape(*coords.shape[:-1], -1)


def create_cssm_cotracker(
    config: Optional[Dict[str, Any]] = None,
    use_cssm_encoder: bool = False,
    cssm_type: str = 'opponent',
) -> CSSMCoTracker:
    """
    Factory function to create CSSM-CoTracker model.

    Args:
        config: Optional configuration dictionary
        use_cssm_encoder: Whether to use CSSM-SHViT encoder
        cssm_type: Type of CSSM ('standard', 'opponent', 'hgru')

    Returns:
        CSSMCoTracker model instance
    """
    default_config = {
        'stride': 4,
        'hidden_dim': 256,
        'latent_dim': 128,
        'corr_levels': 4,
        'corr_radius': 3,
        'num_iters': 4,
    }

    if config is not None:
        default_config.update(config)

    return CSSMCoTracker(
        use_cssm_encoder=use_cssm_encoder,
        cssm_type=cssm_type,
        **default_config,
    )

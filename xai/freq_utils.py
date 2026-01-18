"""
Frequency Analysis for CSSM Blocks in JAX

Two approaches:
1. Simple: Visualize frequency magnitude before/after learned filter
2. Informative: Track frequency changes across blocks
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial
from typing import Dict, List, Tuple, NamedTuple, Callable
import numpy as np


# =============================================================================
# Version 1: Simple - Visualize before/after filter within a single block
# =============================================================================

class FrequencySnapshot(NamedTuple):
    """Stores frequency domain representations at different stages."""
    pre_filter: jnp.ndarray   # After FFT, before learned filter
    post_filter: jnp.ndarray  # After learned filter, before IFFT
    learned_filter: jnp.ndarray  # The filter weights themselves


def extract_frequency_simple(
    x: jnp.ndarray,
    learned_filter: jnp.ndarray,
) -> FrequencySnapshot:
    """
    Extract frequency representations before and after the learned filter.
    
    Args:
        x: Input tensor [batch, time, height, width, channels]
        learned_filter: Learned frequency-domain filter weights
        
    Returns:
        FrequencySnapshot with pre-filter, post-filter, and filter weights
    """
    # FFT over height and width (axes 2 and 3)
    # Using rfft2 since input is real-valued
    pre_filter = jnp.fft.rfft2(x, axes=(2, 3))
    
    # Apply learned filter (elementwise multiply in frequency domain)
    post_filter = pre_filter * learned_filter
    
    return FrequencySnapshot(
        pre_filter=pre_filter,
        post_filter=post_filter,
        learned_filter=learned_filter
    )


def compute_magnitude(freq_tensor: jnp.ndarray) -> jnp.ndarray:
    """Compute log magnitude of complex frequency tensor."""
    magnitude = jnp.abs(freq_tensor)
    # Log scale for better visualization (add small epsilon for stability)
    log_magnitude = jnp.log1p(magnitude)
    return log_magnitude


def visualize_simple_single_timestep(
    snapshot: FrequencySnapshot,
    timestep: int = 0,
    batch_idx: int = 0,
    channel_idx: int = 0,
    save_path: str = None
):
    """
    Visualize before/after filter for a single timestep.
    
    Shows three panels:
    1. Frequency magnitude before filter
    2. Learned filter magnitude
    3. Frequency magnitude after filter
    """
    # Extract single timestep and channel
    pre = snapshot.pre_filter[batch_idx, timestep, :, :, channel_idx]
    post = snapshot.post_filter[batch_idx, timestep, :, :, channel_idx]
    filt = snapshot.learned_filter
    
    # Handle filter shape (might not have batch/time dims)
    if filt.ndim > 2:
        # Assuming filter might be [height, width, channels] or similar
        if filt.ndim == 3:
            filt = filt[:, :, channel_idx]
        elif filt.ndim == 5:
            filt = filt[batch_idx, timestep, :, :, channel_idx]
    
    # Compute magnitudes
    pre_mag = compute_magnitude(pre)
    post_mag = compute_magnitude(post)
    filt_mag = jnp.log1p(jnp.abs(filt))
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    im0 = axes[0].imshow(np.array(pre_mag), cmap='viridis', aspect='auto')
    axes[0].set_title('Pre-filter\n(log magnitude)')
    axes[0].set_xlabel('Frequency (W)')
    axes[0].set_ylabel('Frequency (H)')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(np.array(filt_mag), cmap='viridis', aspect='auto')
    axes[1].set_title('Learned Filter\n(log magnitude)')
    axes[1].set_xlabel('Frequency (W)')
    axes[1].set_ylabel('Frequency (H)')
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(np.array(post_mag), cmap='viridis', aspect='auto')
    axes[2].set_title('Post-filter\n(log magnitude)')
    axes[2].set_xlabel('Frequency (W)')
    axes[2].set_ylabel('Frequency (H)')
    plt.colorbar(im2, ax=axes[2])
    
    # Difference: what changed?
    diff = post_mag - pre_mag
    im3 = axes[3].imshow(np.array(diff), cmap='RdBu_r', aspect='auto',
                         vmin=-jnp.abs(diff).max(), vmax=jnp.abs(diff).max())
    axes[3].set_title('Change\n(post - pre)')
    axes[3].set_xlabel('Frequency (W)')
    axes[3].set_ylabel('Frequency (H)')
    plt.colorbar(im3, ax=axes[3])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_simple_gif(
    snapshot: FrequencySnapshot,
    batch_idx: int = 0,
    channel_idx: int = 0,
    save_path: str = 'frequency_simple.gif',
    fps: int = 5
):
    """
    Create GIF stepping through timesteps for simple before/after visualization.
    """
    n_timesteps = snapshot.pre_filter.shape[1]
    
    # Setup figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Initialize with first timestep
    pre_mag = compute_magnitude(snapshot.pre_filter[batch_idx, 0, :, :, channel_idx])
    post_mag = compute_magnitude(snapshot.post_filter[batch_idx, 0, :, :, channel_idx])
    
    # Handle filter (usually same across timesteps)
    filt = snapshot.learned_filter
    if filt.ndim == 3:
        filt_mag = jnp.log1p(jnp.abs(filt[:, :, channel_idx]))
    elif filt.ndim == 5:
        filt_mag = jnp.log1p(jnp.abs(filt[batch_idx, 0, :, :, channel_idx]))
    else:
        filt_mag = jnp.log1p(jnp.abs(filt))
    
    diff = post_mag - pre_mag
    
    # Create initial plots
    im0 = axes[0].imshow(np.array(pre_mag), cmap='viridis', aspect='auto')
    axes[0].set_title('Pre-filter')
    
    im1 = axes[1].imshow(np.array(filt_mag), cmap='viridis', aspect='auto')
    axes[1].set_title('Learned Filter')
    
    im2 = axes[2].imshow(np.array(post_mag), cmap='viridis', aspect='auto')
    axes[2].set_title('Post-filter')
    
    # For difference, we need consistent scale across frames
    # Pre-compute max diff across all timesteps
    max_diff = 0
    for t in range(n_timesteps):
        pre_t = compute_magnitude(snapshot.pre_filter[batch_idx, t, :, :, channel_idx])
        post_t = compute_magnitude(snapshot.post_filter[batch_idx, t, :, :, channel_idx])
        max_diff = max(max_diff, float(jnp.abs(post_t - pre_t).max()))
    
    im3 = axes[3].imshow(np.array(diff), cmap='RdBu_r', aspect='auto',
                         vmin=-max_diff, vmax=max_diff)
    axes[3].set_title('Change')
    
    title = fig.suptitle(f'Timestep 0 / {n_timesteps - 1}')
    plt.tight_layout()
    
    def update(frame):
        pre_mag = compute_magnitude(snapshot.pre_filter[batch_idx, frame, :, :, channel_idx])
        post_mag = compute_magnitude(snapshot.post_filter[batch_idx, frame, :, :, channel_idx])
        diff = post_mag - pre_mag
        
        im0.set_array(np.array(pre_mag))
        im0.set_clim(vmin=float(pre_mag.min()), vmax=float(pre_mag.max()))
        
        im2.set_array(np.array(post_mag))
        im2.set_clim(vmin=float(post_mag.min()), vmax=float(post_mag.max()))
        
        im3.set_array(np.array(diff))
        
        title.set_text(f'Timestep {frame} / {n_timesteps - 1}')
        
        return [im0, im2, im3, title]
    
    anim = animation.FuncAnimation(
        fig, update, frames=n_timesteps, interval=1000//fps, blit=True
    )
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close()
    
    return save_path


# =============================================================================
# Version 2: Informative - Track frequency changes across blocks
# =============================================================================

class BlockFrequencyInfo(NamedTuple):
    """Frequency information for a single block."""
    block_idx: int
    input_freq: jnp.ndarray   # Frequency repr of block input
    output_freq: jnp.ndarray  # Frequency repr of block output
    learned_filter: jnp.ndarray


def extract_block_info_from_state(
    state,
    x: jnp.ndarray,
    block_output_key: str = 'block_outputs',
    filter_param_path: callable = lambda params, i: params['blocks'][i]['filter'],
    n_blocks: int = None,
) -> List[BlockFrequencyInfo]:
    """
    Extract frequency info using TrainState.
    
    Args:
        state: Your TrainState object
        x: Input tensor [batch, time, height, width, channels]
        block_output_key: Key in intermediates dict for block outputs
        filter_param_path: Function that gets filter params given (params, block_idx)
        n_blocks: Number of blocks (inferred from intermediates if None)
    
    Returns:
        List of BlockFrequencyInfo
    """
    # Call model with intermediates
    output, intermediates = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        x,
        return_intermediates=True,
        train=False
    )
    
    block_outputs = intermediates[block_output_key]
    
    if n_blocks is None:
        n_blocks = len(block_outputs)
    
    learned_filters = [
        filter_param_path(state.params, i) for i in range(n_blocks)
    ]
    
    return extract_frequency_across_blocks(x, block_outputs, learned_filters)


def extract_frequency_across_blocks(
    x: jnp.ndarray,
    block_outputs: List[jnp.ndarray],
    learned_filters: List[jnp.ndarray]
) -> List[BlockFrequencyInfo]:
    """
    Given a list of block outputs (in spatial domain), compute frequency info.
    
    This is an alternative to hooking - if you can easily get block outputs,
    just pass them here.
    
    Args:
        x: Original input [batch, time, height, width, channels]
        block_outputs: List of outputs from each block, same shape as x
        learned_filters: List of learned filter weights for each block
        
    Returns:
        List of BlockFrequencyInfo for each block
    """
    block_info = []
    
    # First block takes original input
    prev_output = x
    
    for i, (block_out, filt) in enumerate(zip(block_outputs, learned_filters)):
        input_freq = jnp.fft.rfft2(prev_output, axes=(2, 3))
        output_freq = jnp.fft.rfft2(block_out, axes=(2, 3))
        
        block_info.append(BlockFrequencyInfo(
            block_idx=i,
            input_freq=input_freq,
            output_freq=output_freq,
            learned_filter=filt
        ))
        
        prev_output = block_out
    
    return block_info


def compute_radial_average(freq_2d: jnp.ndarray) -> jnp.ndarray:
    """
    Compute radial average of 2D frequency magnitude.
    
    Collapses 2D frequency representation to 1D "power vs frequency" curve.
    This is useful for summarizing what frequencies dominate.
    
    Args:
        freq_2d: 2D frequency magnitude [height, width]
        
    Returns:
        1D array of average magnitude at each frequency radius
    """
    h, w = freq_2d.shape
    
    # Create coordinate grids
    cy, cx = h // 2, w // 2  # Center (DC component)
    y, x = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing='ij')
    
    # Compute radius from center for each point
    # Note: for rfft2 output, the layout is different - DC at [0,0]
    # Adjusting for rfft2 layout:
    radius = jnp.sqrt(y**2 + x**2)
    radius = radius.astype(jnp.int32)
    
    # Compute average magnitude at each radius
    max_radius = int(jnp.sqrt(h**2 + w**2)) + 1
    radial_avg = jnp.zeros(max_radius)
    counts = jnp.zeros(max_radius)
    
    # This is a simple loop version - could be vectorized
    for r in range(max_radius):
        mask = (radius == r)
        if mask.sum() > 0:
            radial_avg = radial_avg.at[r].set(freq_2d[mask].mean())
            counts = counts.at[r].set(mask.sum())
    
    # Only return up to meaningful radius
    meaningful_radius = min(h, w) // 2
    return radial_avg[:meaningful_radius]


def visualize_across_blocks_single_timestep(
    block_info_list: List[BlockFrequencyInfo],
    timestep: int = 0,
    batch_idx: int = 0,
    channel_idx: int = 0,
    save_path: str = None
):
    """
    Visualize how frequency content changes across blocks for one timestep.
    
    Shows:
    - Top row: Output frequency magnitude for each block
    - Bottom row: Radial average (power spectrum) for each block
    """
    n_blocks = len(block_info_list)
    
    fig, axes = plt.subplots(2, n_blocks, figsize=(4 * n_blocks, 8))
    
    if n_blocks == 1:
        axes = axes.reshape(2, 1)
    
    for i, info in enumerate(block_info_list):
        # Get output frequency for this block
        out_freq = info.output_freq[batch_idx, timestep, :, :, channel_idx]
        out_mag = compute_magnitude(out_freq)
        
        # Top row: 2D frequency magnitude
        im = axes[0, i].imshow(np.array(out_mag), cmap='viridis', aspect='auto')
        axes[0, i].set_title(f'Block {info.block_idx}\nOutput Freq Magnitude')
        axes[0, i].set_xlabel('Freq (W)')
        axes[0, i].set_ylabel('Freq (H)')
        plt.colorbar(im, ax=axes[0, i])
        
        # Bottom row: Radial average
        radial = compute_radial_average(np.array(out_mag))
        axes[1, i].plot(radial)
        axes[1, i].set_title(f'Block {info.block_idx}\nRadial Average')
        axes[1, i].set_xlabel('Frequency (radius)')
        axes[1, i].set_ylabel('Log Magnitude')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.suptitle(f'Frequency Analysis Across Blocks (t={timestep})')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_across_blocks_gif(
    block_info_list: List[BlockFrequencyInfo],
    batch_idx: int = 0,
    channel_idx: int = 0,
    save_path: str = 'frequency_across_blocks.gif',
    fps: int = 5
):
    """
    Create GIF showing frequency evolution across blocks over time.
    """
    n_blocks = len(block_info_list)
    n_timesteps = block_info_list[0].output_freq.shape[1]
    
    fig, axes = plt.subplots(2, n_blocks, figsize=(4 * n_blocks, 8))
    
    if n_blocks == 1:
        axes = axes.reshape(2, 1)
    
    # Initialize plots
    images = []
    lines = []
    
    for i, info in enumerate(block_info_list):
        out_freq = info.output_freq[batch_idx, 0, :, :, channel_idx]
        out_mag = compute_magnitude(out_freq)
        
        im = axes[0, i].imshow(np.array(out_mag), cmap='viridis', aspect='auto')
        axes[0, i].set_title(f'Block {info.block_idx}')
        images.append(im)
        
        radial = compute_radial_average(np.array(out_mag))
        line, = axes[1, i].plot(radial)
        axes[1, i].set_xlim(0, len(radial))
        # Set y-axis limits based on data range across all timesteps
        all_radials = []
        for t in range(n_timesteps):
            mag_t = compute_magnitude(info.output_freq[batch_idx, t, :, :, channel_idx])
            all_radials.append(compute_radial_average(np.array(mag_t)))
        all_radials = np.array(all_radials)
        axes[1, i].set_ylim(all_radials.min() * 0.9, all_radials.max() * 1.1)
        axes[1, i].grid(True, alpha=0.3)
        lines.append(line)
    
    title = fig.suptitle(f'Timestep 0 / {n_timesteps - 1}')
    plt.tight_layout()
    
    def update(frame):
        for i, info in enumerate(block_info_list):
            out_freq = info.output_freq[batch_idx, frame, :, :, channel_idx]
            out_mag = compute_magnitude(out_freq)
            
            images[i].set_array(np.array(out_mag))
            images[i].set_clim(vmin=float(out_mag.min()), vmax=float(out_mag.max()))
            
            radial = compute_radial_average(np.array(out_mag))
            lines[i].set_ydata(radial)
        
        title.set_text(f'Timestep {frame} / {n_timesteps - 1}')
        return images + lines + [title]
    
    anim = animation.FuncAnimation(
        fig, update, frames=n_timesteps, interval=1000//fps, blit=True
    )
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close()
    
    return save_path


# =============================================================================
# Comparison utilities - for comparing CSSM variants
# =============================================================================

def compare_variants_frequency(
    variant_block_infos: Dict[str, List[BlockFrequencyInfo]],
    timestep: int = 0,
    batch_idx: int = 0,
    channel_idx: int = 0,
    save_path: str = None
):
    """
    Compare frequency characteristics across different CSSM variants.
    
    Args:
        variant_block_infos: Dict mapping variant name -> list of BlockFrequencyInfo
            e.g., {'Standard': [...], 'Gated': [...], 'Neuroscience': [...]}
        timestep: Which timestep to visualize
        batch_idx: Which batch element
        channel_idx: Which channel
        save_path: Optional path to save figure
    """
    variant_names = list(variant_block_infos.keys())
    n_variants = len(variant_names)
    
    # Assume all variants have same number of blocks
    n_blocks = len(variant_block_infos[variant_names[0]])
    
    fig, axes = plt.subplots(n_variants, n_blocks, figsize=(4 * n_blocks, 4 * n_variants))
    
    if n_variants == 1:
        axes = axes.reshape(1, -1)
    if n_blocks == 1:
        axes = axes.reshape(-1, 1)
    
    for v_idx, variant_name in enumerate(variant_names):
        block_infos = variant_block_infos[variant_name]
        
        for b_idx, info in enumerate(block_infos):
            out_freq = info.output_freq[batch_idx, timestep, :, :, channel_idx]
            out_mag = compute_magnitude(out_freq)
            
            im = axes[v_idx, b_idx].imshow(np.array(out_mag), cmap='viridis', aspect='auto')
            
            if v_idx == 0:
                axes[v_idx, b_idx].set_title(f'Block {info.block_idx}')
            if b_idx == 0:
                axes[v_idx, b_idx].set_ylabel(f'{variant_name}')
            
            plt.colorbar(im, ax=axes[v_idx, b_idx])
    
    plt.suptitle(f'Frequency Comparison Across Variants (t={timestep})')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def compare_radial_profiles(
    variant_block_infos: Dict[str, List[BlockFrequencyInfo]],
    block_idx: int = 0,
    timestep: int = 0,
    batch_idx: int = 0,
    channel_idx: int = 0,
    save_path: str = None
):
    """
    Compare radial frequency profiles across variants for a specific block.
    
    Plots all variants on the same axes for direct comparison.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for variant_name, block_infos in variant_block_infos.items():
        info = block_infos[block_idx]
        out_freq = info.output_freq[batch_idx, timestep, :, :, channel_idx]
        out_mag = compute_magnitude(out_freq)
        radial = compute_radial_average(np.array(out_mag))
        
        ax.plot(radial, label=variant_name, linewidth=2)
    
    ax.set_xlabel('Frequency (radius from DC)', fontsize=12)
    ax.set_ylabel('Log Magnitude', fontsize=12)
    ax.set_title(f'Radial Frequency Profile - Block {block_idx} (t={timestep})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    # Create dummy data for testing
    batch, time, height, width, channels = 2, 10, 32, 32, 64
    
    # Dummy input
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch, time, height, width, channels))
    
    # Dummy learned filter (in frequency domain, so rfft2 output shape)
    freq_h, freq_w = height, width // 2 + 1  # rfft2 output shape
    learned_filter = jax.random.normal(
        jax.random.PRNGKey(1), 
        (freq_h, freq_w, channels)
    ) * 0.1 + 1.0  # Center around 1 so it doesn't kill signal
    
    # Version 1: Simple visualization
    print("Running simple frequency extraction...")
    snapshot = extract_frequency_simple(x, learned_filter)
    
    print("Creating single timestep visualization...")
    fig = visualize_simple_single_timestep(
        snapshot, 
        timestep=0, 
        batch_idx=0, 
        channel_idx=0,
        save_path='frequency_simple_t0.png'
    )
    plt.close()
    
    print("Creating GIF over timesteps...")
    create_simple_gif(
        snapshot,
        batch_idx=0,
        channel_idx=0,
        save_path='frequency_simple.gif',
        fps=3
    )
    
    # Version 2: Across blocks (simulated with multiple dummy blocks)
    print("\nSimulating multiple blocks...")
    n_blocks = 4
    block_outputs = []
    learned_filters = []
    
    h = x
    for i in range(n_blocks):
        # Simulate a block that slightly modifies the signal
        key_i = jax.random.PRNGKey(i + 10)
        noise = jax.random.normal(key_i, h.shape) * 0.1
        h = h + noise  # Dummy "block" operation
        block_outputs.append(h)
        
        # Each block has its own filter
        filt_i = jax.random.normal(
            jax.random.PRNGKey(i + 20),
            (freq_h, freq_w, channels)
        ) * 0.1 + 1.0
        learned_filters.append(filt_i)
    
    block_info_list = extract_frequency_across_blocks(x, block_outputs, learned_filters)
    
    print("Creating across-blocks visualization...")
    fig = visualize_across_blocks_single_timestep(
        block_info_list,
        timestep=0,
        batch_idx=0,
        channel_idx=0,
        save_path='frequency_across_blocks_t0.png'
    )
    plt.close()
    
    print("Creating across-blocks GIF...")
    create_across_blocks_gif(
        block_info_list,
        batch_idx=0,
        channel_idx=0,
        save_path='frequency_across_blocks.gif',
        fps=3
    )
    
    print("\nDone! Check the output files:")
    print("  - frequency_simple_t0.png")
    print("  - frequency_simple.gif")
    print("  - frequency_across_blocks_t0.png")
    print("  - frequency_across_blocks.gif")
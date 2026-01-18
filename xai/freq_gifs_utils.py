"""
Generate per-block frequency analysis GIFs from a trained JAX model.

Each block gets its own GIF showing:
- Frequency magnitude (output of block in frequency domain)
- Radial average (1D power spectrum)
- Change view (how this block modified the frequency content)

Usage:
    python generate_frequency_gifs.py

Assumes you have:
    - A TrainState object loaded as `state`
    - An input batch `x` with shape (batch, time, height, width, channels)
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Import from the frequency_analysis module
from freq_utils import compute_magnitude, compute_radial_average


def get_block_data(
    state,
    x: jnp.ndarray,
    block_indices: Optional[List[int]] = None,
) -> Tuple[Dict[int, jnp.ndarray], Dict[int, jnp.ndarray], jnp.ndarray]:
    """
    Extract block outputs and learned filters from model.
    
    Args:
        state: TrainState object
        x: Input tensor (batch, time, height, width, channels)
        block_indices: Which blocks to extract (default: all 12)
    
    Returns:
        block_outputs: Dict mapping block_idx -> output tensor
        learned_filters: Dict mapping block_idx -> filter weights
        x: Original input (for computing change from input)
    """
    if block_indices is None:
        block_indices = list(range(12))
    
    # Call model with intermediates
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    output, intermediates = state.apply_fn(
        variables,
        x,
        return_intermediates=True,
        train=False
    )
    
    # Extract block outputs
    # Keys are like 'cssmblock_0_out', 'cssmblock_1_out', etc.
    block_outputs = {}
    for i in block_indices:
        key = f'cssmblock_{i}_out'
        if key in intermediates:
            block_outputs[i] = intermediates[key]
        else:
            print(f"Warning: {key} not found in intermediates")
    
    # Extract learned filters
    # Params are like state.params['block0']['cssm']['kernel']
    learned_filters = {}
    for i in block_indices:
        param_key = f'block{i}'
        if param_key in state.params:
            learned_filters[i] = state.params[param_key]['cssm']['kernel']
        else:
            print(f"Warning: {param_key} not found in params")
    
    return block_outputs, learned_filters, x


def create_single_block_gif(
    block_idx: int,
    block_output: jnp.ndarray,
    prev_output: jnp.ndarray,
    learned_filter: jnp.ndarray,
    output_dir: str = './frequency_gifs',
    batch_idx: int = 0,
    channel_idx: int = 0,
    fps: int = 3,
) -> str:
    """
    Create a GIF for a single block showing frequency analysis over timesteps.
    
    Layout (2 rows, 3 columns):
    Top row:    [Output Freq Mag] [Filter (Spatial)] [Filter (Freq)]
    Bottom row: [Change]          [Radial Average]   [Input Freq Mag]
    
    Args:
        block_idx: Index of this block
        block_output: Output of this block (batch, time, H, W, channels)
        prev_output: Input to this block (output of previous block, or original input for block 0)
        learned_filter: Filter weights (channels, H, W) - in spatial domain
        output_dir: Directory to save GIF
        batch_idx: Which batch element to visualize
        channel_idx: Which channel to visualize
        fps: Frames per second
    
    Returns:
        Path to saved GIF
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    n_timesteps = block_output.shape[1]
    
    # Setup figure with 2x3 layout
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Block {block_idx} - Frequency Analysis', fontsize=14)
    
    # Compute frequency representations for first timestep
    # Block output in frequency domain
    out_freq_t0 = jnp.fft.rfft2(block_output[batch_idx, 0, :, :, channel_idx], axes=(0, 1))
    out_mag_t0 = compute_magnitude(out_freq_t0)
    
    # Previous block output (input to this block) in frequency domain
    prev_freq_t0 = jnp.fft.rfft2(prev_output[batch_idx, 0, :, :, channel_idx], axes=(0, 1))
    prev_mag_t0 = compute_magnitude(prev_freq_t0)
    
    # Change
    change_t0 = out_mag_t0 - prev_mag_t0
    
    # Learned filter (same across timesteps)
    # Filter shape is (channels, H, W), extract one channel
    filter_spatial = learned_filter[channel_idx, :, :]
    filter_spatial_mag = filter_spatial  # Keep raw values for spatial view
    
    # Convert filter to frequency domain for comparison
    # Use fft2 and fftshift to center DC component
    filter_freq = jnp.fft.fft2(filter_spatial)
    filter_freq_shifted = jnp.fft.fftshift(filter_freq)
    filter_freq_mag = compute_magnitude(filter_freq_shifted)
    
    # Radial average of output
    radial_t0 = compute_radial_average(np.array(out_mag_t0))
    
    # Pre-compute radial limits across all timesteps
    all_radials = []
    for t in range(n_timesteps):
        out_freq_t = jnp.fft.rfft2(block_output[batch_idx, t, :, :, channel_idx], axes=(0, 1))
        out_mag_t = compute_magnitude(out_freq_t)
        all_radials.append(compute_radial_average(np.array(out_mag_t)))
    all_radials = np.array(all_radials)
    radial_min, radial_max = all_radials.min() * 0.9, all_radials.max() * 1.1
    
    # Pre-compute change limits across all timesteps
    max_change = 0
    for t in range(n_timesteps):
        out_freq_t = jnp.fft.rfft2(block_output[batch_idx, t, :, :, channel_idx], axes=(0, 1))
        prev_freq_t = jnp.fft.rfft2(prev_output[batch_idx, t, :, :, channel_idx], axes=(0, 1))
        change_t = compute_magnitude(out_freq_t) - compute_magnitude(prev_freq_t)
        max_change = max(max_change, float(jnp.abs(change_t).max()))
    if max_change == 0:
        max_change = 1.0  # Avoid division issues
    
    # ===== TOP ROW =====
    
    # Panel (0,0): Output frequency magnitude
    im_out = axes[0, 0].imshow(np.array(out_mag_t0), cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Output Freq\n(log magnitude)')
    axes[0, 0].set_xlabel('Freq (W)')
    axes[0, 0].set_ylabel('Freq (H)')
    plt.colorbar(im_out, ax=axes[0, 0])
    
    # Panel (0,1): Learned filter in SPATIAL domain
    im_filt_spatial = axes[0, 1].imshow(np.array(filter_spatial_mag), cmap='RdBu_r', aspect='auto',
                                         vmin=-np.abs(filter_spatial_mag).max(),
                                         vmax=np.abs(filter_spatial_mag).max())
    axes[0, 1].set_title('Learned Filter\n(spatial domain)')
    axes[0, 1].set_xlabel('W')
    axes[0, 1].set_ylabel('H')
    plt.colorbar(im_filt_spatial, ax=axes[0, 1])
    
    # Panel (0,2): Learned filter in FREQUENCY domain
    im_filt_freq = axes[0, 2].imshow(np.array(filter_freq_mag), cmap='viridis', aspect='auto')
    axes[0, 2].set_title('Learned Filter\n(freq domain, centered)')
    axes[0, 2].set_xlabel('Freq (W)')
    axes[0, 2].set_ylabel('Freq (H)')
    plt.colorbar(im_filt_freq, ax=axes[0, 2])
    
    # ===== BOTTOM ROW =====
    
    # Panel (1,0): Change (output - input in frequency domain)
    im_change = axes[1, 0].imshow(np.array(change_t0), cmap='RdBu_r', aspect='auto',
                                   vmin=-max_change, vmax=max_change)
    axes[1, 0].set_title('Freq Change\n(out - in)')
    axes[1, 0].set_xlabel('Freq (W)')
    axes[1, 0].set_ylabel('Freq (H)')
    plt.colorbar(im_change, ax=axes[1, 0])
    
    # Panel (1,1): Radial average
    line, = axes[1, 1].plot(radial_t0, 'b-', linewidth=2, label='Output')
    axes[1, 1].set_title('Radial Average\n(power spectrum)')
    axes[1, 1].set_xlabel('Frequency (radius)')
    axes[1, 1].set_ylabel('Log Magnitude')
    axes[1, 1].set_xlim(0, len(radial_t0))
    axes[1, 1].set_ylim(radial_min, radial_max)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc='upper right')
    
    # Panel (1,2): Input frequency magnitude (for reference)
    im_in = axes[1, 2].imshow(np.array(prev_mag_t0), cmap='viridis', aspect='auto')
    axes[1, 2].set_title('Input Freq\n(log magnitude)')
    axes[1, 2].set_xlabel('Freq (W)')
    axes[1, 2].set_ylabel('Freq (H)')
    plt.colorbar(im_in, ax=axes[1, 2])
    
    # Timestep indicator
    time_text = fig.text(0.02, 0.98, f't=0/{n_timesteps-1}', fontsize=12,
                         verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    def update(frame):
        # Compute frequency representations for this timestep
        out_freq = jnp.fft.rfft2(block_output[batch_idx, frame, :, :, channel_idx], axes=(0, 1))
        out_mag = compute_magnitude(out_freq)
        
        prev_freq = jnp.fft.rfft2(prev_output[batch_idx, frame, :, :, channel_idx], axes=(0, 1))
        prev_mag = compute_magnitude(prev_freq)
        
        change = out_mag - prev_mag
        radial = compute_radial_average(np.array(out_mag))
        
        # Update dynamic plots (filter stays static)
        im_out.set_array(np.array(out_mag))
        im_out.set_clim(vmin=float(out_mag.min()), vmax=float(out_mag.max()))
        
        im_change.set_array(np.array(change))
        
        im_in.set_array(np.array(prev_mag))
        im_in.set_clim(vmin=float(prev_mag.min()), vmax=float(prev_mag.max()))
        
        line.set_ydata(radial)
        
        time_text.set_text(f't={frame}/{n_timesteps-1}')
        
        return [im_out, im_change, im_in, line, time_text]
    
    anim = animation.FuncAnimation(
        fig, update, frames=n_timesteps, interval=1000//fps, blit=True
    )
    
    save_path = f'{output_dir}/block_{block_idx:02d}_frequency.gif'
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close()
    
    return save_path


def generate_all_block_gifs(
    state,
    x: jnp.ndarray,
    output_dir: str = './frequency_gifs',
    block_indices: Optional[List[int]] = None,
    batch_idx: int = 0,
    channel_idx: int = 0,
    fps: int = 3,
) -> List[str]:
    """
    Generate frequency analysis GIFs for all (or specified) blocks.
    
    Args:
        state: TrainState object
        x: Input tensor (batch, time, height, width, channels)
        output_dir: Directory to save GIFs
        block_indices: Which blocks to process (default: all 12)
        batch_idx: Which batch element to visualize
        channel_idx: Which channel to visualize
        fps: Frames per second
    
    Returns:
        List of paths to saved GIFs
    """
    if block_indices is None:
        block_indices = list(range(12))
    
    print(f"Extracting block data for blocks: {block_indices}")
    block_outputs, learned_filters, original_input = get_block_data(
        state, x, block_indices
    )
    
    # We need to compute "previous output" for each block
    # Block 0's input is the original input (after patch embedding)
    # Block N's input is Block N-1's output
    
    # For the "change" computation, we need the input to each block
    # Since we only have block outputs, we'll use:
    # - For block 0: original_input (or we could skip change view)
    # - For block N: block N-1's output
    
    saved_paths = []
    
    for i in block_indices:
        print(f"Processing block {i}...")
        
        block_output = block_outputs[i]
        
        # Determine previous output (input to this block)
        if i == 0:
            # Block 0's input is original input
            # Note: this is before patch embedding, so might not match dimensionally
            # You might need to adjust this based on your architecture
            prev_output = original_input
            
            # If dimensions don't match, we can compute change relative to 
            # a zero baseline or skip the change view for block 0
            if prev_output.shape != block_output.shape:
                print(f"  Note: Block 0 input shape {prev_output.shape} != output shape {block_output.shape}")
                print(f"  Using zeros for change comparison")
                prev_output = jnp.zeros_like(block_output)
        else:
            # Use previous block's output
            if (i - 1) in block_outputs:
                prev_output = block_outputs[i - 1]
            else:
                print(f"  Warning: Block {i-1} output not available, using zeros")
                prev_output = jnp.zeros_like(block_output)
        
        learned_filter = learned_filters[i]
        
        save_path = create_single_block_gif(
            block_idx=i,
            block_output=block_output,
            prev_output=prev_output,
            learned_filter=learned_filter,
            output_dir=output_dir,
            batch_idx=batch_idx,
            channel_idx=channel_idx,
            fps=fps,
        )
        
        saved_paths.append(save_path)
        print(f"  Saved: {save_path}")
    
    return saved_paths


# =============================================================================
# Example usage / test with dummy data
# =============================================================================

if __name__ == "__main__":
    # This is a test with dummy data
    # Replace this section with your actual state loading
    
    print("Running with dummy data for testing...")
    print("In practice, replace this with your actual state and input.\n")
    
    # Create a mock state-like object for testing
    class MockState:
        def __init__(self):
            self.params = {}
            self.batch_stats = {}
            
            # Create dummy params matching your structure
            for i in range(12):
                self.params[f'block{i}'] = {
                    'cssm': {
                        'kernel': np.random.randn(384, 15, 15).astype(np.float32) * 0.1
                    }
                }
        
        def apply_fn(self, variables, x, return_intermediates=False, train=False):
            # Mock forward pass that returns dummy intermediates
            batch, time, h, w, c = x.shape
            
            intermediates = {}
            current = x
            
            for i in range(12):
                # Simulate some transformation
                noise = jax.random.normal(jax.random.PRNGKey(i), current.shape) * 0.1
                current = current + noise
                intermediates[f'cssmblock_{i}_out'] = current
            
            output = jnp.mean(current, axis=(1, 2, 3))  # Dummy classification output
            
            if return_intermediates:
                return output, intermediates
            return output
    
    # Create mock state and input
    state = MockState()
    
    # Input shape: (batch, time, height, width, channels)
    x = jax.random.normal(jax.random.PRNGKey(42), (1, 8, 14, 14, 384))
    
    # Generate GIFs for a few blocks (not all 12 for quick testing)
    saved_paths = generate_all_block_gifs(
        state=state,
        x=x,
        output_dir='./test_frequency_gifs',
        block_indices=[0, 1, 2],  # Just first 3 blocks for testing
        batch_idx=0,
        channel_idx=0,
        fps=3,
    )
    
    print(f"\nGenerated {len(saved_paths)} GIFs:")
    for path in saved_paths:
        print(f"  {path}")


# =============================================================================
# Template for your actual usage
# =============================================================================

"""
# In your notebook or script:

from generate_frequency_gifs import generate_all_block_gifs

# Load your state (however you normally do it)
state = load_checkpoint(...)  # Your loading code

# Get a sample input batch
x = next(iter(your_dataloader))  # Or however you get input

# Generate all GIFs
saved_paths = generate_all_block_gifs(
    state=state,
    x=x,
    output_dir='./my_frequency_analysis',
    block_indices=None,  # None = all 12 blocks
    batch_idx=0,
    channel_idx=0,  # Or pick a different channel
    fps=3,
)

# Or generate for specific blocks only:
saved_paths = generate_all_block_gifs(
    state=state,
    x=x,
    output_dir='./my_frequency_analysis',
    block_indices=[0, 5, 11],  # Just these blocks
    batch_idx=0,
    channel_idx=0,
    fps=3,
)
"""
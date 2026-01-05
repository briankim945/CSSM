import jax
import jax.numpy as jnp
from flax import linen as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
from PIL import Image
from random import randint, choice
import os

from typing import Tuple

from src.data import IMAGENET_MEAN, IMAGENET_STD


def compute_gradcam(gradients: jnp.ndarray, activations: jnp.ndarray) -> jnp.ndarray:
    """Compute GradCAM heatmap"""
    gradients = gradients.squeeze(0)
    activations = activations.squeeze(0)
    
    pooled_grads = jnp.mean(gradients, axis=(0, 1))
    weighted = activations * pooled_grads[None, None, :]
    heatmap = jnp.sum(weighted, axis=-1)
    
    heatmap = jnp.maximum(heatmap, 0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    return heatmap


def get_heatmap(state, params, perturbations, input_im, input_label, layer_name='cssmblock_11_out'):
    # Inference with activations
    print("\nInference with activations:")
    logits, acts = state.apply_fn(
        {'params': params},
        input_im,
        training=False,
        return_activations=True
    )
    print(f"  Output shape: {logits.shape}")
    print(f"  Activations: {list(acts.keys())}")
    
    # GradCAM
    print("\nGradCAM:")
    heatmap = gradcam_apply(
        state=state,
        params=params,
        perturbations=perturbations,
        image=input_im,
        target_class=input_label[0],
        layer_name=layer_name
    )
    print(f"  Heatmap shape: {heatmap.shape}")
    print(f"  Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    
    return heatmap


def gradcam_apply(
    state: nn.Module,
    params: dict,
    perturbations: dict,
    image: jnp.ndarray,
    target_class: int,
    layer_name: str = 'conv3_out'
) -> jnp.ndarray:
    """Complete GradCAM with cleaner API"""
    
    # Step 1: Get gradients
    def loss_fn(params, perturbations, x, target):
        variables = {'params': params, 'perturbations': perturbations}
        logits = state.apply_fn(variables, x, training=False,)  # return_activations=False by default
        return logits[0, target]
    
    grads = jax.grad(loss_fn, argnums=1)(params, perturbations, image, target_class)
    layer_grads = grads[layer_name]
    print(f"Shape of layer_grads for layer {layer_name}: {layer_grads.shape}")
    
    # Step 2: Get activations - SAME MODEL INSTANCE!
    _, activations = state.apply_fn(
        {'params': params},
        image,
        training=False,
        return_activations=True  # ← Just pass the flag!
    )
    layer_activations = activations[layer_name]
    
    # Step 3: Compute GradCAM
    heatmap = compute_gradcam(layer_grads, layer_activations)
    
    return heatmap


def visualize_gradcam_simple(
    image: jnp.ndarray,
    heatmap: jnp.ndarray,
    alpha: float = 0.4,
    colormap: str = 'jet',
    image_name: str = None
):
    """
    Simple side-by-side visualization of image and GradCAM
    
    Args:
        image: Original image (1, H, W, C) or (H, W, C) or (H, W)
        heatmap: GradCAM heatmap (H, W)
        alpha: Transparency for overlay (0-1)
        colormap: Matplotlib colormap for heatmap
    """
    # Remove batch dimension if present
    if image.ndim == 4:
        image = image[0]  # (H, W, C)
    
    # Handle grayscale vs RGB
    if image.ndim == 3 and image.shape[-1] == 1:
        image_display = image[:, :, 0]  # Remove channel dim for grayscale
        cmap_image = 'gray'
    elif image.ndim == 3:
        image_display = image  # RGB
        cmap_image = None
    else:
        image_display = image  # Already (H, W)
        cmap_image = 'gray'
    
    # Resize heatmap to match image size if needed
    if heatmap.shape != image_display.shape[:2]:
        zoom_factors = (
            image_display.shape[0] / heatmap.shape[0],
            image_display.shape[1] / heatmap.shape[1]
        )
        heatmap_resized = zoom(heatmap, zoom_factors, order=1)
    else:
        heatmap_resized = heatmap
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Original image
    if cmap_image:
        axes[0].imshow(image_display, cmap=cmap_image)
    else:
        axes[0].imshow(image_display)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # Plot 2: Heatmap alone
    im = axes[1].imshow(heatmap_resized, cmap=colormap)
    axes[1].set_title('GradCAM Heatmap', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot 3: Overlay
    if cmap_image:
        axes[2].imshow(image_display, cmap=cmap_image)
    else:
        axes[2].imshow(image_display)
    axes[2].imshow(heatmap_resized, cmap=colormap, alpha=alpha)
    axes[2].set_title(f'GradCAM Overlay (α={alpha})', fontsize=14)
    axes[2].axis('off')

    if image_name is not None:
        plt.suptitle(image_name)
    
    plt.tight_layout()
    plt.show()


def visualize_gradcam_video(
    video: jnp.ndarray,
    heatmaps: jnp.ndarray,
    alpha: float = 0.4,
    colormap: str = 'jet',
    image_name: str = None,
    pan: bool = False,
    layer_name: str = None,
):
    """
    Simple side-by-side visualization of image and GradCAM
    
    Args:
        video: Original image (1, D, H, W, C) or (D, H, W, C) or (D, H, W)
        heatmap: GradCAM heatmap (H, W)
        alpha: Transparency for overlay (0-1)
        colormap: Matplotlib colormap for heatmap
    """
    # Remove batch dimension if present
    if video.ndim == 5:
        video = video[0]  # (H, W, C)
    D = video.shape[0]
    assert video.shape[0] == heatmaps.shape[0]
    
    # Handle grayscale vs RGB
    if video.ndim == 4 and video.shape[-1] == 1:
        video_display = video[:, :, 0]  # Remove channel dim for grayscale
        cmap_image = 'gray'
    elif video.ndim == 4:
        video_display = video  # RGB
        cmap_image = None
    else:
        video_display = video  # Already (H, W)
        cmap_image = 'gray'
    
    # Resize heatmap to match image size if needed
    if heatmaps.shape != video_display.shape[:2]:
        zoom_factors = (
            1,
            video_display.shape[1] / heatmaps.shape[1],
            video_display.shape[2] / heatmaps.shape[2]
        )
        heatmaps_resized = zoom(heatmaps, zoom_factors, order=1)
    else:
        heatmaps_resized = heatmaps
    
    # Create figure with 3 subplots
    # fig, axes = plt.subplots(D, 3, figsize=(15, 5))
    fig, axes = plt.subplots(D, 3, figsize=(15,5*D))
    
    for i in range(D):
        image_display = video_display[i]
        heatmap_resized = heatmaps_resized[i]

        # Plot 1: Original image
        if cmap_image:
            axes[i, 0].imshow(image_display, cmap=cmap_image)
        else:
            axes[i, 0].imshow(image_display)
        axes[i, 0].set_title('Original Image', fontsize=14)
        axes[i, 0].axis('off')
        
        # Plot 2: Heatmap alone
        im = axes[i, 1].imshow(heatmap_resized, cmap=colormap)
        axes[i, 1].set_title('GradCAM Heatmap', fontsize=14)
        axes[i, 1].axis('off')
        plt.colorbar(im, ax=axes[i,1], fraction=0.046, pad=0.04)
        
        # Plot 3: Overlay
        if cmap_image:
            axes[i, 2].imshow(image_display, cmap=cmap_image)
        else:
            axes[i, 2].imshow(image_display)
        axes[i, 2].imshow(heatmap_resized, cmap=colormap, alpha=alpha)
        axes[i, 2].set_title(f'GradCAM Overlay (α={alpha})', fontsize=14)
        axes[i, 2].axis('off')

    if image_name is not None:
        plt.suptitle(image_name)
    
    plt.tight_layout()
    # plt.show()
    folder = image_name.split('.')[0].split('/')[0]
    filename = image_name.split('.')[0].split('/')[1]
    
    if layer_name is not None:
        filename = f"{filename}_{layer_name}"

    if pan:
        os.makedirs(f'images/{folder}_pan', exist_ok=True)
        plt.savefig(f'images/{folder}_pan/{filename}.png')
    else:
        os.makedirs(f'images/{folder}', exist_ok=True)
        plt.savefig(f'images/{image_name.split('.')[0]}.png')


def create_image_pan_seq(
    img_path: str,
    size: int = 224,
    seq_len: int = 8,
    channels: int = 3,
    jump: int = 10,
):
    img = Image.open(img_path).convert('RGB')

    lr = randint(0, 1)
    ud = randint(0, 1)

    w, h = img.size

    master_size = seq_len * jump + size

    scale = master_size / min(w, h) * 1.15  # Slight over-scale for crop
    new_w, new_h = int(w * scale), int(h * scale)

    # Center crop
    left = (new_w - size) // 2
    top = (new_h - size) // 2

    pan_sequence = np.zeros((seq_len, size, size, channels), dtype=np.float32)

    left, h_jump = choice(
        [
            (left - (seq_len // 2) * jump, jump),
            (left + (seq_len // 2) * jump, -jump)
        ]
    )
    top, v_jump = choice(
        [
            (top - (seq_len // 2) * jump, jump),
            (top + (seq_len // 2) * jump, -jump)
        ]
    )

    img_crops = []

    # fig, axes = plt.subplots(2, 4, figsize=(5*4, 5*2))

    for i in range(seq_len):
        cur_left = left + h_jump * i
        cur_top = top + v_jump * i
        # print(cur_left, cur_top, cur_left + size, cur_top + size)
        img_crop = img.crop((cur_left, cur_top, cur_left + size, cur_top + size))
        img_crops.append(img_crop)
        pan_sequence[i] = np.array(img_crop, dtype=np.float32)
        # axes[i // 4, i % 4].imshow(img_crop)

    # plt.show()

    # Convert to array and normalize
    pan_sequence = pan_sequence / 255.0
    pan_sequence = (pan_sequence - IMAGENET_MEAN) / IMAGENET_STD

    return pan_sequence, np.stack(img_crops, axis=0)

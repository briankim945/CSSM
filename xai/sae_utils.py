import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import sys, os

from src.data import VideoDataLoader, load_image_val, restore_image

from typing import Dict, List, Tuple, NamedTuple, Literal


class EncoderOutput(NamedTuple):
    top_acts: jnp.ndarray      # Activations of the top-k latents
    top_indices: jnp.ndarray   # Indices of the top-k features
    pre_acts: jnp.ndarray      # Activations before top-k selection


class ForwardOutput(NamedTuple):
    sae_out: jnp.ndarray
    latent_acts: jnp.ndarray
    latent_indices: jnp.ndarray
    fvu: jnp.ndarray
    auxk_loss: jnp.ndarray
    multi_topk_fvu: jnp.ndarray
    unordered_latent_acts: jnp.ndarray


class SparseCoder(nn.Module):
    """
    Sparse Autoencoder / Transcoder in JAX/Flax.
    
    NOTE: "groupmax" activation is not yet implemented. Only "topk" is supported.
    TODO: Implement groupmax activation for parity with PyTorch version.
    """
    d_in: int
    num_latents: int | None = None
    expansion_factor: int = 32
    k: int = 32
    activation: Literal["topk"] = "topk"  # TODO: Add "groupmax" support
    normalize_decoder: bool = True
    transcode: bool = False
    multi_topk: bool = False
    skip_connection: bool = False

    def setup(self):
        self._num_latents = self.num_latents or self.d_in * self.expansion_factor

        # Encoder: Linear layer (weight shape: [d_in, num_latents] in Flax convention)
        self.encoder = nn.Dense(
            self._num_latents,
            use_bias=True,
            kernel_init=nn.initializers.lecun_normal(),
            bias_init=nn.initializers.zeros,
        )

        # Decoder weight: [num_latents, d_in]
        if self.transcode:
            self.W_dec = self.param(
                "W_dec",
                nn.initializers.zeros,
                (self._num_latents, self.d_in),
            )
        else:
            # Initialize to match encoder weights (transposed)
            # In Flax, Dense kernel is [d_in, num_latents], so we transpose
            self.W_dec = self.param(
                "W_dec",
                nn.initializers.lecun_normal(),
                (self._num_latents, self.d_in),
            )

        # Decoder bias
        self.b_dec = self.param("b_dec", nn.initializers.zeros, (self.d_in,))

        # Optional skip connection: [d_in, d_in]
        if self.skip_connection:
            self.W_skip = self.param(
                "W_skip",
                nn.initializers.zeros,
                (self.d_in, self.d_in),
            )

    def encode(self, x: jnp.ndarray) -> EncoderOutput:
        """Encode input and select top-k latents."""
        if not self.transcode:
            x = x - self.b_dec

        # Linear + ReLU
        pre_acts = nn.relu(self.encoder(x))

        # Top-k selection
        if self.activation == "topk":
            top_acts, top_indices = jax.lax.top_k(pre_acts, self.k)
        else:
            # TODO: Implement groupmax
            # For groupmax, the logic would be:
            #   1. Reshape pre_acts to [..., k, num_latents // k]
            #   2. Take max along last dimension to get top_acts
            #   3. Compute indices by adding group offsets to argmax indices
            raise NotImplementedError(
                f"Activation '{self.activation}' not implemented. Only 'topk' is currently supported."
            )

        return EncoderOutput(top_acts, top_indices, pre_acts)

    def decode(self, top_acts: jnp.ndarray, top_indices: jnp.ndarray) -> jnp.ndarray:
        """Decode from sparse latent representation."""
        # Gather decoder weights for active latents
        # W_dec shape: [num_latents, d_in]
        # top_indices shape: [..., k]
        # top_acts shape: [..., k]

        # Gather: [..., k, d_in]
        gathered = self.W_dec[top_indices]

        # Weighted sum over k dimension: [..., d_in]
        y = jnp.einsum("...k,...kd->...d", top_acts, gathered)

        return y + self.b_dec

    def __call__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray | None = None,
        dead_mask: jnp.ndarray | None = None,
    ) -> ForwardOutput:
        """
        Forward pass with optional training losses.
        
        Args:
            x: Input tensor of shape [batch, d_in]
            y: Optional target tensor for transcoding. If None, uses x (autoencoding).
            dead_mask: Optional boolean mask of shape [num_latents] indicating dead features.
                       Only needed during training for auxk loss computation.
        
        Returns:
            ForwardOutput containing reconstruction and training metrics.
            For inference, you can ignore fvu, auxk_loss, and multi_topk_fvu.
        """
        top_acts, top_indices, pre_acts = self.encode(x)

        # Default target is input (autoencoding)
        if y is None:
            y = x

        # Decode
        sae_out = self.decode(top_acts, top_indices)

        # Optional skip connection
        if self.skip_connection:
            sae_out = sae_out + x @ self.W_skip.T

        # Residual
        e = y - sae_out

        # Variance for normalization
        total_variance = jnp.sum((y - y.mean(axis=0)) ** 2)

        # AuxK loss for dead features
        if dead_mask is not None:
            num_dead = jnp.sum(dead_mask).astype(jnp.int32)
            k_aux = y.shape[-1] // 2

            # Compute auxk loss only if there are dead features
            def compute_auxk_loss():
                k_aux_actual = jnp.minimum(k_aux, num_dead)
                scale = jnp.minimum(num_dead / k_aux, 1.0)

                # Mask out living latents (set to -inf so they're never selected)
                auxk_latents = jnp.where(dead_mask[None], pre_acts, -jnp.inf)
                
                # Select top-k_aux dead latents
                # Note: k_aux_actual must be static for top_k, so we use k_aux as upper bound
                auxk_acts, auxk_indices = jax.lax.top_k(auxk_latents, k_aux)

                e_hat = self.decode(auxk_acts, auxk_indices)
                loss = jnp.sum((e_hat - jax.lax.stop_gradient(e)) ** 2)
                return scale * loss / total_variance

            auxk_loss = jax.lax.cond(
                num_dead > 0,
                compute_auxk_loss,
                lambda: jnp.array(0.0),
            )
        else:
            auxk_loss = jnp.array(0.0)

        # FVU (fraction of variance unexplained)
        l2_loss = jnp.sum(e ** 2)
        fvu = l2_loss / total_variance

        # Multi-TopK FVU
        if self.multi_topk:
            top_acts_multi, top_indices_multi = jax.lax.top_k(pre_acts, 4 * self.k)
            sae_out_multi = self.decode(top_acts_multi, top_indices_multi)
            multi_topk_fvu = jnp.sum((sae_out_multi - y) ** 2) / total_variance
        else:
            multi_topk_fvu = jnp.array(0.0)

        return ForwardOutput(
            sae_out=sae_out,
            latent_acts=top_acts,
            latent_indices=top_indices,
            fvu=fvu,
            auxk_loss=auxk_loss,
            multi_topk_fvu=multi_topk_fvu,
            unordered_latent_acts=pre_acts,
        )


def train_sae_on_activations(
    activations: jnp.ndarray,  # [N, d_in]
    y: jnp.ndarray = None,
    d_sae: int = 8192,
    sparsity_coef: float = 0.01,
    num_epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    transcode: bool = False,
    skip_connection: bool = False,
):
    """Train SAE on CNN activations."""
    
    d_in = activations.shape[-1]
    
    # Initialize model
    rng = jax.random.PRNGKey(0)
    # model = SAE(d_in=d_in, d_sae=d_sae)
    if transcode:
        model = SparseCoder(d_in, d_sae, transcode=transcode, skip_connection=skip_connection)
    else:
        model = SparseCoder(d_in, d_sae)
    params = model.init(rng, jnp.ones((1, d_in)))
    
    # Create optimizer
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    
    # Training loop
    @jax.jit
    def train_step(state, batch, dead_mask=None, auxk_alpha=sparsity_coef, y=None):
        def loss_fn(params):
            output = state.apply_fn(params, batch, y=y, dead_mask=dead_mask)
            # features, recon = state.apply_fn(params, batch)
            recon_loss = jnp.mean((batch - output.sae_out) ** 2)
            sparsity_loss = jnp.mean(jnp.abs(output.latent_acts))
            total_loss = output.fvu + auxk_alpha * output.auxk_loss
            # total_loss = recon_loss + sparsity_coef * sparsity_loss
            return total_loss, {
                'recon': recon_loss,
                'sparsity': sparsity_loss,
                'active': jnp.mean(output.latent_acts > 0)
            }
        
        (loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)
        
        state = state.apply_gradients(grads=grads)
        return state, loss, metrics
    
    # Handle reshaping for SAE
    if len(activations.shape) > 2:
        activations = jnp.reshape(activations, (-1, activations.shape[-1]))

    if y is not None and len(y.shape) > 2:
        y = jnp.reshape(y, (-1, y.shape[-1]))

    assert activations.shape == y.shape, f"{activations.shape} does not match {y.shape}"
    
    # Train
    num_samples = activations.shape[0]
    for epoch in range(num_epochs):
        # Shuffle data
        perm = jax.random.permutation(rng, num_samples)
        activations_shuffled = activations[perm]
        if y is not None:
            y = y[perm]
        
        # Batch training
        for i in tqdm(range(0, num_samples, batch_size)):
            batch = activations_shuffled[i:i+batch_size]
            if y is not None:
                y_batch = y[i:i+batch_size]

            state, loss, metrics = train_step(state, batch, y=y_batch)
        
        print(f"Epoch {epoch}: Loss={loss:.4f}, "
              f"Active={metrics['active']:.2%}")
        print()
    
    return state, model


def extract_activations(state, params, images, target_layer):
    """
    Get the internal activations from your CNN.
    
    Args:
        cnn_model: Your trained CNN (ResNet, VGG, etc.)
        images: Input images
        target_layer: Which layer to extract from (e.g., 'conv3', 'block4')
    
    Returns:
        Activations from that layer
    """
    # This is using the model you want to analyze
    output, intermediates = state.apply_fn(
        {'params': params},
        images,
        training=False,
        capture_intermediates=True
    )
    
    # Pull out the specific layer
    layer_activations = intermediates['intermediates'][target_layer]['__call__'][0]
    
    return layer_activations  # Shape: [batch, H, W, C] or [batch, C]


def extract_and_create_im_seqs(
    paths, seq_len = 8,
):
    return np.repeat(
        np.stack([load_image_val(path) for path in paths])[:, np.newaxis, :, :, :],
        seq_len,
        axis=1
    )


def extract_activations_with_capture(
    state: nn.Module,
    params: Dict,
    dataloader: type[VideoDataLoader],
    target_layers: List[str],
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Extract activations from specific layers.
    
    Args:
        model: Flax model
        params: Model parameters
        dataset: Input data [N, H, W, C]
        target_layers: List of layer names to extract (e.g., ['conv3', 'dense1'])
    
    Returns:
        Dictionary mapping layer names to activations
    """
    
    # Storage for activations
    layer_activations = {layer: [] for layer in target_layers}
    label_storage = []
    names_storage = []
        
    # Process in batches
    with tqdm(dataloader) as pbar:
        for batch in pbar:
            videos, labels, names = batch

            # Forward pass with intermediate capture
            output, intermediates = state.apply_fn(
                {'params': params},
                videos,
                training=False,
                capture_intermediates=True  # KEY: This captures all intermediates
            )
            
            # Extract desired layers
            for layer_name in target_layers:
                # Intermediates are stored by layer name
                if layer_name in intermediates['intermediates']:
                    layer_acts = intermediates['intermediates'][layer_name]['__call__'][0]
                    layer_activations[layer_name].append(np.array(layer_acts))
                else:
                    print(f"Warning: Layer {layer_name} not found in intermediates")
                    available = list(intermediates['intermediates'].keys())
                    print(f"Available layers: {available[:10]}...")
            
            # Labels and image names
            label_storage.append(np.array(labels))
            names_storage.extend(names)
    
    # Concatenate all batches
    for layer_name in target_layers:
        layer_activations[layer_name] = np.concatenate(
            layer_activations[layer_name], axis=0
        )
        print(f"{layer_name}: shape {layer_activations[layer_name].shape}")

    label_storage = np.concatenate(label_storage, axis=0)
    
    return layer_activations, label_storage, np.array(names_storage)


def save_activations(
    activations: Dict[str, np.ndarray],
    save_path: str
):
    """Save activations to disk."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(save_path, **activations)
    
    file_size_mb = save_path.stat().st_size / (1024 ** 2)
    print(f"\n✓ Saved activations to {save_path}")
    print(f"  File size: {file_size_mb:.1f} MB")


def load_activations(load_path: str) -> Dict[str, np.ndarray]:
    """Load previously saved activations."""
    data = np.load(load_path)
    return {key: data[key] for key in data.keys()}


def find_alive_features(
    sae_state, 
    target_state,
    target_layer, 
    dataset, 
    num_features,
    batch_size: int = 64,
    threshold=0.01
):
    """
    Find features that actually activate (not dead).
    
    A feature is "alive" if it activates above threshold
    on at least some percentage of samples.
    """
    feature_max_activations = np.zeros(num_features)
    feature_mean_activations = np.zeros(num_features)
    feature_activation_frequency = np.zeros(num_features)

    num_samples = dataset.shape[0]
    
    total_samples = 0
    
    for i in tqdm(range(0, num_samples, batch_size)):
        locs = dataset[i:i+batch_size]
        images = extract_and_create_im_seqs(locs)

        # Get activations
        acts = extract_activations(target_state, target_state.params, images, target_layer)
        acts_flat = jnp.reshape(acts, (-1, acts.shape[-1]))
        # sae_features, _ = sae_state.apply_fn(sae_state.params, acts_flat)  # [batch, num_features]
        output = sae_state.apply_fn(sae_state.params, acts_flat)
        # sae_features = output.latent_acts
        sae_features = output.unordered_latent_acts

        # Track statistics
        feature_max_activations = np.maximum(
            feature_max_activations,
            sae_features.max(axis=0)
        )
        feature_mean_activations += sae_features.sum(axis=0)
        feature_activation_frequency += (sae_features > threshold).sum(axis=0)
        
        total_samples += len(images)
    
    feature_mean_activations /= total_samples
    feature_activation_frequency /= total_samples
    
    # Classify features
    alive_features = []
    dead_features = []
    
    for feat_idx in range(num_features):
        if feature_max_activations[feat_idx] < threshold:
            dead_features.append(feat_idx)
        else:
            alive_features.append(feat_idx)
    
    print(f"Alive features: {len(alive_features)}/{num_features}")
    print(f"Dead features: {len(dead_features)}/{num_features}")
    print(f"Dead percentage: {100*len(dead_features)/num_features:.1f}%")
    
    # Return sorted by mean activation (most active first)
    alive_features_sorted = sorted(
        alive_features,
        key=lambda idx: feature_mean_activations[idx],
        reverse=True
    )
    
    return {
        'alive': alive_features_sorted,
        'dead': dead_features,
        'max_activations': feature_max_activations,
        'mean_activations': feature_mean_activations,
        'activation_frequency': feature_activation_frequency
    }


def random_feature_sample(alive_features, n=20, seed=42):
    """Randomly sample features to explore."""
    np.random.seed(seed)
    sampled = np.random.choice(alive_features, size=n, replace=False)
    
    print(f"Randomly selected {n} features to analyze:")
    print(sampled)
    
    return sampled


def find_class_specific_features(
    sae_state, 
    target_state,
    target_layer, 
    dataset, labels_set, top_n=10,
    batch_size: int = 64,
):
    """
    Find features that strongly discriminate between classes.
    """
    # Collect feature activations per class
    features_by_class = defaultdict(list)

    num_samples = dataset.shape[0]
    
    for i in tqdm(range(0, num_samples, batch_size)):
        locs = dataset[i:i+batch_size]
        labels = labels_set[i:i+batch_size]
        images = extract_and_create_im_seqs(locs)
        acts = extract_activations(target_state, target_state.params, images, target_layer)
        acts_flat = jnp.reshape(acts, (-1, acts.shape[-1]))
        # sae_features, _ = sae_state.apply_fn(sae_state.params, acts_flat)
        output = sae_state.apply_fn(sae_state.params, acts_flat)
        sae_features = output.unordered_latent_acts
        
        for feat_vec, label in zip(sae_features, labels):
            features_by_class[label].append(feat_vec)
    
    # Compute mean activation per class
    class_means = {}
    for label, feature_list in features_by_class.items():
        class_means[label] = np.mean(feature_list, axis=0)
    
    # Find most discriminative features for each class
    class_specific = {}
    
    for target_class in class_means.keys():
        target_mean = class_means[target_class]
        
        # Compare to average of all other classes
        other_classes_mean = np.mean([
            class_means[cls] for cls in class_means.keys() 
            if cls != target_class
        ], axis=0)
        
        # Difference (how much more this class activates each feature)
        difference = target_mean - other_classes_mean
        
        # Top features for this class
        top_features = np.argsort(difference)[-top_n:][::-1]
        
        class_specific[target_class] = [
            (feat_idx, difference[feat_idx]) 
            for feat_idx in top_features
        ]
    
    return class_specific


def features_by_max_activation(feature_stats, top_n=50):
    """Find features with highest maximum activations."""
    alive_features = feature_stats['alive']
    max_acts = feature_stats['max_activations']
    
    # Sort by max activation
    sorted_by_max = sorted(
        alive_features,
        key=lambda idx: max_acts[idx],
        reverse=True
    )
    
    print(f"Top {top_n} features by maximum activation:")
    for i, feat_idx in enumerate(sorted_by_max[:top_n]):
        print(f"  {i+1}. Feature {feat_idx}: max={max_acts[feat_idx]:.3f}")
    
    return sorted_by_max[:top_n]


def visualize_top_k_grid(images, activations, labels, feature_idx, rows=4, cols=5):
    """Create a grid visualization of top-k images."""
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            ax.imshow(images[idx])
            ax.set_title(
                f"{labels[idx]}\n{activations[idx]:.3f}",
                fontsize=10,
                fontweight='bold'
            )
            ax.axis('off')
        else:
            ax.axis('off')
    
    fig.suptitle(
        f"Feature {feature_idx} - Top {len(images)} Activating Images",
        fontsize=16,
        fontweight='bold'
    )
    plt.tight_layout()
    
    return fig


def retrieve_image_from_metadata(dataset, metadata, batch_size=64):
    """
    Retrieve the actual image using metadata.
    
    Args:
        dataset: Your data loader
        metadata: Dict with 'batch_idx' and 'image_idx'
    
    Returns:
        image: The actual image tensor/array
    """
    
    batch_idx_target = metadata['batch_idx']
    image_idx_target = metadata['image_idx']
    
    # Navigate to correct batch
    num_samples = dataset.shape[0]
    for current_batch_idx in range(0, num_samples, batch_size):
        if current_batch_idx == batch_idx_target:
            # Found the right batch, extract the image
            locs = dataset[current_batch_idx:current_batch_idx+batch_size]
            # return images[image_idx]
            return load_image_val(locs[image_idx_target])
    
    raise ValueError(f"Batch {batch_idx_target} not found in dataset")


def find_top_k_with_metadata(all_activations, all_metadata, k=20):
    """
    Find top-k activating examples with full spatial information.
    
    Options:
    1. Top-k activation vectors (might be multiple from same image)
    2. Top-k images (aggregate spatial locations first)
    """
    
    # OPTION 1: Top-k activation vectors (most informative for debugging)
    # ===================================================================
    top_k_vector_indices = np.argsort(all_activations)[-k:][::-1]
    
    top_k_vectors = []
    for idx in top_k_vector_indices:
        metadata = all_metadata[idx]
        top_k_vectors.append({
            'activation': float(all_activations[idx]),
            'batch_idx': metadata['batch_idx'],
            'image_idx': metadata['image_idx'],
            'time_step': metadata['time_step'],
            'label': metadata['image_label'],
            'spatial_location': (metadata['spatial_h'], metadata['spatial_w']),
            'global_idx': metadata['global_idx']
        })
    
    print("\nTop-k activation vectors:")
    for i, vec in enumerate(top_k_vectors):
        print(f"{i+1}. Act={vec['activation']:.3f}, "
              f"Image={vec['batch_idx']},{vec['image_idx']}, "
              f"Label={vec['label']}, "
              f"Spatial=({vec['spatial_location'][0]},{vec['spatial_location'][1]})")
    
    # Example output:
    # 1. Act=0.956, Image=3,7, Label=cat, Spatial=(1,2)  ← Cat #7, left ear region
    # 2. Act=0.943, Image=3,7, Label=cat, Spatial=(1,4)  ← Cat #7, right ear region
    # 3. Act=0.912, Image=5,2, Label=cat, Spatial=(1,3)  ← Cat #2, ear region
    # ...
    
    # OPTION 2: Top-k images (aggregate spatial locations)
    # ===================================================================
    # Group by image
    
    image_to_activations = defaultdict(list)
    
    for idx, metadata in enumerate(all_metadata):
        image_key = (metadata['batch_idx'], metadata['image_idx'])
        image_to_activations[image_key].append({
            'activation': float(all_activations[idx]),
            'spatial_h': metadata['spatial_h'],
            'spatial_w': metadata['spatial_w'],
            'time_step': metadata['time_step'],
            'label': metadata['image_label']
        })
    
    # Aggregate per image (e.g., take max)
    image_max_activations = []
    
    for image_key, acts_list in image_to_activations.items():
        activations = [a['activation'] for a in acts_list]
        max_activation = max(activations)
        max_idx = activations.index(max_activation)
        
        image_max_activations.append({
            'batch_idx': image_key[0],
            'image_idx': image_key[1],
            'max_activation': max_activation,
            'max_location': (acts_list[max_idx]['spatial_h'], 
                           acts_list[max_idx]['spatial_w'],
                           acts_list[max_idx]['time_step']),
            'mean_activation': np.mean(activations),
            'num_active_locations': sum(1 for a in activations if a > 0.1),
            'total_locations': sum(1 for _ in activations),
            'label': acts_list[0]['label'],
            'all_spatial_activations': acts_list
        })
    
    # Sort by max activation
    image_max_activations.sort(key=lambda x: x['max_activation'], reverse=True)
    
    top_k_images = image_max_activations[:k]
    
    print("\nTop-k images (by max spatial activation):")
    for i, img in enumerate(top_k_images):
        print(f"{i+1}. Max={img['max_activation']:.3f}, "
              f"Mean={img['mean_activation']:.3f}, "
              f"Label={img['label']}, "
              f"MaxAt=({img['max_location'][0]},{img['max_location'][1]}), "
              f"Active={img['num_active_locations']}/49")
    
    # Example output:
    # 1. Max=0.956, Mean=0.234, Label=cat, MaxAt=(1,2), Active=8/49
    # 2. Max=0.912, Mean=0.198, Label=cat, MaxAt=(1,3), Active=6/49
    # ...
    
    return top_k_vectors, top_k_images


def visualize_top_k_with_spatial_info(
    dataset,
    top_k_images,  # From find_top_k_with_metadata
    feature_idx,
    rows=4,
    cols=5,
    t_count=8,
    h_count=14,
    w_count=14,
    first_only=True, # Not handling a sequence of images
    batch_size=64,
    pair_show=True,
):
    """
    Visualize top-k images with spatial activation overlays.
    """
    
    if pair_show:
        fig, axes = plt.subplots(rows, cols * 2, figsize=(4 * cols * 2, 4 * rows))
    else:
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    axes = axes.flatten()

    # Carrier for video sequences
    sequences = []
    images = []

    # for idx_super, ax in enumerate(axes.flat):
    for idx_super, ax in enumerate(axes):
        if first_only:
            if pair_show:
                idx = idx_super // 2
            else:
                idx = idx_super

            if idx >= len(top_k_images):
                ax.axis('off')
                continue
            
            img_info = top_k_images[idx]
            
            # Retrieve image
            image = retrieve_image_from_metadata(dataset, img_info, batch_size=batch_size)
            if len(image.shape) == 4: # Get only first image for now
                image = image[0]
            
            # Get spatial activations for this image
            spatial_acts = img_info['all_spatial_activations']

            # Create spatiotemporal map
            spatiotemporal_map = np.zeros((t_count, h_count, w_count))
            
            # Reconstruct spatial map
            # spatial_map = np.zeros((h_count, w_count))
            for act_info in spatial_acts:
                t, h, w = act_info['time_step'], act_info['spatial_h'], act_info['spatial_w']
                if t is not None and h is not None and w is not None:
                    spatiotemporal_map[t, h, w] = act_info['activation']

            # for t in range(t_count):
            #     # Fix for valid image value range
            #     spatiotemporal_map[t] = np.maximum(spatiotemporal_map[t], 0)
            #     spatiotemporal_map[t] = (spatiotemporal_map[t] - spatiotemporal_map[t].min()) / (spatiotemporal_map[t].max() - spatiotemporal_map[t].min() + 1e-8)

            spatiotemporal_maps_upsampled = []

            for t in range(t_count):
                # Upsample spatial map to image size
                spatial_map_upsampled = cv2.resize(
                    spatiotemporal_map[t],
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
                spatiotemporal_maps_upsampled.append(spatial_map_upsampled)

            sequences.append(spatiotemporal_maps_upsampled)
            images.append([restore_image(image) for _ in range(t_count)])
            
            # # Display
            # ax.imshow(restore_image(image))
            # if (pair_show and idx_super % 2 == 1) or not pair_show:
            #     ax.imshow(spatial_map_upsampled, alpha=0.5, cmap='hot', vmin=0, vmax=1)
            
            # Title with information
            ax.set_title(
                f"#{idx+1}: {img_info['label']}\n"
                f"Max={img_info['max_activation']:.2f} at "
                f"({img_info['max_location'][0]},{img_info['max_location'][1]})\n"
                f"Active: {img_info['num_active_locations']}/{h_count * w_count}",
                fontsize=10
            )
            ax.axis('off')
        else:
            raise(NotImplementedError('Not yet implemented'))
        
    # ims = []

    # Initialize axes
    for i, seq in enumerate(sequences):
        ax = axes[i]
        ax.imshow(seq[0], animated=True)
        # ax.set_title(
        #     f"#{idx+1}: {img_info['label']}\n"
        #     f"Max={img_info['max_activation']:.2f} at "
        #     f"({img_info['max_location'][0]},{img_info['max_location'][1]})\n"
        #     f"Active: {img_info['num_active_locations']}/{h_count * w_count}",
        #     fontsize=10
        # )
        # axes[i].axis('off')
        # ims.append(im)

    def animate(frame):
        for idx, (im, seq) in enumerate(zip(images, sequences)):
            ax = axes[idx]
            ax.clear()

            # Display
            # ax.imshow(restore_image(image))
            ax.imshow(im[frame], animated=True)
            if (pair_show and idx % 2 == 1) or not pair_show:
                ax.imshow(seq[frame], alpha=0.5, cmap='hot', vmin=0, vmax=1)
            
            # Title with information
            ax.set_title(
                f"#{idx+1}: {img_info['label']}\n"
                f"Max={img_info['max_activation']:.2f} at "
                f"({img_info['max_location'][0]},{img_info['max_location'][1]})\n"
                f"Active: {img_info['num_active_locations']}/{h_count * w_count}\n"
                f"at timestep {frame}",
                fontsize=10
            )
        return []
    
    fig.suptitle(
        f"Feature {feature_idx} - Top {len(top_k_images)} Images with Spatial Activations",
        fontsize=16,
        fontweight='bold'
    )

    ani = animation.FuncAnimation(
        fig, animate,
        frames=t_count,
        interval=100,
        blit=True
    )
    
    plt.tight_layout()
    return ani, fig


def save_feature_analysis_with_metadata(
    feature_idx,
    all_activations,
    all_metadata,
    save_dir='feature_reports',
    tag='default'
):
    """Save analysis results with full metadata."""
    
    from pathlib import Path
    import json
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save compact metadata (just what you need for retrieval)
    compact_metadata = []
    
    for idx, metadata in enumerate(all_metadata):
        compact_metadata.append({
            'global_idx': int(metadata['global_idx']),
            'batch_idx': int(metadata['batch_idx']),
            'image_idx': int(metadata['image_idx']),
            'label': int(metadata['image_label']),
            'spatial_h': int(metadata['spatial_h']),
            'spatial_w': int(metadata['spatial_w']),
            'activation': float(all_activations[idx])
        })
    
    # Find top-k
    top_k_indices = np.argsort(all_activations)[-100:][::-1]  # Save top-100
    
    top_k_metadata = [compact_metadata[i] for i in top_k_indices]
    
    # Save
    report = {
        'feature_idx': int(feature_idx),
        'total_activations': len(all_activations),
        'statistics': {
            'mean': float(all_activations.mean()),
            'max': float(all_activations.max()),
            'std': float(all_activations.std()),
        },
        'top_100_activations': top_k_metadata,
        # Don't save ALL metadata - would be huge!
        # Only save top activations for retrieval
    }
    
    json_path = save_dir / f'{tag}_{feature_idx:04d}' / f'feature_{feature_idx:04d}_metadata.json'
    with open(json_path, 'w') as f:
        json.dump(report, indent=2, fp=f)
    
    print(f"Saved metadata to {json_path}")
    
    return report


def load_and_visualize_saved_feature(feature_idx, dataset, save_dir='feature_reports'):
    """Load saved metadata and visualize."""
    
    import json
    from pathlib import Path
    
    json_path = Path(save_dir) / f'feature_{feature_idx:04d}_metadata.json'
    
    with open(json_path) as f:
        report = json.load(f)
    
    # Get top-k metadata
    top_k = report['top_100_activations'][:20]  # Top 20
    
    # Retrieve and visualize images
    for i, metadata in enumerate(top_k):
        image = retrieve_image_from_metadata(dataset, metadata)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.title(
            f"#{i+1}: {metadata['label']}, "
            f"Act={metadata['activation']:.3f}, "
            f"Spatial=({metadata['spatial_h']},{metadata['spatial_w']})"
        )
        plt.axis('off')
        plt.show()


def create_spatial_visualization(image, spatiotemporal_map, feature_idx, img_info):
    """
    Create a comprehensive spatial visualization showing:
    1. Original image
    2. Spatial activation heatmap
    3. Overlay of heatmap on image
    
    Args:
        image: The original image [H, W, 3]
        spatial_map: Activation values at each spatial location [7, 7]
        feature_idx: Which SAE feature
        img_info: Dict with metadata about this image
    
    Returns:
        fig: matplotlib figure
    """
    import matplotlib.pyplot as plt
    import cv2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Mark the maximum activation location
    max_t, max_h, max_w = img_info['max_location']
    
    def set_axes(image, spatial_map, t, init=False):
        # ========================================================================
        # Panel 1: Original Image
        # ========================================================================
        axes[0].imshow(restore_image(image), animated=True)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # ========================================================================
        # Panel 2: Spatial Activation Heatmap
        # ========================================================================
        # spatial_map = np.maximum(spatial_map, 0)
        # spatial_map = (spatial_map - spatial_map.min()) / (spatial_map.max() - spatial_map.min() + 1e-8)
        im1 = axes[1].imshow(spatial_map, cmap='hot', interpolation='nearest', animated=True)
        axes[1].set_title(
            f'Feature {feature_idx} Activation Map\n'
            f'Max: {img_info["max_activation"]:.3f} at '
            f'{img_info["max_location"]} at timestep {t}',
            fontsize=12,
            fontweight='bold'
        )
        
        # Add grid lines to show spatial structure
        axes[1].set_xticks(np.arange(spatial_map.shape[1]))
        axes[1].set_yticks(np.arange(spatial_map.shape[0]))
        axes[1].grid(which='both', color='white', linestyle='-', linewidth=0.5, alpha=0.3)

        if max_t is not None and max_h is not None and max_w is not None and t == max_t:
            axes[1].plot(max_w, max_h, 'r*', markersize=20, 
                        markeredgecolor='white', markeredgewidth=2)
        
        # Colorbar
        if init:
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # ========================================================================
        # Panel 3: Overlay on Original Image
        # ========================================================================
        # Upsample spatial map to match image resolution
        spatial_map_upsampled = cv2.resize(
            spatial_map,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Normalize image to [0, 1] if needed
        if image.max() > 1.0:
            image_normalized = image / 255.0
        else:
            image_normalized = image
        
        # Create overlay
        axes[2].imshow(restore_image(image_normalized), animated=True)
        axes[2].imshow(spatial_map_upsampled, alpha=0.5, cmap='hot', 
                    vmin=0, vmax=spatial_map.max(), animated=True)
        axes[2].set_title(
            f'Overlay\n'
            f'Label: {img_info["label"]}, '
            f'Mean: {img_info["mean_activation"]:.3f}',
            fontsize=12,
            fontweight='bold'
        )
        axes[2].axis('off')

    set_axes(image, spatiotemporal_map[0], 0, init=True)
    
    # Overall title
    fig.suptitle(
        f'Feature {feature_idx} Spatial Localization\n'
        f'Active locations: {img_info["num_active_locations"]}/{img_info["total_locations"]}',
        fontsize=14,
        fontweight='bold',
        y=1.02
    )

    def animate(frame):
        for i in range(3):
            ax = axes[i]
            ax.clear()
        set_axes(image, spatiotemporal_map[frame], frame)
        
        return []
    
    ani = animation.FuncAnimation(
        fig, animate,
        frames=spatiotemporal_map.shape[0],
        interval=100,  # milliseconds between frames
        blit=False
    )
    
    plt.tight_layout()
    
    return ani, fig


def visualize_feature_spatial_activation(
    sae_state, 
    target_state,
    feature_idx,
    loc,
    target_layer='conv3'
):
    """
    Show spatial activation pattern of a feature on a specific image.
    Similar to Grad-CAM but for SAE features.
    """
    # Get CNN activations with spatial structure
    image = extract_and_create_im_seqs([loc])
    acts = extract_activations(target_state, target_state.params, image, target_layer)[0]
    # Shape: [1, T, H, W, C] for conv layers
    
    batch, T, H, W, C = acts.shape
    
    # Reshape to process each spatial location
    acts_flat = acts.reshape(batch * T * H * W, C)  # [batch*T*H*W, C]
    
    # Get SAE features for each location
    sae_features, _ = sae_state.apply_fn(sae_state.params, acts_flat)
    
    # Extract our feature and reshape to spatial
    feature_map = sae_features[:, feature_idx].reshape(H, W)  # [H, W]
    
    # Upsample to original image size
    feature_map_upsampled = cv2.resize(
        feature_map, 
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )
    
    # Create heatmap overlay
    heatmap = plt.cm.jet(feature_map_upsampled)[:, :, :3]
    overlay = 0.8 * image + 0.2 * heatmap
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(feature_map_upsampled, cmap='hot')
    axes[1].set_title(f"Feature {feature_idx} Activation Map")
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig, feature_map


def analyze_and_document(
    sae_state, 
    target_state,
    target_layer, 
    dataset,
    labels_set,
    num_samples,
    feature_idx,
    save_dir='feature_reports',
    k_top_images=20,
    batch_size=64,
    t_count=8,
    tag="default",
):
    """
    Complete analysis and documentation for a single feature.
    
    Returns and saves:
    - Basic statistics
    - Top-k activating images
    - Spatial visualizations
    - Class distribution
    - Manual interpretation
    """
    
    import json
    from pathlib import Path
    from datetime import datetime
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Feature {feature_idx} Analysis")
    print(f"{'='*70}")
    
    # ========================================================================
    # 1. BASIC STATISTICS
    # ========================================================================
    print("\n1. Computing statistics...")
    
    all_activations = []
    all_labels = []
    all_images_metadata = []

    global_activation_idx = 0
    
    for batch_idx in tqdm(range(0, num_samples, batch_size)):
        locs = dataset[batch_idx:batch_idx+batch_size]
        labels = labels_set[batch_idx:batch_idx+batch_size]

        # Get activations
        images = extract_and_create_im_seqs(locs)
        acts = extract_activations(target_state, target_state.params, images, target_layer)
        B, T, H, W, C = acts.shape
        acts_flat = acts.reshape(-1, acts.shape[-1])
        
        # Get SAE features
        # sae_features, _ = sae_state.apply_fn(sae_state.params, acts_flat)
        output = sae_state.apply_fn(sae_state.params, acts_flat)
        sae_features = output.unordered_latent_acts
        feature_acts = sae_features[:, feature_idx]

        # Repeat labels for each SAE feature
        expanded_labels = np.repeat(labels, T * H * W)
        
        # all_activations.extend(feature_acts)
        # all_labels.extend(expanded_labels)
        
        # Track metadata for later retrieval
        for img_idx in range(batch_size):
            # for spatial_idx in range(math.prod(acts.shape[1:-1])):
            for t_pos in range(T):
                for h_pos in range(H):
                    for w_pos in range(W):
                        spatial_idx = h_pos * W + w_pos

                        # Calculate which activation vector this is
                        activation_vector_idx = img_idx * (math.prod(acts.shape[1:-1])) + spatial_idx
                        
                        # The actual activation value
                        act_value = feature_acts[activation_vector_idx]
                        
                        # Spatial location (if convolutional)
                        # h_pos = spatial_idx // acts.shape[-2] # W
                        # w_pos = spatial_idx % acts.shape[-2] # W
                        
                        # Store activation
                        all_activations.append(act_value)
                        
                        # Store expanded label
                        all_labels.append(labels[img_idx])
                        
                        # Store detailed metadata
                        all_images_metadata.append({
                            'global_idx': global_activation_idx,  # Unique ID
                            'batch_idx': batch_idx,               # Which batch?
                            'image_idx': img_idx,                 # Which image in batch?
                            'image_label': labels[img_idx],       # What class?
                            'spatial_idx': spatial_idx,           # Which spatial location?
                            'time_step': t_pos,                   # Time step in feature map
                            'spatial_h': h_pos,                   # Row in feature map
                            'spatial_w': w_pos,                   # Col in feature map
                            'activation_value': float(act_value), # The actual activation
                        })
                        
                        global_activation_idx += 1
    
    all_activations = np.array(all_activations)
    all_labels = np.array(all_labels)
    
    # Compute statistics
    stats = {
        'feature_idx': int(feature_idx),
        'mean': float(all_activations.mean()),
        'std': float(all_activations.std()),
        'max': float(all_activations.max()),
        'min': float(all_activations.min()),
        'median': float(np.median(all_activations)),
        'pct_active_01': float((all_activations > 0.1).mean() * 100),  # % > 0.1
        'pct_active_05': float((all_activations > 0.5).mean() * 100),  # % > 0.5
        'is_dead': bool(all_activations.max() < 0.01),
        'sparsity_level': None  # Will categorize below
    }
    
    # Categorize sparsity
    pct_active = stats['pct_active_01']
    if pct_active < 1:
        stats['sparsity_level'] = 'very_rare'
    elif pct_active < 5:
        stats['sparsity_level'] = 'rare'
    elif pct_active < 20:
        stats['sparsity_level'] = 'moderate'
    elif pct_active < 50:
        stats['sparsity_level'] = 'common'
    else:
        stats['sparsity_level'] = 'very_common'
    
    print(f"  Mean activation: {stats['mean']:.4f}")
    print(f"  Max activation: {stats['max']:.4f}")
    print(f"  Active (>0.1): {stats['pct_active_01']:.2f}%")
    print(f"  Sparsity level: {stats['sparsity_level']}")
    
    # ========================================================================
    # 2. PER-CLASS STATISTICS
    # ========================================================================
    print("\n2. Computing per-class statistics...")
    
    class_stats = {}
    unique_labels = np.unique(all_labels)
    
    for label in unique_labels:
        class_mask = all_labels == label
        class_acts = all_activations[class_mask]
        
        class_stats[str(label)] = {
            'mean': float(class_acts.mean()),
            'max': float(class_acts.max()),
            'pct_active': float((class_acts > 0.1).mean() * 100),
            'count': int(class_mask.sum())
        }
    
    # Find most discriminative class
    class_means = {label: class_stats[str(label)]['mean'] 
                   for label in unique_labels}
    most_active_class = max(class_means, key=class_means.get)
    
    print(f"  Most active class: {most_active_class} "
          f"(mean={class_means[most_active_class]:.3f})")
    
    for label in unique_labels:
        cs = class_stats[str(label)]
        print(f"    {label}: mean={cs['mean']:.3f}, "
              f"max={cs['max']:.3f}, active={cs['pct_active']:.1f}%")
    
    # ========================================================================
    # 3. TOP-K ACTIVATING EXAMPLES
    # ========================================================================
    print(f"\n3. Finding top-{k_top_images} activating images...")
    
    # Get top-k indices
    # top_k_indices = np.argsort(all_activations)[-k_top_images:][::-1]
    
    # top_k_data = {
    #     'activations': [float(all_activations[i]) for i in top_k_indices],
    #     'labels': [str(all_labels[i]) for i in top_k_indices],
    #     'metadata': [all_images_metadata[i] for i in top_k_indices]
    # }
    
    # # Actually retrieve and save the images
    # top_images = []
    # for idx in top_k_indices:
    #     metadata = all_images_metadata[idx]
    #     # Retrieve actual image from dataset
    #     # (implementation depends on your dataset structure)
    #     image = retrieve_image_from_dataset(dataset, metadata, batch_size=batch_size)
    #     top_images.append(image)

    top_k_vectors, top_k_images = find_top_k_with_metadata(all_activations, all_images_metadata)
    
    # Visualize top-k
    # fig = visualize_top_k_grid(
    #     top_images,
    #     top_k_data['activations'],
    #     top_k_data['labels'],
    #     feature_idx,
    #     rows=4,
    #     cols=5
    # )
    ani, fig = visualize_top_k_with_spatial_info(
        dataset, top_k_images, feature_idx, h_count=H, w_count=W, t_count=t_count,
        batch_size=batch_size, pair_show=True
    )

    # Retrieve actual images
    top_images_list = []
    for img_info in top_k_images:
        image = retrieve_image_from_metadata(dataset, img_info, batch_size=batch_size)
        top_images_list.append(image)
    
    # Save visualization
    fig_path = save_dir / f'{tag}_{feature_idx:04d}' / f'feature_{feature_idx:04d}_topk.gif'
    os.makedirs(save_dir / f'{tag}_{feature_idx:04d}', exist_ok=True)
    # ani.save(fig_path, dpi=150, bbox_inches='tight', writer='pillow', fps=10)
    ani.save(fig_path, dpi=150, writer='pillow', fps=10)
    plt.close(fig)
    del ani
    
    print(f"  Saved top-k visualization to {fig_path}")
    
    # ========================================================================
    # 4. SPATIAL LOCALIZATION (for top 3 images)
    # ========================================================================
    print("\n4. Computing spatial localization for top 3 images...")
    
    spatial_visualizations = []
    
    for i in range(min(3, len(top_k_images))):
        img_info = top_k_images[i]
        image = top_images_list[i]
        
        print(f"  Image {i+1}: Max={img_info['max_activation']:.3f} at "
              f"{img_info['max_location']}, Label={img_info['label']}")
        
        # Reconstruct spatial map
        T, H, W = 8, 14, 14 # Adjust based on your architecture
        spatiotemporal_map = np.zeros((T, H, W))
        
        # for spatial_data in img_info['all_spatial_data']:
        #     h, w = spatial_data['spatial_h'], spatial_data['spatial_w']
        for spatial_data in img_info['all_spatial_activations']:
            t, h, w = spatial_data['time_step'], spatial_data['spatial_h'], spatial_data['spatial_w']
            if h is not None and w is not None:
                spatiotemporal_map[t, h, w] = spatial_data['activation']
        
        # Visualize
        ani, spatial_fig = create_spatial_visualization(
            image, spatiotemporal_map, feature_idx, img_info
        )
        
        spatial_path = save_dir / f'{tag}_{feature_idx:04d}' / f'feature_{feature_idx:04d}_spatial_img{i+1}.gif'
        os.makedirs(save_dir / f'{tag}_{feature_idx:04d}', exist_ok=True)
        # spatial_fig.save(spatial_path, writer='pillow', fps=10, dpi=150, bbox_inches='tight')
        ani.save(spatial_path, writer='pillow', fps=10, dpi=150)
        plt.close(spatial_fig)
        del ani
        
        spatial_visualizations.append({
            'rank': i + 1,
            'max_activation': img_info['max_activation'],
            'mean_activation': img_info['mean_activation'],
            'max_location': img_info['max_location'],
            'path': str(spatial_path)
        })
    
    print(f"  Saved {len(spatial_visualizations)} spatial visualizations")
    
    # ========================================================================
    # 5. ACTIVATION DISTRIBUTION PLOT
    # ========================================================================
    print("\n5. Creating activation distribution plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Overall histogram
    axes[0].hist(all_activations, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(0.1, color='red', linestyle='--', 
                    label='Active threshold (0.1)', linewidth=2)
    axes[0].axvline(stats['mean'], color='green', linestyle='--',
                    label=f"Mean ({stats['mean']:.3f})", linewidth=2)
    axes[0].set_xlabel('Activation Value', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'Feature {feature_idx} Distribution', fontsize=14)
    axes[0].legend()
    axes[0].set_yscale('log')  # Log scale to see rare activations
    
    # Per-class boxplot
    class_data = [all_activations[all_labels == label] 
                  for label in unique_labels]
    bp = axes[1].boxplot(class_data, labels=unique_labels, patch_artist=True)
    
    # Color boxes by mean activation
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    axes[1].axhline(0.1, color='red', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Activation Value', fontsize=12)
    axes[1].set_xlabel('Class', fontsize=12)
    axes[1].set_title('Activation by Class', fontsize=14)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    dist_path = save_dir / f'{tag}_{feature_idx:04d}' / f'feature_{feature_idx:04d}_distribution.png'
    os.makedirs(save_dir / f'{tag}_{feature_idx:04d}', exist_ok=True)
    fig.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved distribution plot to {dist_path}")
    
    # ========================================================================
    # 6. MANUAL INTERPRETATION
    # ========================================================================
    print("\n6. Manual interpretation:")
    print(f"  View the top-k images at: {fig_path}")
    print(f"  What pattern do you see?")
    
    # In interactive mode, prompt for interpretation
    # In batch mode, leave blank for later annotation
    # interpretation = input("  Your interpretation (or press Enter to skip): ")
    interpretation = None
    
    if not interpretation:
        interpretation = "NOT_YET_INTERPRETED"
    
    # ========================================================================
    # 7. CONFIDENCE RATING (optional)
    # ========================================================================
    if interpretation != "NOT_YET_INTERPRETED":
        print("\n  How confident are you in this interpretation?")
        print("    1 = Very unsure, 2 = Somewhat unsure, 3 = Neutral,")
        print("    4 = Somewhat confident, 5 = Very confident")
        confidence_input = input("  Confidence (1-5, or Enter to skip): ")
        
        try:
            confidence = int(confidence_input)
            confidence = max(1, min(5, confidence))  # Clamp to 1-5
        except:
            confidence = None
    else:
        confidence = None
    
    # ========================================================================
    # 8. COMPILE AND SAVE REPORT
    # ========================================================================
    print("\n7. Compiling report...")
    
    # report = {
    #     'feature_idx': feature_idx,
    #     'timestamp': datetime.now().isoformat(),
        
    #     # Statistics
    #     'statistics': stats,
    #     'class_statistics': class_stats,
    #     'most_active_class': str(most_active_class),
        
    #     # Top-k information
    #     'top_k': {
    #         'k': k_top_images,
    #         'activations': top_k_data['activations'],
    #         'labels': top_k_data['labels'],
    #         'metadata': top_k_data['metadata']
    #     },
        
    #     # Spatial information
    #     'spatial_visualizations': spatial_visualizations,
        
    #     # Interpretation
    #     'interpretation': interpretation,
    #     'confidence': confidence,
        
    #     # File paths
    #     'visualizations': {
    #         'top_k_grid': str(fig_path),
    #         'distribution': str(dist_path),
    #         'spatial': [sv['path'] for sv in spatial_visualizations]
    #     }
    # }
    
    # # Save as JSON
    # json_path = save_dir / f'feature_{feature_idx:04d}_report.json'
    # with open(json_path, 'w') as f:
    #     json.dump(report, indent=2, fp=f)
    
    # print(f"  Saved report to {json_path}")

    report = save_feature_analysis_with_metadata(
        feature_idx, all_activations, all_images_metadata, tag=tag,
        save_dir=save_dir
    )
    
    # ========================================================================
    # 9. SUMMARY
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"Feature {feature_idx} Summary:")
    print(f"  Sparsity: {stats['sparsity_level']} ({stats['pct_active_01']:.1f}% active)")
    print(f"  Max activation: {stats['max']:.3f}")
    print(f"  Most active class: {most_active_class}")
    print(f"  Interpretation: {interpretation}")
    if confidence:
        print(f"  Confidence: {confidence}/5")
    print(f"{'='*70}\n")
    
    return report

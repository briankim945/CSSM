"""
Main training script for CSSM models.

Supports two architectures:
- ConvNeXt-style: Pure/Hybrid CSSM blocks with ConvNeXt structure
- ViT-style: Clean pre-norm transformer blocks with CSSM replacing attention

Training features:
- Cosine learning rate scheduling
- Gradient clipping
- Validation evaluation
- Checkpointing (orbax)
- Weights & Biases logging
"""

import argparse
import os
import time
import sys
import random
from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from tqdm import tqdm
import orbax.checkpoint as ocp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.models.convnext import ModelFactory
from src.models.cssm_vit import CSSMViT
from src.models.baseline_vit import BaselineViT
from src.data import IMAGENETTE_CLASSES, load_image_val
from xai.integrated_gradients_utils import integrated_gradients
from xai.utils import create_image_pan_seq, prep_image


class TrainState(train_state.TrainState):
    """Extended train state with additional tracking."""
    epoch: int = 0


def get_init_state(
    rng: jax.Array,
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    total_steps: int = 10000,
    warmup_steps: int = 500,
    grad_clip: float = 1.0,
) -> Tuple[TrainState, optax.GradientTransformation]:
    """
    Initialize training state with optimizer and LR schedule.

    Args:
        rng: Random key for initialization
        model: The model to train
        learning_rate: Peak learning rate
        weight_decay: Weight decay coefficient
        total_steps: Total training steps for LR schedule
        warmup_steps: Number of warmup steps
        grad_clip: Maximum gradient norm

    Returns:
        Tuple of (train_state, optimizer)
    """
    # Create dummy input for initialization
    dummy_input = jnp.ones((1, 8, 224, 224, 3))
    variables = model.init({'params': rng, 'dropout': rng}, dummy_input, training=False)
    params = variables['params']

    # Learning rate schedule: warmup + cosine decay
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=learning_rate * 0.01,
    )

    # Optimizer: AdamW with gradient clipping
    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )

    # Wrap with apply_if_finite to skip updates on NaN/Inf gradients
    # This prevents NaN propagation and helps debug where instability originates
    tx = optax.apply_if_finite(tx, max_consecutive_errors=5)

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        epoch=0,
    )

    return state, tx, variables


def display_prediction(state, current_image, current_label):
    logits, intermediates = state.apply_fn(
        {'params': state.params},
        current_image,
        training=False,
        mutable='intermediates',
    )
    prediction = jnp.argmax(logits, -1)

    print("Prediction: ", prediction)
    print("Label: ", current_label)


def main():
    parser = argparse.ArgumentParser(description='Train CSSM models')

    # Architecture selection
    parser.add_argument('--arch', type=str, choices=['convnext', 'vit', 'baseline'], default='convnext',
                        help='Architecture: convnext, vit (CSSM-ViT), or baseline (standard ViT)')

    # Model configuration (ConvNeXt-style)
    parser.add_argument('--mode', type=str, choices=['pure', 'hybrid'], default='pure',
                        help='[convnext] Model mode: pure (CSSM replaces conv) or hybrid')
    parser.add_argument('--cssm', type=str, choices=['standard', 'opponent'], default='opponent',
                        help='CSSM type: standard or opponent (gated)')
    parser.add_argument('--mixing', type=str, choices=['dense', 'depthwise'], default='depthwise',
                        help='Mixing type: dense (multi-head) or depthwise')
    parser.add_argument('--no_concat_xy', action='store_true',
                        help='Disable [X,Y] concat+project in GatedOpponentCSSM')
    parser.add_argument('--gate_activation', type=str, default='sigmoid',
                        choices=['sigmoid', 'softplus_clamped', 'tanh_scaled'],
                        help='Gate activation for coupling (sigmoid=bounded [0,1], default)')

    # Model configuration (ViT-style and baseline)
    parser.add_argument('--embed_dim', type=int, default=384,
                        help='[vit/baseline] Embedding dimension (192=tiny, 384=small, 768=base)')
    parser.add_argument('--depth', type=int, default=12,
                        help='[vit/baseline] Number of transformer blocks')
    parser.add_argument('--patch_size', type=int, default=16,
                        help='[vit/baseline] Patch size for stem')
    parser.add_argument('--no_pos_embed', action='store_true',
                        help='[vit/baseline] Disable position embeddings')
    parser.add_argument('--num_heads', type=int, default=6,
                        help='[baseline] Number of attention heads')
    parser.add_argument('--no_temporal_attn', action='store_true',
                        help='[baseline] Disable temporal attention blocks')
    parser.add_argument('--temporal_attn_every', type=int, default=3,
                        help='[baseline] Add temporal attention every N spatial blocks')
    
    # Data configuration
    parser.add_argument('--seq_len', type=int, default=8,
                        help='Sequence length (number of frames)')
    parser.add_argument('--target_class', type=str, default='n03417042',
                        help='Imagenette class being targetted')
    parser.add_argument('--seq_pan', action='store_true',
                        help='Create a panning sequence from a single image instead of repeating')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set random seed
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    num_classes = len(IMAGENETTE_CLASSES)

    # Create model based on architecture
    if args.arch == 'vit':
        model = CSSMViT(
            num_classes=num_classes,
            embed_dim=args.embed_dim,
            depth=args.depth,
            patch_size=args.patch_size,
            cssm_type=args.cssm,
            dense_mixing=(args.mixing == 'dense'),
            concat_xy=not args.no_concat_xy,
            gate_activation=args.gate_activation,
            use_pos_embed=not args.no_pos_embed,
        )
    elif args.arch == 'baseline':
        model = BaselineViT(
            num_classes=num_classes,
            embed_dim=args.embed_dim,
            depth=args.depth,
            patch_size=args.patch_size,
            num_heads=args.num_heads,
            use_temporal_attn=not args.no_temporal_attn,
            temporal_attn_every=args.temporal_attn_every,
            use_pos_embed=not args.no_pos_embed,
        )
    else:
        model = ModelFactory(
            mode=args.mode,
            cssm_type=args.cssm,
            mixing=args.mixing,
            num_classes=num_classes,
            concat_xy=not args.no_concat_xy,
            gate_activation=args.gate_activation,
        )

    # Initialize training state
    state, _, variables = get_init_state(
        rng=init_rng,
        model=model,
    )

    # Count parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"Model parameters: {num_params:,}\n")

    # Setup checkpointing (must be absolute path for orbax)
    checkpointer = ocp.StandardCheckpointer()

    #### RESTORE

    explore_state = checkpointer.restore(
        # '/users/bkim53/code_directories/CSSM/checkpoints/vit_standard_d12_e384_v1/epoch_45',
        '/Users/briankim/Documents/Research/OneVision/CepstralSSM.nosync/CSSM/checkpoints/vit_standard_d12_e384/epoch_100',
        target=state
        # abstract_my_tree
    )

    for c in IMAGENETTE_CLASSES:
        print(f"FOR CLASS {c}")
        print()
        for _ in range(10):

            ### TABULATION
            # random_file_name = random.choice(os.listdir(f"data/{args.target_class}"))
            random_file_name = random.choice(os.listdir(f"data/{c}"))
            # target_path = f'data/{args.target_class}/{random_file_name}'
            target_path = f'data/{c}/{random_file_name}'
            print(random_file_name)

            # Get individual feature
            if args.seq_pan:
                train_features, display_features = create_image_pan_seq(target_path, jump=16)
                train_features = train_features[np.newaxis, :, :, :, :]
                display_features = display_features[np.newaxis, :, :, :, :]
            else:
                train_features = np.repeat(
                    load_image_val(target_path)[np.newaxis, np.newaxis, :, :, :],
                    args.seq_len,
                    axis=1
                )
                display_features = np.repeat(
                    np.array(Image.open(target_path).convert('RGB'))[np.newaxis, np.newaxis, :, :, :],
                    args.seq_len,
                    axis=1
                )
            
            class_to_idx = {c: i for i, c in enumerate(IMAGENETTE_CLASSES)}
            # train_labels = np.array([class_to_idx[args.target_class]])
            train_labels = np.array([class_to_idx[c]])


            curr_image = train_features[None,0]
            curr_label = train_labels[None,0].item()
            print(curr_image.shape, curr_label)

            attributions = integrated_gradients(
                explore_state, curr_image, curr_label
            )
            print("attributions:", attributions.shape)
            attributions = prep_image(attributions[0], flatten_channels=True)
            print("Attributions prepped!")

            assert args.seq_len == attributions.shape[0]

            fig, ax = plt.subplots()

            # ax.gray()
            ax.imshow(prep_image(load_image_val(target_path)), animated=True)
            ax.imshow(attributions[0], animated=True, alpha=0.5)
            ax.axis('off')

            def animate(frame):
                ax.clear()

                # Display
                # ax.gray()
                # ax.imshow(attributions[frame], animated=True)
                ax.imshow(prep_image(load_image_val(target_path)), animated=True)
                ax.imshow(attributions[frame], animated=True, alpha=0.5)
                ax.axis('off')
                
                # Title with information
                ax.set_title(f"Attribution Map for {target_path} at frame {frame}")
                return []

            ani = animation.FuncAnimation(
                fig, animate,
                frames=args.seq_len,
                interval=100,
                blit=True
            )

            # ani.save(f"integrated_gradients_outputs/{args.target_class}_{random_file_name.split('.')[0]}.gif")
            os.makedirs(f"integrated_gradients_outputs/{c}", exist_ok=True)
            ani.save(f"integrated_gradients_outputs/{c}/{random_file_name.split('.')[0]}.gif")
            plt.close(fig)
            del ani

            print()
        print()


if __name__ == '__main__':
    main()

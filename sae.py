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
from flax.core.frozen_dict import freeze
from tqdm import tqdm
import orbax.checkpoint as ocp
import numpy as np
from PIL import Image

from src.models.convnext import ModelFactory
from src.models.cssm_vit import CSSMViT, cssm_vit_tiny, cssm_vit_small
from src.models.baseline_vit import BaselineViT, baseline_vit_tiny, baseline_vit_small
from src.data import IMAGENETTE_CLASSES, load_image_val, get_dataset_info, get_imagenette_video_loader_train_val_split
from src.pathfinder_data import get_pathfinder_loader, get_pathfinder_info
from xai.sae_utils import extract_activations_with_capture, save_activations, train_sae_on_activations, \
        load_activations, SparseCoder, \
        find_alive_features, find_class_specific_features, random_feature_sample, \
        features_by_max_activation, analyze_and_document


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
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per device')
    parser.add_argument('--create_act', action='store_true',
                        help='Create activations from data folder')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data folder to create activations from')
    
    # SAE configuration
    parser.add_argument('--act_train', action='store_true',
                        help='Flag to train a new SAE model')
    parser.add_argument('--act_source', type=str, default='activation.npz',
                        help='NPZ file to pull activations from')
    parser.add_argument('--act_save', type=str, default='checkpoints/sae_trained',
                        help='Location to checkpoint a saved SAE model')
    parser.add_argument('--act_val_source', type=str, default='activations/val_cssmblocks_8_9_10_11.npz',
                        help='Location to val data activations')
    parser.add_argument('--act_val_labels', type=str, default='activations/val_labels.npz',
                        help='Location to val labels')
    parser.add_argument('--act_val_names', type=str, default='activations/val_names.npz',
                        help='Location to val names')
    parser.add_argument('--transcoder', action='store_true',
                        help='Flag to train a transcoder model')
    parser.add_argument('--skip_connect', action='store_true',
                        help='Flag to include a skip connection in the transcoder model')
    
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

    lr = 1e-5

    # Get dataloaders
    if args.create_act:
        train_loader, val_loader = get_imagenette_video_loader_train_val_split(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            sequence_length=args.seq_len,
            split="",
            return_names=True,
            class_sample_size=300,
        )

        train_activations, train_labels, train_names = extract_activations_with_capture(
            explore_state, explore_state.params, train_loader,
            ['block8', 'block9', 'block10', 'block11']
        )
        save_activations(train_activations, 'activations/train_cssmblocks_8_9_10_11.npz')
        save_activations({"labels": train_labels}, 'activations/train_labels.npz')
        save_activations({"names": train_names}, 'activations/train_names.npz')
        val_activations, val_labels, val_names = extract_activations_with_capture(
            explore_state, explore_state.params, val_loader,
            ['block8', 'block9', 'block10', 'block11']
        )
        save_activations(val_activations, 'activations/val_cssmblocks_8_9_10_11.npz')
        save_activations({"labels": val_labels}, 'activations/val_labels.npz')
        save_activations({"names": val_names}, 'activations/val_names.npz')
    elif args.act_train:
        if args.transcoder:
            x_layer_name = 'block9'
            y_layer_name = 'block11'
            d_sae = 16384

            train_activations = load_activations(args.act_source)
            layer_act_inputs = train_activations[x_layer_name]
            layer_act_targets = train_activations[y_layer_name]
            print("Layer activations:")
            print(f"\tType: {type(layer_act_inputs)}")
            print(f"\tShape: {layer_act_inputs.shape}")
            sae_state, sae_model = train_sae_on_activations(
                layer_act_inputs,
                y=layer_act_targets,
                d_sae=d_sae,
                transcode=True,
                learning_rate=lr,
                skip_connection=args.skip_connect,
                num_epochs=20,
            )
            checkpointer.save(f"{args.act_save}_{d_sae}_layer_skip_connect_{args.skip_connect}_{x_layer_name}_to_{y_layer_name}_reshape", sae_state)
        else:
            layer_name = 'block11'

            train_activations = load_activations(args.act_source)
            layer_train_activations = train_activations['block11']
            print("Layer activations:")
            print(f"\tType: {type(layer_train_activations)}")
            print(f"\tShape: {layer_train_activations.shape}")
            sae_state, sae_model = train_sae_on_activations(layer_train_activations)
            checkpointer.save(f"{args.act_save}_layer_{layer_name}_reshape", sae_state)
    else:
        if args.transcoder:
            x_layer_name = 'block9'
            y_layer_name = 'block11'
            layer_name = 'block8'
            s_dir = 'feature_reports/transcoder'
            d_sae = 8192
        else:
            layer_name = 'block11'
            s_dir = 'feature_reports'
            d_sae = 8192

        # Load validation activations
        val_activations = load_activations(args.act_val_source)[layer_name]
        val_labels = np.load(args.act_val_labels)['labels']
        val_names = np.load(args.act_val_names, allow_pickle=True)['names']
        print(val_activations.shape, val_labels.shape)
        # print(val_labels[:5])
        # print(val_names[:5])

        num_samples = val_names.shape[0]
        b_size = 8

        # if len(val_activations.shape) > 2:
        #     val_activations = jnp.reshape(val_activations, (-1, val_activations.shape[-1]))

        # val_dataset_zipped = zip(val_activations, val_labels)
        # val_dataset_zipped = list(zip(val_names, val_labels))

        # Restore model
        d_in = val_activations.shape[-1]
        # d_in = val_names.shape[-1]
        # sae_model = SAE(d_in=d_in, d_sae=d_sae)
        k_features = 32
        sae_model = SparseCoder(d_in, d_sae, k=k_features)
        params = sae_model.init(rng, jnp.ones((1, d_in)))
        tx = optax.adam(lr)
        sae_state = train_state.TrainState.create(
            apply_fn=sae_model.apply,
            params=params,
            tx=tx
        )
        if args.transcoder:
            restored_sae_state = checkpointer.restore(
                # '/users/bkim53/code_directories/CSSM/checkpoints/vit_standard_d12_e384_v1/epoch_45',
                f'/Users/briankim/Documents/Research/OneVision/CepstralSSM.nosync/CSSM/checkpoints/transcoder_trained_layer_{x_layer_name}_to_{y_layer_name}_reshape',
                target=sae_state
                # abstract_my_tree
            )
        else:
            restored_sae_state = checkpointer.restore(
                # '/users/bkim53/code_directories/CSSM/checkpoints/vit_standard_d12_e384_v1/epoch_45',
                '/Users/briankim/Documents/Research/OneVision/CepstralSSM.nosync/CSSM/checkpoints/sparsecoder_trained_layer_block11_reshape',
                target=sae_state
                # abstract_my_tree
            )

        # # Top-k images for a feature
        # find_top_k_activating_images(
        #     restored_sae_state, feature_idx,
        #     explore_state, layer_name,
        #     val_dataset
        # )

        # Pre-work Find alive features
        # feature_stats = find_alive_features(
        #     restored_sae_state, explore_state, layer_name, val_names, k_features, batch_size=b_size)
        feature_stats = find_alive_features(
            restored_sae_state, explore_state, layer_name, val_names, d_sae, batch_size=b_size)
        #     restored_sae_state, explore_state, layer_name, val_activations, d_sae)

        # Random 20 features
        random_20 = random_feature_sample(feature_stats['alive'], n=20)
        print(f"Random 20: {random_20}")
        for feat_idx in random_20:
            analyze_and_document(
                restored_sae_state, explore_state, layer_name, val_names, val_labels,
                num_samples, feat_idx, save_dir=s_dir, batch_size=b_size, tag="random20")

        # Top features by class
        class_features = find_class_specific_features(
            restored_sae_state, explore_state, layer_name, val_names, val_labels,
            batch_size=b_size)
        for class_name, features in class_features.items():
            print(f"\n{class_name} features:")
            print(f"\t{features[:5]}")
            for feat_idx, _ in features[:5]:
                analyze_and_document(
                    restored_sae_state, explore_state, layer_name, val_names, val_labels,
                    num_samples, feat_idx, save_dir=s_dir, batch_size=b_size, tag=f"topclass_{class_name}")

        # Top 20 by activation
        top_20 = features_by_max_activation(feature_stats, top_n=20)
        print(f"Top 20: {top_20}")
        for feat_idx in top_20:
            analyze_and_document(
                restored_sae_state, explore_state, layer_name, val_names, val_labels,
                num_samples, feat_idx, save_dir=s_dir, batch_size=b_size, tag="top20act")


if __name__ == '__main__':
    main()

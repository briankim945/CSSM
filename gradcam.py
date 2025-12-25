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

from src.models.convnext import ModelFactory
from src.models.cssm_vit import CSSMViT, cssm_vit_tiny, cssm_vit_small
from src.models.baseline_vit import BaselineViT, baseline_vit_tiny, baseline_vit_small
from src.data import get_imagenette_video_loader, get_dataset_info, get_imagenette_video_loader_train_val_split
from src.pathfinder_data import get_pathfinder_loader, get_pathfinder_info


class TrainState(train_state.TrainState):
    """Extended train state with additional tracking."""
    epoch: int = 0


def create_train_state(
    rng: jax.Array,
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    total_steps: int,
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


@partial(jax.jit, static_argnums=(3,))
def train_step(
    state: TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    rng: jax.Array,
    num_classes: int,
) -> Tuple[TrainState, Dict[str, float]]:
    """
    Single training step.

    Args:
        state: Current training state
        batch: Tuple of (videos, labels)
        rng: Random key for dropout
        num_classes: Number of output classes

    Returns:
        Tuple of (updated_state, metrics_dict)
    """
    videos, labels = batch

    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            videos,
            training=True,
            rngs={'dropout': rng},
        )
        one_hot = jax.nn.one_hot(labels, num_classes)
        loss = optax.softmax_cross_entropy(logits, one_hot).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)

    # Update state
    state = state.apply_gradients(grads=grads)

    # Compute metrics
    acc = jnp.mean(jnp.argmax(logits, -1) == labels)

    metrics = {
        'train_loss': loss,
        'train_acc': acc,
    }

    return state, metrics


@partial(jax.jit, static_argnums=(2,))
def eval_step(
    state: TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    num_classes: int,
) -> Dict[str, float]:
    """
    Single evaluation step.

    Args:
        state: Current training state
        batch: Tuple of (videos, labels)
        num_classes: Number of output classes

    Returns:
        Dictionary of metrics
    """
    videos, labels = batch

    logits = state.apply_fn(
        {'params': state.params},
        videos,
        training=False,
    )

    one_hot = jax.nn.one_hot(labels, num_classes)
    loss = optax.softmax_cross_entropy(logits, one_hot).mean()
    acc = jnp.mean(jnp.argmax(logits, -1) == labels)

    return {
        'val_loss': loss,
        'val_acc': acc,
    }


def evaluate(
    state: TrainState,
    val_loader,
    num_classes: int,
    num_batches: int = None,
) -> Dict[str, float]:
    """
    Run full validation evaluation.

    Args:
        state: Current training state
        val_loader: Validation data iterator
        num_classes: Number of output classes
        num_batches: Max batches to evaluate (None = all)

    Returns:
        Dictionary of averaged metrics
    """
    total_loss = 0.0
    total_acc = 0.0
    count = 0

    for i, batch in enumerate(val_loader):
        if num_batches is not None and i >= num_batches:
            break

        metrics = eval_step(state, batch, num_classes)
        total_loss += float(metrics['val_loss'])
        total_acc += float(metrics['val_acc'])
        count += 1

    return {
        'val_loss': total_loss / max(count, 1),
        'val_acc': total_acc / max(count, 1),
    }


def loss_fn(state, params, perturbations, X, Y):
    variables = {
        'params': params,
        'perturbations': perturbations
    }
    logits = state.apply_fn(variables, X, training=False)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, Y)
    loss = loss.mean()
    return loss


def prep_image(image):
    # Normalize the gradient values to be between 0-1
    max_val= np.max(image)
    min_val = np.min(image)
    image = (image - min_val) / (max_val - min_val)
    # Convert the grads to uint8 for displaying
    image = np.uint8(image * 255)
    return image

def prep_video(video, flatten=True):
    pass


def display_prediction(state, current_image, current_label):
    # current_image = X_val[None, index]
    # rng = jax.random.PRNGKey(42)
    # init_variables = forward_fn.init(rng, current_image)
    # init_perturbations = init_variables["perturbations"]
    # prediction, state = make_predictions(X_val[None, index], params, init_perturbations)
    logits = state.apply_fn(
        {'params': state.params},
        current_image,
        training=False,
        mutable='intermediates',
    )
    prediction = jnp.argmax(logits, -1)

    # label = Y_val[index]
    print("Prediction: ", prediction)
    print("Label: ", current_label)

    # display_image = current_image.reshape((28, 28)) * 255
    # plt.gray()
    # plt.imshow(display_image, interpolation='nearest')
    # plt.axis('off')
    # plt.title("Input Image")
    # plt.show()

    # return

    # Extract final conv layers values
    # final_conv_layer = state["intermediates"]["final_conv_layer"][0][0]
    target_layer = state.params['block11']['cssm']['kernel']
    # Get final conv gradients
    perturbations = freeze({"target": target_layer})
    # final_conv_grads = grad(loss_fn, argnums=1)(params, perturbations, current_image, label[None, ...])
    target_grads = jax.grad(loss_fn, argnums=1)(state, state.params, perturbations, current_image, current_label)
    print(target_grads.keys())
    # target_grads = target_grads["target"]
    target_grads = target_grads['block11']['cssm']['kernel']
    print(target_grads.shape)
    # Get weights using global average pooling
    weights = jnp.mean(target_grads, axis=(0, 1))
    # Get the weighted sum of all the filters
    cam = jnp.dot(target_layer, weights)
    cam = prep_image(cam)
    print(cam.shape)

    # plt.gray()
    # plt.imshow(cam, interpolation='nearest')
    # plt.axis('off')
    # plt.title("Attribution Map")
    # plt.show()


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

    # Training configuration
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per device')
    parser.add_argument('--seq_len', type=int, default=8,
                        help='Sequence length (number of frames)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Peak learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay coefficient')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm')

    # Logging and checkpointing
    parser.add_argument('--project', type=str, default='cssm-convnext',
                        help='Wandb project name')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_every', type=int, default=10,
                        help='Log metrics every N steps')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='Evaluate every N epochs')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--run_label', type=str, default=None,
                        help='Any labels to add to the run name')                   

    # Data
    parser.add_argument('--dataset', type=str, choices=['imagenette', 'pathfinder'], default='imagenette',
                        help='Dataset to use: imagenette (video) or pathfinder (static)')
    parser.add_argument('--data_dir', type=str,
                        default='/media/data_cifs/projects/prj_video_imagenet/fftconv/data/imagenette2-320',
                        help='Path to dataset directory')
    parser.add_argument('--pathfinder_difficulty', type=str, choices=['9', '14', '20'], default='9',
                        help='[pathfinder] Contour length difficulty (9=easy, 20=hard)')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')

    args = parser.parse_args()

    # Generate run name from config
    if args.arch == 'vit':
        run_name = f"vit_{args.cssm}_d{args.depth}_e{args.embed_dim}"
    elif args.arch == 'baseline':
        temporal_str = f"_temp{args.temporal_attn_every}" if not args.no_temporal_attn else "_notime"
        run_name = f"baseline_d{args.depth}_e{args.embed_dim}_h{args.num_heads}{temporal_str}"
    else:
        run_name = f"{args.mode}_{args.cssm}_{args.mixing}"

    # Add dataset to run name
    if args.dataset == 'pathfinder':
        run_name = f"pf{args.pathfinder_difficulty}_{run_name}"

    # Add run label, if any
    if args.run_label is not None:
        run_name = f"{run_name}_{args.run_label}"
    print(f"\n{'='*60}")
    print(f"Running Configuration: {run_name}")
    print(f"Architecture: {args.arch.upper()}")
    print(f"{'='*60}\n")

    # Set random seed
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    # Get dataset info
    if args.dataset == 'pathfinder':
        # Set default data dir for pathfinder if not specified
        if 'imagenette' in args.data_dir:
            args.data_dir = '/media/data_cifs_lrs/projects/prj_LRA/PathFinder/pathfinder300_new_2025'
        dataset_info = get_pathfinder_info(args.pathfinder_difficulty)
        dataset_name = f"Pathfinder (difficulty={args.pathfinder_difficulty})"
    else:
        dataset_info = get_dataset_info()
        dataset_name = "Imagenette"

    num_classes = dataset_info['num_classes']
    train_size = dataset_info['train_size']
    steps_per_epoch = train_size // args.batch_size
    total_steps = steps_per_epoch * args.epochs

    print(f"Dataset: {dataset_name}")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Train samples: {train_size}")
    print(f"  Num classes: {num_classes}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}\n")

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
    state, _, variables = create_train_state(
        rng=init_rng,
        model=model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        total_steps=total_steps,
        grad_clip=args.grad_clip,
    )

    # Count parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"Model parameters: {num_params:,}\n")

    # Setup checkpointing (must be absolute path for orbax)
    checkpoint_dir = os.path.abspath(os.path.join(args.checkpoint_dir, run_name))
    # os.makedirs(checkpoint_dir, exist_ok=True)

    checkpointer = ocp.StandardCheckpointer()
    # checkpoint_path = ocp.test_utils.erase_and_create_empty(checkpoint_dir)

    # Training loop
    global_step = 0
    best_val_acc = 0.0

    ### TABULATION
    train_loader, val_loader = get_imagenette_video_loader_train_val_split(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        sequence_length=args.seq_len,
    )
    train_features, train_labels = next(iter(train_loader))
    print(model.tabulate(jax.random.PRNGKey(64), train_features))
    print()

    #### RESTORE

    restored_state = checkpointer.restore(
        # '/users/bkim53/code_directories/CSSM/checkpoints/vit_standard_d12_e384_v1/epoch_45',
        '/users/bkim53/code_directories/CSSM/checkpoints/vit_standard_d12_e384/epoch_100',
        target=state
        # abstract_my_tree
    )
    # print(raw_restore)

    # assert jax.tree_util.tree_all(jax.tree.map(lambda x, y: (x == y).all(), state.params, restored_state.params))
    print(restored_state.params.keys())
    print(restored_state.params['block11'].keys())
    print(restored_state.params['block11']['cssm'].keys())
    print(restored_state.params['block11']['cssm']['kernel'].shape)
    print(variables['params'].keys())
    print('perturbations' in variables)
    # assert jax.tree_util.tree_all(jax.tree.map(lambda x, y: (x == y).all(), state.params, variables['params']))
    # print(variables['perturbations'].keys())

    val_metrics = evaluate(restored_state, val_loader, num_classes)
    print(val_metrics)

    curr_image = train_features[None,0]
    curr_label = train_labels[None,0]
    display_prediction(restored_state, curr_image, curr_label)


if __name__ == '__main__':
    main()

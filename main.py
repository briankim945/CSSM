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
import re
import time
from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
import wandb
from flax import linen as nn
from flax.training import train_state
from tqdm import tqdm
import orbax.checkpoint as ocp

from src.models.convnext import ModelFactory
from src.models.cssm_vit import CSSMViT, cssm_vit_tiny, cssm_vit_small
from src.models.baseline_vit import BaselineViT, baseline_vit_tiny, baseline_vit_small
from src.models.deit3 import DeiT3Large, deit3_large_patch16_384, deit3_base_patch16_224
from src.models.cssm_deit3 import CSSMDeiT3Large, cssm_deit3_large_patch16_384, cssm_deit3_base_patch16_224
from src.models.cssm_shvit import CSSMSHViT
from src.models.shvit import SHViT
from src.data import get_imagenette_video_loader, get_dataset_info #, get_imagenet_loader, get_imagenet_info
from src.pathfinder_data import get_pathfinder_loader, get_pathfinder_info, get_pathfinder_tfrecord_loader


class TrainState(train_state.TrainState):
    """Extended train state with additional tracking."""
    batch_stats: dict
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
        batch_stats=variables['batch_stats'] if 'batch_stats' in variables else {},
        tx=tx,
        epoch=0,
    )

    return state, tx


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
        logits, updates = state.apply_fn(
            {'params': params},
            videos,
            training=True,
            rngs={'dropout': rng},
            mutable=['batch_stats'],
        )
        one_hot = jax.nn.one_hot(labels, num_classes)
        loss = optax.softmax_cross_entropy(logits, one_hot).mean()
        return loss, (logits, updates)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = grad_fn(state.params)

    # Update state
    state = state.apply_gradients(grads=grads)
    if 'batch_stats' in updates:
        state = state.replace(batch_stats=updates['batch_stats'])

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

    variables = {'params': state.params, 'batch_stats': state.batch_stats}

    logits = state.apply_fn(
        variables,
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


def main():
    parser = argparse.ArgumentParser(description='Train CSSM models')

    # Architecture selection
    parser.add_argument('--arch', type=str,
                        choices=['convnext', 'vit', 'baseline', 'deit3', 'cssm_deit3', 'cssm_shvit', 'shvit'],
                        default='convnext',
                        help='Architecture: convnext, vit (CSSM-ViT), baseline (ViT), deit3, cssm_deit3')

    # Model configuration (ConvNeXt-style)
    parser.add_argument('--mode', type=str, choices=['pure', 'hybrid'], default='pure',
                        help='[convnext] Model mode: pure (CSSM replaces conv) or hybrid')
    parser.add_argument('--cssm', type=str, choices=['standard', 'gated', 'opponent', 'bilinear', 'linear', 'linear_opponent', 'hgru', 'hgru_bi'], default='opponent',
                        help='CSSM type: hgru (2x2 linear opponent), hgru_bi (3x3 with Z interaction channel), opponent (2x2 basic), standard, gated')
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
                        help='[vit/baseline] Patch size (only used with --stem_mode patch)')
    parser.add_argument('--stem_mode', type=str, default='conv',
                        choices=['patch', 'conv', 'resnet'],
                        help='[vit] Stem: patch (ViT-style), conv (SHViT overlapping), resnet (7x7+pool)')
    parser.add_argument('--no_pos_embed', action='store_true',
                        help='[vit/baseline] Disable position embeddings')
    parser.add_argument('--num_heads', type=int, default=6,
                        help='[baseline] Number of attention heads')
    parser.add_argument('--no_temporal_attn', action='store_true',
                        help='[baseline] Disable temporal attention blocks')
    parser.add_argument('--temporal_attn_every', type=int, default=3,
                        help='[baseline] Add temporal attention every N spatial blocks')
    parser.add_argument('--rope_mode', type=str, default='none',
                        choices=['spatiotemporal', 'temporal', 'none'],
                        help='[vit] RoPE position encoding mode')
    parser.add_argument('--block_size', type=int, default=1,
                        help='[vit] Block size for LMME channel mixing (1=depthwise, >1=channel mixing). Works with gated and opponent.')
    parser.add_argument('--kernel_size', type=int, default=15,
                        help='[vit] Spatial kernel size for CSSM excitation/inhibition kernels')
    parser.add_argument('--use_dwconv', action='store_true',
                        help='[vit] Use DWConv in MLP (adds params, matches SHViT)')
    parser.add_argument('--output_act', type=str, default='none',
                        choices=['none', 'gelu', 'silu'],
                        help='[vit] Output activation after CSSM (adds nonlinearity)')
    parser.add_argument('--layer_scale_init', type=float, default=1e-6,
                        help='[vit] Layer scale initialization (1e-6 for stable deep training, 1.0 for shallow)')
    parser.add_argument('--gate_rank', type=int, default=0,
                        help='[vit] Low-rank gate bottleneck (0=full rank, try 16-64 to reduce gate params)')

    # Training configuration
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per device')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (0=sequential)')
    parser.add_argument('--prefetch_batches', type=int, default=2,
                        help='Number of batches to prefetch')
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
    parser.add_argument('--run_name', type=str, default=None,
                        help='Run name for checkpoints and wandb (default: auto-generated from config)')
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

    # Data
    parser.add_argument('--dataset', type=str, choices=['imagenette', 'pathfinder', 'imagenet'], default='imagenette',
                        help='Dataset: imagenette, pathfinder, or imagenet')
    parser.add_argument('--data_dir', type=str,
                        default='/media/data_cifs/projects/prj_video_imagenet/fftconv/data/imagenette2-320',
                        help='Path to dataset directory')
    parser.add_argument('--imagenet_dir', type=str,
                        default='/gpfs/data/shared/imagenet/ILSVRC2012',
                        help='Path to ImageNet directory')
    parser.add_argument('--pathfinder_difficulty', type=str, choices=['9', '14', '20'], default='9',
                        help='[pathfinder] Contour length difficulty (9=easy, 20=hard)')
    parser.add_argument('--tfrecord_dir', type=str, default=None,
                        help='[pathfinder] TFRecord directory (if set, uses TFRecord loader for max I/O perf)')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size (224 or 384 for DeiT3)')

    # Checkpoint resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., checkpoints/run_name/epoch_190)')
    parser.add_argument('--resume_epoch', type=int, default=None,
                        help='Override epoch to resume from (default: auto-detect from checkpoint)')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')

    args = parser.parse_args()

    # Generate run name from config (or use provided name)
    if args.run_name:
        run_name = args.run_name
    else:
        if args.arch == 'vit':
            run_name = f"vit_{args.cssm}_d{args.depth}_e{args.embed_dim}"
        elif args.arch == 'baseline':
            temporal_str = f"_temp{args.temporal_attn_every}" if not args.no_temporal_attn else "_notime"
            run_name = f"baseline_d{args.depth}_e{args.embed_dim}_h{args.num_heads}{temporal_str}"
        elif args.arch == 'cssm_shvit':
            run_name = f"cssm_shvit_{args.cssm}_e{args.embed_dim}"
        elif args.arch == 'shvit':
            run_name = f"shvit"#f"shvit_{args.cssm}_e{args.embed_dim}"
        else:
            run_name = f"{args.mode}_{args.cssm}_{args.mixing}"

        # Add dataset to run name
        if args.dataset == 'pathfinder':
            run_name = f"pf{args.pathfinder_difficulty}_{run_name}"
    print(f"\n{'='*60}")
    print(f"Running Configuration: {run_name}")
    print(f"Architecture: {args.arch.upper()}")
    print(f"{'='*60}\n")

    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.project,
            name=run_name,
            config=vars(args),
        )

    # Set random seed
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    # Get dataset info
    if args.dataset == 'pathfinder':
        # Set default data dir for pathfinder if not specified
        if 'imagenette' in args.data_dir:
            args.data_dir = '/media/data_cifs_lrs/projects/prj_LRA/PathFinder/pathfinder300_new_2025'
        dataset_info = get_pathfinder_info(args.pathfinder_difficulty, root=args.data_dir)
        dataset_name = f"Pathfinder (difficulty={args.pathfinder_difficulty})"
    elif args.dataset == 'imagenet':
        dataset_info = get_imagenet_info()
        dataset_name = "ImageNet"
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
            stem_mode=args.stem_mode,
            cssm_type=args.cssm,
            dense_mixing=(args.mixing == 'dense'),
            concat_xy=not args.no_concat_xy,
            gate_activation=args.gate_activation,
            use_pos_embed=not args.no_pos_embed,
            rope_mode=args.rope_mode,
            block_size=args.block_size,
            gate_rank=args.gate_rank,
            kernel_size=args.kernel_size,
            use_dwconv=args.use_dwconv,
            output_act=args.output_act,
            layer_scale_init=args.layer_scale_init,
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
    elif args.arch == 'deit3':
        model = DeiT3Large(
            num_classes=num_classes,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            patch_size=args.patch_size,
        )
    elif args.arch == 'cssm_deit3':
        model = CSSMDeiT3Large(
            num_classes=num_classes,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            patch_size=args.patch_size,
            cssm_type=args.cssm,
            dense_mixing=(args.mixing == 'dense'),
            concat_xy=not args.no_concat_xy,
            gate_activation=args.gate_activation,
            num_timesteps=args.seq_len,
        )
    elif args.arch == 'cssm_shvit':
        model = CSSMSHViT(
            num_classes=num_classes,
            # embed_dim=args.embed_dim,
            cssm_type=args.cssm,
            dense_mixing=(args.mixing == 'dense'),
            block_size=args.block_size,
            gate_activation=args.gate_activation,
            num_timesteps=args.seq_len,
            rope_mode=args.rope_mode,
            use_dwconv=args.use_dwconv,
            output_act=args.output_act,
        )
    elif args.arch == 'shvit':
        model = SHViT(
            num_classes=num_classes,
            # embed_dim=args.embed_dim,
            # cssm_type=args.cssm,
            # dense_mixing=(args.mixing == 'dense'),
            # block_size=args.block_size,
            # gate_activation=args.gate_activation,
            # num_timesteps=args.seq_len,
            # rope_mode=args.rope_mode,
            # use_dwconv=args.use_dwconv,
            # output_act=args.output_act,
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
    state, _ = create_train_state(
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

    if not args.no_wandb:
        wandb.config.update({'num_params': num_params})

    # Setup checkpointing (must be absolute path for orbax)
    checkpoint_dir = os.path.abspath(os.path.join(args.checkpoint_dir, run_name))
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpointer = ocp.StandardCheckpointer()

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    best_val_acc = 0.0

    if args.resume:
        resume_path = os.path.abspath(args.resume)
        if os.path.exists(resume_path):
            print(f"\n{'='*60}")
            print(f"Resuming from checkpoint: {resume_path}")
            print(f"{'='*60}\n")

            # Restore the state (params + optimizer state)
            state = checkpointer.restore(resume_path, state)

            # Determine starting epoch
            if args.resume_epoch is not None:
                start_epoch = args.resume_epoch
            else:
                # Try to extract epoch from checkpoint path (e.g., epoch_190)
                match = re.search(r'epoch_(\d+)', resume_path)
                if match:
                    start_epoch = int(match.group(1))
                else:
                    # Use epoch from state if available
                    start_epoch = int(state.epoch) if hasattr(state, 'epoch') else 0

            # Calculate global_step based on resumed epoch
            global_step = start_epoch * steps_per_epoch

            print(f"Resumed from epoch {start_epoch}, global_step {global_step}")
            print(f"Will continue training for epochs {start_epoch + 1} to {args.epochs}\n")
        else:
            print(f"WARNING: Checkpoint not found at {resume_path}, starting from scratch")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        state = state.replace(epoch=epoch)

        # Create fresh data loaders each epoch
        if args.dataset == 'pathfinder':
            if args.tfrecord_dir:
                # Use TFRecord loader for max I/O performance
                train_loader = get_pathfinder_tfrecord_loader(
                    tfrecord_dir=args.tfrecord_dir,
                    difficulty=args.pathfinder_difficulty,
                    batch_size=args.batch_size,
                    num_frames=args.seq_len,
                    split='train',
                    shuffle=True,
                    prefetch_batches=args.prefetch_batches,
                )
            else:
                train_loader = get_pathfinder_loader(
                    root=args.data_dir,
                    difficulty=args.pathfinder_difficulty,
                    batch_size=args.batch_size,
                    num_frames=args.seq_len,
                    split='train',
                    num_workers=args.num_workers,
                    prefetch_batches=args.prefetch_batches,
                )
        elif args.dataset == 'imagenet':
            train_loader = get_imagenet_loader(
                data_dir=args.imagenet_dir,
                split='train',
                batch_size=args.batch_size,
                image_size=args.image_size,
                sequence_length=args.seq_len,
            )
        else:
            train_loader = get_imagenette_video_loader(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                sequence_length=args.seq_len,
                split='train',
            )

        # Training
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        epoch_step_times = []

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True) as pbar:
            for batch in pbar:
                rng, step_rng = jax.random.split(rng)

                # Time the training step
                step_start = time.perf_counter()
                state, metrics = train_step(state, batch, step_rng, num_classes)
                # Block until computation completes for accurate timing
                jax.block_until_ready(metrics)
                step_time = time.perf_counter() - step_start
                epoch_step_times.append(step_time)

                epoch_loss += float(metrics['train_loss'])
                epoch_acc += float(metrics['train_acc'])
                num_batches += 1
                global_step += 1

                # Update progress bar with timing
                pbar.set_postfix({
                    'loss': f"{metrics['train_loss']:.4f}",
                    'acc': f"{metrics['train_acc']:.4f}",
                    'ms/step': f"{step_time*1000:.1f}",
                })

                # Log to wandb
                if not args.no_wandb and global_step % args.log_every == 0:
                    # Get current learning rate from optimizer state
                    lr = args.lr  # Simplified - actual LR from schedule
                    wandb.log({
                        'train/loss': float(metrics['train_loss']),
                        'train/acc': float(metrics['train_acc']),
                        'train/learning_rate': lr,
                        'train/step': global_step,
                        'timing/step_ms': step_time * 1000,
                        'timing/throughput_samples_per_sec': args.batch_size / step_time,
                    }, step=global_step)

        # Epoch summary
        avg_train_loss = epoch_loss / max(num_batches, 1)
        avg_train_acc = epoch_acc / max(num_batches, 1)

        # Timing statistics
        avg_step_time_ms = sum(epoch_step_times) / len(epoch_step_times) * 1000 if epoch_step_times else 0
        min_step_time_ms = min(epoch_step_times) * 1000 if epoch_step_times else 0
        max_step_time_ms = max(epoch_step_times) * 1000 if epoch_step_times else 0
        throughput = args.batch_size / (sum(epoch_step_times) / len(epoch_step_times)) if epoch_step_times else 0

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        print(f"  Timing: avg={avg_step_time_ms:.1f}ms/step, min={min_step_time_ms:.1f}ms, max={max_step_time_ms:.1f}ms, throughput={throughput:.1f} samples/sec")

        # Validation
        if (epoch + 1) % args.eval_every == 0:
            if args.dataset == 'pathfinder':
                if args.tfrecord_dir:
                    val_loader = get_pathfinder_tfrecord_loader(
                        tfrecord_dir=args.tfrecord_dir,
                        difficulty=args.pathfinder_difficulty,
                        batch_size=args.batch_size,
                        num_frames=args.seq_len,
                        split='val',
                        shuffle=False,
                        prefetch_batches=args.prefetch_batches,
                    )
                else:
                    val_loader = get_pathfinder_loader(
                        root=args.data_dir,
                        difficulty=args.pathfinder_difficulty,
                        batch_size=args.batch_size,
                        num_frames=args.seq_len,
                        split='val',
                        shuffle=False,
                        num_workers=args.num_workers,
                        prefetch_batches=args.prefetch_batches,
                    )
            elif args.dataset == 'imagenet':
                val_loader = get_imagenet_loader(
                    data_dir=args.imagenet_dir,
                    split='val',
                    batch_size=args.batch_size,
                    image_size=args.image_size,
                    sequence_length=args.seq_len,
                )
            else:
                val_loader = get_imagenette_video_loader(
                    data_dir=args.data_dir,
                    batch_size=args.batch_size,
                    sequence_length=args.seq_len,
                    split='val',
                )
            val_metrics = evaluate(state, val_loader, num_classes)

            print(f"Epoch {epoch+1} - Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val Acc: {val_metrics['val_acc']:.4f}")

            if not args.no_wandb:
                wandb.log({
                    'val/loss': val_metrics['val_loss'],
                    'val/acc': val_metrics['val_acc'],
                    'epoch': epoch + 1,
                    'timing/epoch_avg_step_ms': avg_step_time_ms,
                    'timing/epoch_min_step_ms': min_step_time_ms,
                    'timing/epoch_max_step_ms': max_step_time_ms,
                    'timing/epoch_throughput': throughput,
                }, step=global_step)

            # Track best model
            if val_metrics['val_acc'] > best_val_acc:
                best_val_acc = val_metrics['val_acc']
                print(f"  New best validation accuracy: {best_val_acc:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}')
            checkpointer.save(ckpt_path, state)
            print(f"  Saved checkpoint to {ckpt_path}")

    # Wait for any pending checkpoint operations to complete
    checkpointer.wait_until_finished()

    # Final summary
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"  Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"{'='*60}\n")

    if not args.no_wandb:
        wandb.log({'best_val_acc': best_val_acc})
        wandb.finish()


if __name__ == '__main__':
    main()
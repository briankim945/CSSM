import argparse, random, os
from typing import Tuple, Dict
from functools import partial

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import optax
import numpy as np
from PIL import Image

from src.models.convnext import ModelFactory
# from src.models.cssm_vit import CSSMViT, cssm_vit_tiny, cssm_vit_small
from src.models.cssm_vit_old import CSSMViT
from src.models.baseline_vit import BaselineViT, baseline_vit_tiny, baseline_vit_small
from src.data import IMAGENETTE_CLASSES, load_image_val
from xai.utils import get_init_state, TrainState


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

    # variables = {'params': state.params, 'batch_stats': state.batch_stats}
    variables = {'params': state.params}

    logits, activations = state.apply_fn(
        variables,
        videos,
        training=False,
        return_activations=True,
    )

    one_hot = jax.nn.one_hot(labels, num_classes)
    loss = optax.softmax_cross_entropy(logits, one_hot).mean()
    acc = jnp.mean(jnp.argmax(logits, -1) == labels)

    return {
        'val_loss': loss,
        'val_acc': acc,
    }, activations


def main():
    parser = argparse.ArgumentParser(description='Train CSSM models')

    # Architecture selection
    parser.add_argument('--arch', type=str, choices=['convnext', 'vit', 'baseline'], default='convnext',
                        help='Architecture: convnext, vit (CSSM-ViT), or baseline (standard ViT)')
    
    # Model configuration (ConvNeXt-style)
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
            # gate_activation=args.gate_activation,
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

    # Setup checkpointing (must be absolute path for orbax)
    checkpointer = ocp.StandardCheckpointer()

    restored_state = checkpointer.restore(
        # '/users/bkim53/code_directories/CSSM/checkpoints/vit_standard_d12_e384_v1/epoch_45',
        '/Users/briankim/Documents/Research/OneVision/CepstralSSM.nosync/CSSM/checkpoints/vit_standard_d12_e384/epoch_100',
        target=state
        # abstract_my_tree
    )

    ### Getting data from a single imge

    c = IMAGENETTE_CLASSES[0]
    class_to_idx = {c: i for i, c in enumerate(IMAGENETTE_CLASSES)}
    random_file_name = random.choice(os.listdir(f"data/{c}"))
    target_path = f'data/{c}/{random_file_name}'
    print(random_file_name)

    train_features = np.repeat(
        load_image_val(target_path)[np.newaxis, np.newaxis, :, :, :],
        args.seq_len,
        axis=1
    )

    _, intermediates = eval_step(
        restored_state, (train_features, jnp.array([class_to_idx[c]])),
        len(IMAGENETTE_CLASSES)
    )

    print(jax.tree_util.tree_map(lambda x: x.shape, intermediates))
    print(jax.tree_util.tree_map(lambda x: x.shape, restored_state.params))


if __name__ == '__main__':
    main()

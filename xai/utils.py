import numpy as np
from scipy.ndimage import zoom
from PIL import Image
from random import randint, choice
from typing import Tuple

import jax
import jax.numpy as jnp
import optax
import wandb
from flax import linen as nn
from flax.training import train_state

from src.data import IMAGENET_MEAN, IMAGENET_STD


class TrainState(train_state.TrainState):
    """Extended train state with additional tracking."""
    # batch_stats: dict
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
        # batch_stats=variables['batch_stats'] if 'batch_stats' in variables else {},
        tx=tx,
        epoch=0,
    )

    return state, tx, variables


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


def prep_image(image, flatten_channels=False):
    if flatten_channels:
        image = np.mean(np.abs(image), axis=-1)
    # Normalize the gradient values to be between 0-1
    max_val= np.max(image)
    min_val = np.min(image)
    image = (image - min_val) / (max_val - min_val)
    # Convert the grads to uint8 for displaying
    image = np.uint8(image * 255)
    return image  

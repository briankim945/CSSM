"""
Multi-GPU training utilities for JAX.

Uses pmap for data parallelism with proper gradient synchronization.
Designed for H200 multi-GPU training.
"""

import jax
import jax.numpy as jnp
from flax import jax_utils
from flax.training import train_state
from typing import Callable, Any, Tuple
import optax


def replicate_state(state: train_state.TrainState) -> train_state.TrainState:
    """
    Replicate train state across all devices.

    Args:
        state: Single-device train state

    Returns:
        Replicated train state (one copy per device)
    """
    return jax_utils.replicate(state)


def unreplicate_state(state: train_state.TrainState) -> train_state.TrainState:
    """
    Get single-device state from replicated state.

    Args:
        state: Replicated train state

    Returns:
        Single-device train state (from first device)
    """
    return jax_utils.unreplicate(state)


def shard_batch(batch: Tuple, num_devices: int = None) -> Tuple:
    """
    Shard batch across devices.

    Reshapes (B, ...) to (num_devices, B // num_devices, ...).
    Batch size must be divisible by num_devices.

    Args:
        batch: Tuple of arrays (images, labels)
        num_devices: Number of devices (defaults to jax.device_count())

    Returns:
        Sharded batch with leading device dimension
    """
    if num_devices is None:
        num_devices = jax.device_count()

    def _shard(x):
        # Reshape first dimension to (devices, batch_per_device, ...)
        batch_size = x.shape[0]
        assert batch_size % num_devices == 0, \
            f"Batch size {batch_size} not divisible by num_devices {num_devices}"
        return x.reshape((num_devices, batch_size // num_devices) + x.shape[1:])

    return jax.tree_util.tree_map(_shard, batch)


def create_parallel_train_step(
    train_step_fn: Callable,
    axis_name: str = 'batch'
) -> Callable:
    """
    Create parallelized training step with gradient synchronization.

    The train_step_fn should internally use jax.lax.pmean for gradient sync.

    Args:
        train_step_fn: Training step function with signature
            (state, batch, rng, axis_name) -> (state, metrics)
        axis_name: Axis name for pmean gradient sync

    Returns:
        pmap-ed training step function
    """
    def parallel_train_step(state, batch, rng):
        return train_step_fn(state, batch, rng, axis_name=axis_name)

    return jax.pmap(
        parallel_train_step,
        axis_name=axis_name,
        donate_argnums=(0,)  # Donate state buffer for efficiency
    )


def create_parallel_eval_step(
    eval_step_fn: Callable,
    axis_name: str = 'batch'
) -> Callable:
    """
    Create parallelized evaluation step.

    Args:
        eval_step_fn: Evaluation step function
        axis_name: Axis name for pmean metric sync

    Returns:
        pmap-ed evaluation step function
    """
    return jax.pmap(eval_step_fn, axis_name=axis_name)


def split_rng_for_devices(rng: jax.Array) -> jax.Array:
    """
    Split RNG key for each device.

    Args:
        rng: Single RNG key

    Returns:
        Array of RNG keys, one per device
    """
    return jax.random.split(rng, jax.device_count())


def get_global_batch_size(local_batch_size: int) -> int:
    """
    Get effective global batch size across all devices.

    Args:
        local_batch_size: Batch size per device

    Returns:
        Total batch size across all devices
    """
    return local_batch_size * jax.device_count()


def make_train_step(model, num_classes: int):
    """
    Create a training step function with proper gradient synchronization.
    Supports models with BatchNorm (batch_stats collection).

    Args:
        model: Flax model
        num_classes: Number of output classes

    Returns:
        Training step function compatible with pmap
    """
    def train_step(state, batch, rng, batch_stats=None, axis_name='batch'):
        videos, labels = batch

        def loss_fn(params):
            variables = {'params': params}
            if batch_stats is not None:
                variables['batch_stats'] = batch_stats

            output = state.apply_fn(
                variables,
                videos,
                training=True,
                rngs={'dropout': rng},
                mutable=['batch_stats'] if batch_stats is not None else False,
            )

            if batch_stats is not None:
                logits, new_state = output
                new_batch_stats = new_state.get('batch_stats', None)
            else:
                logits = output
                new_batch_stats = None

            one_hot = jax.nn.one_hot(labels, num_classes)
            loss = optax.softmax_cross_entropy(logits, one_hot).mean()
            return loss, (logits, new_batch_stats)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (logits, new_batch_stats)), grads = grad_fn(state.params)

        # Synchronize gradients across devices
        grads = jax.lax.pmean(grads, axis_name=axis_name)
        loss = jax.lax.pmean(loss, axis_name=axis_name)

        # Sync batch_stats across devices
        if new_batch_stats is not None:
            new_batch_stats = jax.lax.pmean(new_batch_stats, axis_name=axis_name)

        # Update state
        state = state.apply_gradients(grads=grads)

        # Compute accuracy
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(preds == labels)
        acc = jax.lax.pmean(acc, axis_name=axis_name)

        return state, {'loss': loss, 'acc': acc}, new_batch_stats

    return train_step


def make_eval_step(model, num_classes: int):
    """
    Create an evaluation step function.
    Supports models with BatchNorm (batch_stats collection).

    Args:
        model: Flax model
        num_classes: Number of output classes

    Returns:
        Evaluation step function compatible with pmap
    """
    def eval_step(state, batch, batch_stats=None, axis_name='batch'):
        videos, labels = batch

        variables = {'params': state.params}
        if batch_stats is not None:
            variables['batch_stats'] = batch_stats

        logits = state.apply_fn(
            variables,
            videos,
            training=False,
        )

        # Loss
        one_hot = jax.nn.one_hot(labels, num_classes)
        loss = optax.softmax_cross_entropy(logits, one_hot).mean()
        loss = jax.lax.pmean(loss, axis_name=axis_name)

        # Accuracy
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(preds == labels)
        acc = jax.lax.pmean(acc, axis_name=axis_name)

        return {'loss': loss, 'acc': acc}

    return eval_step


def make_train_step_mixup(model, num_classes: int, label_smoothing: float = 0.0):
    """
    Create a training step function that supports soft labels (mixup/cutmix).

    This version expects labels to already be one-hot or soft labels (B, num_classes)
    instead of class indices (B,).

    Args:
        model: Flax model
        num_classes: Number of output classes
        label_smoothing: Label smoothing factor (applied to soft labels)

    Returns:
        Training step function compatible with pmap
    """
    def train_step(state, batch, rng, batch_stats=None, axis_name='batch'):
        videos, soft_labels = batch  # soft_labels is (B, num_classes)

        def loss_fn(params):
            variables = {'params': params}
            if batch_stats is not None:
                variables['batch_stats'] = batch_stats

            output = state.apply_fn(
                variables,
                videos,
                training=True,
                rngs={'dropout': rng},
                mutable=['batch_stats'] if batch_stats is not None else False,
            )

            if batch_stats is not None:
                logits, new_state = output
                new_batch_stats = new_state.get('batch_stats', None)
            else:
                logits = output
                new_batch_stats = None

            # Apply label smoothing to soft labels
            if label_smoothing > 0:
                soft_labels_smooth = soft_labels * (1.0 - label_smoothing) + label_smoothing / num_classes
            else:
                soft_labels_smooth = soft_labels

            # Cross-entropy with soft labels
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            loss = -jnp.sum(soft_labels_smooth * log_probs, axis=-1).mean()
            return loss, (logits, new_batch_stats)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (logits, new_batch_stats)), grads = grad_fn(state.params)

        # Synchronize gradients across devices
        grads = jax.lax.pmean(grads, axis_name=axis_name)
        loss = jax.lax.pmean(loss, axis_name=axis_name)

        # Sync batch_stats across devices
        if new_batch_stats is not None:
            new_batch_stats = jax.lax.pmean(new_batch_stats, axis_name=axis_name)

        # Update state
        state = state.apply_gradients(grads=grads)

        # Compute accuracy (use argmax of soft labels as ground truth for mixup)
        preds = jnp.argmax(logits, axis=-1)
        # For mixup, compute accuracy against the dominant class
        true_labels = jnp.argmax(soft_labels, axis=-1)
        acc = jnp.mean(preds == true_labels)
        acc = jax.lax.pmean(acc, axis_name=axis_name)

        return state, {'loss': loss, 'acc': acc}, new_batch_stats

    return train_step


def make_train_step_bce(model, num_classes: int):
    """
    Create a training step function using Binary Cross Entropy loss.

    DeiT III uses BCE instead of softmax cross-entropy, treating each
    class as an independent binary classification problem.

    Args:
        model: Flax model
        num_classes: Number of output classes

    Returns:
        Training step function compatible with pmap
    """
    def train_step(state, batch, rng, batch_stats=None, axis_name='batch'):
        videos, soft_labels = batch  # soft_labels is (B, num_classes)

        def loss_fn(params):
            variables = {'params': params}
            if batch_stats is not None:
                variables['batch_stats'] = batch_stats

            output = state.apply_fn(
                variables,
                videos,
                training=True,
                rngs={'dropout': rng},
                mutable=['batch_stats'] if batch_stats is not None else False,
            )

            if batch_stats is not None:
                logits, new_state = output
                new_batch_stats = new_state.get('batch_stats', None)
            else:
                logits = output
                new_batch_stats = None

            # Binary Cross Entropy with logits
            # BCE treats each class independently
            # loss = -y * log(sigmoid(x)) - (1-y) * log(1 - sigmoid(x))
            loss = optax.sigmoid_binary_cross_entropy(logits, soft_labels).mean()
            return loss, (logits, new_batch_stats)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (logits, new_batch_stats)), grads = grad_fn(state.params)

        # Synchronize gradients across devices
        grads = jax.lax.pmean(grads, axis_name=axis_name)
        loss = jax.lax.pmean(loss, axis_name=axis_name)

        # Sync batch_stats across devices
        if new_batch_stats is not None:
            new_batch_stats = jax.lax.pmean(new_batch_stats, axis_name=axis_name)

        # Update state
        state = state.apply_gradients(grads=grads)

        # Compute accuracy (use argmax of soft labels as ground truth)
        preds = jnp.argmax(logits, axis=-1)
        true_labels = jnp.argmax(soft_labels, axis=-1)
        acc = jnp.mean(preds == true_labels)
        acc = jax.lax.pmean(acc, axis_name=axis_name)

        return state, {'loss': loss, 'acc': acc}, new_batch_stats

    return train_step


def make_eval_step_with_params(model, num_classes: int):
    """
    Create an evaluation step function that takes explicit params.
    Useful for EMA evaluation where params differ from state.params.

    Args:
        model: Flax model
        num_classes: Number of output classes

    Returns:
        Evaluation step function compatible with pmap
    """
    def eval_step(params, batch, batch_stats=None, axis_name='batch', apply_fn=None):
        videos, labels = batch

        variables = {'params': params}
        if batch_stats is not None:
            variables['batch_stats'] = batch_stats

        logits = apply_fn(
            variables,
            videos,
            training=False,
        )

        # Loss
        one_hot = jax.nn.one_hot(labels, num_classes)
        loss = optax.softmax_cross_entropy(logits, one_hot).mean()
        loss = jax.lax.pmean(loss, axis_name=axis_name)

        # Accuracy
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(preds == labels)
        acc = jax.lax.pmean(acc, axis_name=axis_name)

        return {'loss': loss, 'acc': acc}

    return eval_step

import numpy as np
import jax
import jax.numpy as jnp


def integrated_gradients(state, inputs, target_class, baseline=None, n_steps=50):
    """
    Compute Integrated Gradients attribution.
    
    Args:
        model_fn: Function that takes inputs and returns logits
        inputs: Input array (e.g., image)
        baseline: Baseline input (e.g., black image)
        target_class: Class index to attribute
        n_steps: Number of interpolation steps
    """
    if baseline is None:
        baseline = 0*inputs
    assert(baseline.shape == inputs.shape)

    # Create interpolated inputs along the path
    alphas = jnp.linspace(0, 1, n_steps)
    interpolated = baseline + alphas[:, None, None, None, None] * (inputs - baseline)
    print("interpolated:", interpolated.shape)
    
    # Define function to get gradient w.r.t. input for target class
    def get_grad(x):
        def forward(x):
            # logits = model_fn(x)
            logits = state.apply_fn(
                {'params': state.params}, 
                x[None, :, :, :, :],
                training=False,
            )
            return logits[0, target_class]
        return jax.grad(forward)(x)
    
    # Compute gradients at all interpolation points (vectorized)
    grads = jax.vmap(get_grad)(interpolated)
    
    # Average gradients and scale by input difference
    avg_grads = jnp.mean(grads, axis=0)
    integrated_grads = (inputs - baseline) * avg_grads
    
    return integrated_grads


def smooth_grad(model_fn, inputs, target_class, noise_level=0.1, n_samples=50, key=None):
    """SmoothGrad: average gradients over noisy inputs."""
    if key is None:
        key = jax.random.PRNGKey(0)
    
    noise = jax.random.normal(key, (n_samples,) + inputs.shape) * noise_level
    noisy_inputs = inputs + noise
    
    def get_grad(x):
        def forward(x):
            return model_fn(x)[target_class]
        return jax.grad(forward)(x)
    
    grads = jax.vmap(get_grad)(noisy_inputs)
    return jnp.mean(grads, axis=0)

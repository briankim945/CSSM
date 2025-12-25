import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from typing import Tuple



class SAE(nn.Module):
    """Minimal SAE for computer vision activations."""
    d_in: int
    d_sae: int
    
    @nn.compact
    def __call__(self, x):
        # Decoder bias (learned)
        b_dec = self.param('b_dec', nn.initializers.zeros, (self.d_in,))
        
        # Encoder
        x_centered = x - b_dec
        W_enc = self.param('W_enc', nn.initializers.xavier_uniform(), 
                          (self.d_in, self.d_sae))
        b_enc = self.param('b_enc', nn.initializers.zeros, (self.d_sae,))
        features = nn.relu(jnp.dot(x_centered, W_enc) + b_enc)
        
        # Decoder (tied weights)
        reconstruction = jnp.dot(features, W_enc.T) + b_dec
        
        return features, reconstruction


def train_sae_on_activations(
    cnn_activations: jnp.ndarray,  # [N, d_in]
    d_sae: int = 8192,
    sparsity_coef: float = 0.01,
    num_epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-3
):
    """Train SAE on CNN activations."""
    
    d_in = cnn_activations.shape[1]
    
    # Initialize model
    rng = jax.random.PRNGKey(0)
    model = SAE(d_in=d_in, d_sae=d_sae)
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
    def train_step(state, batch):
        def loss_fn(params):
            features, recon = model.apply({'params': params}, batch)
            recon_loss = jnp.mean((batch - recon) ** 2)
            sparsity_loss = jnp.mean(jnp.abs(features))
            total_loss = recon_loss + sparsity_coef * sparsity_loss
            return total_loss, {
                'recon': recon_loss,
                'sparsity': sparsity_loss,
                'active': jnp.mean(features > 0)
            }
        
        (loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)
        
        state = state.apply_gradients(grads=grads)
        return state, loss, metrics
    
    # Train
    num_samples = cnn_activations.shape[0]
    for epoch in range(num_epochs):
        # Shuffle data
        perm = jax.random.permutation(rng, num_samples)
        activations_shuffled = cnn_activations[perm]
        
        # Batch training
        for i in range(0, num_samples, batch_size):
            batch = activations_shuffled[i:i+batch_size]
            state, loss, metrics = train_step(state, batch)
        
        print(f"Epoch {epoch}: Loss={loss:.4f}, "
              f"Active={metrics['active']:.2%}")
    
    return state, model

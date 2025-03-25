import jax
import jax.numpy as jnp
import optax
from flax import nnx

from vae.modules import Decoder, Encoder, GaussianPosterior


class VariationalAutoEncoder:
    def __init__(self, encoder: Encoder, decoder: Decoder):
        self.encoder = encoder
        self.decoder = decoder

    def __call__(self, x: jax.Array, rngs: nnx.Rngs) -> tuple[GaussianPosterior, jax.Array]:
        posterior = self.encode(x)
        latent = posterior.sample(rngs=rngs)
        x_hat = self.decode(latent)
        return posterior, x_hat

    def encode(self, x: jax.Array) -> GaussianPosterior:
        moments = self.encoder(x)
        posterior = GaussianPosterior(moments)
        return posterior

    def decode(self, latent: jax.Array) -> jax.Array:
        x_hat = self.decoder(latent)
        return x_hat


def loss_fn(
    model: VariationalAutoEncoder, x: jax.Array, rngs: nnx.Rngs, beta: float
) -> tuple[jax.Array, ...]:
    posterior, predictions = model(x, rngs)
    kl = -0.5 * jnp.sum(1 + posterior.logvar - (posterior.mean**2) - posterior.var, axis=(1, 2, 3))
    kl_loss = jnp.mean(kl)
    recon_loss = jnp.mean(optax.squared_error(predictions=predictions, targets=x))
    loss = beta * kl_loss + recon_loss
    return loss, kl_loss, recon_loss, predictions

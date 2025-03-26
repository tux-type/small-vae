import jax
from flax import nnx

from vae.modules import Decoder, Encoder, GaussianPosterior


class VariationalAutoEncoder(nnx.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder

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

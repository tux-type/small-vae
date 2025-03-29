import jax
from flax import nnx

from vae.modules import Decoder, Encoder, GaussianPosterior


class VariationalAutoEncoder(nnx.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device: jax.Device):
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder
        self.device: jax.Device = device

    def __call__(self, x: jax.Array, rngs: nnx.Rngs) -> tuple[GaussianPosterior, jax.Array]:
        posterior = self.encode(x)
        latent = posterior.sample(rngs=rngs, device=self.device)
        x_hat = self.decode(latent)
        return posterior, x_hat

    def encode(self, x: jax.Array) -> GaussianPosterior:
        moments = self.encoder(x)
        posterior = GaussianPosterior(moments)
        return posterior

    def decode(self, latent: jax.Array) -> jax.Array:
        x_hat = self.decoder(latent)
        return x_hat

    def reconstruct(self, x: jax.Array) -> jax.Array:
        posterior = self.encode(x)
        latent = posterior.mean
        x_hat = self.decode(latent)
        return x_hat

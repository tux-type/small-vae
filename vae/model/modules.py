import jax
import jax.numpy as jnp
from flax import nnx

from vae.model.blocks import ConvBlock, DownBlock, UpBlock


class Encoder(nnx.Module):
    def __init__(
        self,
        in_features: int,
        num_features: int,
        latent_features: int,
        rngs: nnx.Rngs,
        resolution: int,
        feature_multipliers: tuple[int, ...] = (1, 2, 4),
        double_latent_features: bool = True,
    ):
        self.in_features = in_features
        self.num_features = num_features
        self.latent_features = latent_features

        self.resolution = resolution
        self.num_resolutions = len(feature_multipliers) + 1

        self.image_projection = ConvBlock(in_features, num_features, is_residual=True, rngs=rngs)

        self.down_blocks = []
        in_features_curr = num_features
        for mult in feature_multipliers:
            out_features_curr = mult * num_features
            self.down_blocks.append(DownBlock(in_features_curr, out_features_curr, rngs=rngs))
            in_features_curr = out_features_curr

        self.mid_block = ConvBlock(in_features_curr, in_features_curr, rngs=rngs)

        self.feature_aggregation = nnx.Conv(
            in_features_curr,
            2 * latent_features if double_latent_features else latent_features,
            kernel_size=(3, 3),
            strides=1,
            padding=1,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        h = self.image_projection(x)

        for down_block in self.down_blocks:
            h = down_block(h)

        h = self.mid_block(h)

        out = self.feature_aggregation(h)
        return out


class Decoder(nnx.Module):
    def __init__(
        self,
        latent_features: int,
        num_features: int,
        out_features: int,
        rngs: nnx.Rngs,
        resolution: int,
        feature_multipliers: tuple[int, ...] = (1, 2, 4),
    ):
        self.latent_features = latent_features
        self.num_features = num_features
        self.out_features = out_features

        self.resolution = resolution
        self.num_resolutions = len(feature_multipliers) + 1
        latent_resolution = resolution // (2 ** (self.num_resolutions - 1))
        self.latent_shape = (1, latent_resolution, latent_resolution, latent_features)

        in_features_curr = num_features * feature_multipliers[-1]

        self.latent_projection = ConvBlock(
            latent_features,
            in_features_curr,
            is_residual=True,
            rngs=rngs,
        )

        self.mid_block = ConvBlock(in_features_curr, in_features_curr, rngs=rngs)

        self.up_blocks = []

        for mult in reversed(feature_multipliers):
            out_features_curr = mult * num_features
            self.up_blocks.append(UpBlock(in_features_curr, out_features_curr, rngs=rngs))
            in_features_curr = out_features_curr

        self.feature_aggregation = nnx.Conv(
            in_features_curr,
            out_features,
            kernel_size=(3, 3),
            strides=1,
            padding=1,
            rngs=rngs,
        )

    def __call__(self, z: jax.Array) -> jax.Array:
        h = self.latent_projection(z)

        h = self.mid_block(h)

        for up_block in self.up_blocks:
            h = up_block(h)

        out = self.feature_aggregation(h)
        return out


class GaussianPosterior:
    def __init__(self, moments: jax.Array):
        self.moments = moments
        self.mean, self.logvar = jnp.split(moments, 2, -1)
        # Avoid underflow and overflow when exponentiating
        self.logvar = jnp.clip(self.logvar, -30.0, 20.0)
        self.var = jnp.exp(self.logvar)
        self.std = jnp.exp(0.5 * self.logvar)

    def sample(self, rngs: nnx.Rngs, device: jax.Device) -> jax.Array:
        latent = self.mean + self.std * jax.random.normal(
            rngs.latent(), shape=self.mean.shape
        ).to_device(device)
        return latent

import jax
import jax.numpy as jnp
from flax import nnx


class ConvBlock(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rngs: nnx.Rngs,
        is_residual: bool = False,
    ):
        self.is_residual = is_residual
        self.conv1 = nnx.Sequential(
            nnx.Conv(
                in_features=in_features,
                out_features=out_features,
                kernel_size=(3, 3),
                strides=1,
                padding=1,
                rngs=rngs,
            ),
            nnx.GroupNorm(num_features=out_features, num_groups=8, rngs=rngs),
            nnx.relu,
        )
        self.conv2 = nnx.Sequential(
            nnx.Conv(
                in_features=out_features,
                out_features=out_features,
                kernel_size=(3, 3),
                strides=1,
                padding=1,
                rngs=rngs,
            ),
            nnx.GroupNorm(num_features=out_features, num_groups=8, rngs=rngs),
            nnx.relu,
            nnx.Conv(
                in_features=out_features,
                out_features=out_features,
                kernel_size=(3, 3),
                strides=1,
                padding=1,
                rngs=rngs,
            ),
            nnx.GroupNorm(num_features=out_features, num_groups=8, rngs=rngs),
            nnx.relu,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = self.conv1(x)
        out = self.conv2(residual)
        if self.is_residual:
            return out / 1.414
        else:
            return out


class DownBlock(nnx.Module):
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
        self.conv = nnx.Sequential(
            ConvBlock(in_features, out_features, rngs, is_residual=False),
            ConvBlock(out_features, out_features, rngs, is_residual=False),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        out = self.conv(x)
        out = nnx.max_pool(out, window_shape=(2, 2), strides=(2, 2))
        return out


class UpBlock(nnx.Module):
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
        self.conv = nnx.Sequential(
            nnx.ConvTranspose(
                in_features=in_features,
                out_features=out_features,
                kernel_size=(2, 2),
                strides=(2, 2),
                rngs=rngs,
            ),
            ConvBlock(out_features, out_features, rngs=rngs, is_residual=False),
            ConvBlock(out_features, out_features, rngs=rngs, is_residual=False),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        out = self.conv(x)
        return out


class Encoder(nnx.Module):
    def __init__(
        self,
        in_features: int,
        num_features: int,
        latent_features: int,
        rngs: nnx.Rngs,
        feature_multipliers: tuple[int, ...] = (1, 2, 4),
        double_latent_features: bool = True,
    ):
        self.in_features = in_features
        self.num_features = num_features
        self.latent_features = latent_features

        self.num_resolutions = len(feature_multipliers) + 1

        self.image_projection = ConvBlock(
            in_features, num_features, is_residual=True, rngs=rngs
        )

        self.down_blocks = []
        in_features_curr = num_features
        for mult in feature_multipliers:
            out_features_curr = mult * num_features
            self.down_blocks.append(
                DownBlock(in_features_curr, out_features_curr, rngs=rngs)
            )
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
    def __init__(self):
        pass

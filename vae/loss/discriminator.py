from functools import partial

import jax
from flax import nnx


class DiscriminatorBlock(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rngs: nnx.Rngs,
        use_norm: bool,
        kernel_size: tuple[int, int] = (4, 4),
        strides: int = 2,
        padding: int = 1,
    ):
        self.use_norm = use_norm
        layers = []
        layers.append(
            nnx.Conv(
                in_features=in_features,
                out_features=out_features,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                rngs=rngs,
            )
        )
        if use_norm:
            layers.append(nnx.GroupNorm(num_features=out_features, num_groups=8, rngs=rngs))
        layers.append(partial(nnx.leaky_relu, negative_slope=0.2))

        self.conv = nnx.Sequential(*layers)

    def __call__(self, x: jax.Array) -> jax.Array:
        out = self.conv(x)
        return out


class Discriminator(nnx.Module):
    def __init__(
        self,
        in_features: int,
        num_features: int,
        rngs: nnx.Rngs,
        feature_multipliers: tuple[int, ...] = (2, 4, 8),
    ):
        self.in_features = in_features
        self.num_features = num_features
        self.feature_multipliers = feature_multipliers

        self.image_projection = DiscriminatorBlock(
            in_features=in_features, out_features=num_features, strides=2, rngs=rngs, use_norm=False
        )

        self.down_blocks = []
        in_features_curr = num_features
        for i, mult in enumerate(feature_multipliers):
            out_features_curr = mult * num_features
            strides = 2 if i < len(feature_multipliers) - 1 else 1
            self.down_blocks.append(
                DiscriminatorBlock(
                    in_features=in_features_curr,
                    out_features=out_features_curr,
                    strides=strides,
                    rngs=rngs,
                    use_norm=True,
                )
            )
            in_features_curr = out_features_curr

        self.feature_aggregation = nnx.Conv(
            in_features_curr, 1, kernel_size=(4, 4), strides=1, padding=1, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        h = self.image_projection(x)
        for down_block in self.down_blocks:
            h = down_block(h)

        out = self.feature_aggregation(h)
        return out

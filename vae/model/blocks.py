import jax
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
            ConvBlock(in_features, out_features, rngs=rngs, is_residual=False),
            ConvBlock(out_features, out_features, rngs=rngs, is_residual=False),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        B, H, W, C = x.shape
        out = jax.image.resize(x, shape=(B, H * 2, W * 2, C), method="nearest")
        out = self.conv(out)
        return out

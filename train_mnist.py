import logging

import ray
import ray.data

from vae.vae import VariationalAutoEncoder, loss_fn

ray.data.DataContext.get_current().enable_progress_bars = False
logging.getLogger("ray.data").setLevel(logging.WARNING)

ray.init(num_cpus=12, num_gpus=1)

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx


def normalise_batch(samples: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    data = np.array(samples["image"]).astype(np.float32)
    # Pad from 28x28 to 32x32 for easier downsampling and upsampling
    data = np.pad(data, ((0, 0), (2, 2), (2, 2)), mode="constant", constant_values=-1)
    normalised_data = ((data / 255.0) - 0.5) / 0.5
    samples["image"] = normalised_data
    return samples


@nnx.jit
def train_step(
    model: VariationalAutoEncoder,
    optimiser: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    rngs: nnx.Rngs,
    x: jax.Array,
):
    grad_fn = nnx.value_and_grad(f=loss_fn, has_aux=True)
    (loss, kl_loss, recon_loss, predictions), grads = grad_fn(model, x, rngs)
    metrics.update(loss=loss, kl_loss=kl_loss, recon_loss=recon_loss)
    optimiser.update(grads)


def train_mnist():
    pass

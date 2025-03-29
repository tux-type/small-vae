# ruff: noqa: E402
import logging

import ray
import ray.data

ray.data.DataContext.get_current().enable_progress_bars = False
logging.getLogger("ray.data").setLevel(logging.WARNING)

ray.init(num_cpus=12, num_gpus=1)

import jax
import jax.numpy as jnp
import matplotlib.image
import numpy as np
import optax
from flax import nnx
from tqdm import tqdm

from vae.modules import Decoder, Encoder
from vae.vae import VariationalAutoEncoder


def normalise_batch(samples: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    data = np.array(samples["image"]).astype(np.float32)
    # Pad from 28x28 to 32x32 for easier downsampling and upsampling
    data = np.pad(data, ((0, 0), (2, 2), (2, 2)), mode="constant", constant_values=-1)
    normalised_data = ((data / 255.0) - 0.5) / 0.5
    samples["image"] = normalised_data
    return samples


def loss_fn(
    model: VariationalAutoEncoder, x: jax.Array, rngs: nnx.Rngs, beta: float
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    posterior, predictions = model(x, rngs)
    kl = -0.5 * jnp.sum(1 + posterior.logvar - (posterior.mean**2) - posterior.var, axis=(1, 2, 3))
    kl_loss = kl.sum() / x.shape[0]
    recon_loss = jnp.mean(optax.squared_error(predictions=predictions, targets=x))
    loss = beta * kl_loss + recon_loss
    return loss, (kl_loss, recon_loss, predictions)


@nnx.jit
def train_step(
    model: VariationalAutoEncoder,
    optimiser: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    rngs: nnx.Rngs,
    x: jax.Array,
):
    grad_fn = nnx.value_and_grad(f=loss_fn, has_aux=True)
    (loss, aux_data), grads = grad_fn(model, x, rngs, beta=0.01)
    kl_loss, recon_loss, predictions = aux_data
    metrics.update(loss=loss, kl_loss=kl_loss, recon_loss=recon_loss)
    optimiser.update(grads)


def train_mnist(num_epochs: int):
    init_rngs = nnx.Rngs(params=0)
    training_rngs = nnx.Rngs(latent=1)
    sampling_rngs = nnx.Rngs(latent=2)

    encoder = Encoder(
        in_features=1,
        num_features=128,
        latent_features=4,
        rngs=init_rngs,
        resolution=32,
        feature_multipliers=(1, 2, 4),
        double_latent_features=True,
    )
    decoder = Decoder(
        latent_features=4,
        num_features=128,
        out_features=1,
        rngs=init_rngs,
        resolution=32,
        feature_multipliers=(1, 2, 4),
    )
    vae = VariationalAutoEncoder(
        encoder=encoder,
        decoder=decoder,
        device=jax.devices("gpu")[0],
    )

    train_dataset, validation_dataset = (
        ray.data.read_images("data/mnist", override_num_blocks=12)
        .map_batches(normalise_batch)
        .random_shuffle(seed=42)
        .materialize()
        .train_test_split(test_size=0.1)
    )

    optimiser = nnx.Optimizer(vae, optax.adam(2e-4))
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        kl_loss=nnx.metrics.Average("kl_loss"),
        recon_loss=nnx.metrics.Average("recon_loss"),
    )

    metrics_history = {"loss": [], "kl_loss": [], "recon_loss": []}

    for epoch in range(num_epochs):
        train_dataset = train_dataset.random_shuffle().materialize()
        progress_bar = tqdm(
            train_dataset.iter_batches(
                prefetch_batches=1, batch_size=128, batch_format="numpy", drop_last=True
            )
        )
        for batch in progress_bar:
            x = jnp.array(batch["image"])[:, :, :, None]
            train_step(vae, optimiser, metrics, training_rngs, x)

            for metric, value in metrics.compute().items():
                metrics_history[metric].append(value)
            loss = metrics_history["loss"][-1]
            kl_loss = metrics_history["kl_loss"][-1]
            recon_loss = metrics_history["recon_loss"][-1]
            progress_bar.set_description(
                f"loss: {loss:.4f} kl_loss: {kl_loss:.4f} recon_loss: {recon_loss:.4f}"
            )

        if epoch % 5 == 0 or epoch == (num_epochs - 1):
            print(f"Sampling Epoch: {epoch}")
            x = jnp.array(validation_dataset.take_batch(8, batch_format="numpy")["image"])[
                :, :, :, None
            ]
            x_hat = vae.reconstruct(x)
            xs = jnp.concatenate(
                (jnp.concatenate(x, axis=1), jnp.concatenate(x_hat, axis=1)), axis=0
            )
            xs = ((xs + 1) * 127.5).clip(0, 255)
            # Squeeze since grayscale
            matplotlib.image.imsave(
                f"imgs/mnist/image_{epoch}.png",
                xs.squeeze(axis=2).astype(jnp.uint8),
                cmap="binary",
            )


if __name__ == "__main__":
    train_mnist(num_epochs=10)

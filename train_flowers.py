# ruff: noqa: E402
import logging

import ray
import ray.data

ray.data.DataContext.get_current().enable_progress_bars = False
logging.getLogger("ray.data").setLevel(logging.WARNING)

ray.init(num_cpus=12, num_gpus=1)

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.image
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from tqdm import tqdm

from vae.modules import Decoder, Encoder
from vae.vae import VariationalAutoEncoder


def crop_resize(img: np.ndarray, size: tuple[int, int] = (32, 32)) -> np.ndarray:
    # Expected img shape = (height, width, channels)
    height, width = img.shape[:2]
    crop_size = min(height, width)

    x_start = (width - crop_size) // 2
    y_start = (height - crop_size) // 2

    cropped_img = img[y_start : y_start + crop_size, x_start : x_start + crop_size, :]

    # Resize method = Nearest
    nearest_x_idx = (np.arange(size[1]) * crop_size / size[1]).astype(np.uint16)
    nearest_y_idx = (np.arange(size[0]) * crop_size / size[0]).astype(np.uint16)

    resized_img = cropped_img[nearest_y_idx.reshape(-1, 1), nearest_x_idx, :]

    return resized_img


def normalise_batch(samples: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    data = np.array([crop_resize(image) for image in samples["image"]]).astype(np.float32)
    normalised_data = ((data / 255.0) - 0.5) / 0.5
    samples["image"] = normalised_data
    return samples


def loss_fn(
    model: VariationalAutoEncoder, x: jax.Array, rngs: nnx.Rngs, beta: float
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    posterior, predictions = model(x, rngs)
    kl = -0.5 * jnp.sum(1 + posterior.logvar - (posterior.mean**2) - posterior.var, axis=(1, 2, 3))
    kl_loss = kl.sum() / x.shape[0]
    recon_loss = optax.squared_error(predictions=predictions, targets=x).mean()
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
    (loss, aux_data), grads = grad_fn(model, x, rngs, beta=0.0001)
    kl_loss, recon_loss, predictions = aux_data
    metrics.update(loss=loss, kl_loss=kl_loss, recon_loss=recon_loss)
    optimiser.update(grads)


def save_sample_images(path: str, original_images: jax.Array, reconstructed_images: jax.Array):
    original_images = jnp.concatenate(original_images, axis=1)
    reconstructed_images = jnp.concatenate(reconstructed_images, axis=1)
    concat_images = jnp.concatenate((original_images, reconstructed_images), axis=0)
    concat_images = ((concat_images + 1) * 127.5 + 0.5).clip(0, 255)
    concat_images = np.array(concat_images.astype(jnp.uint8))
    matplotlib.image.imsave(path, concat_images)


def train_mnist(num_epochs: int):
    init_rngs = nnx.Rngs(params=0)
    training_rngs = nnx.Rngs(latent=1)
    sampling_rngs = nnx.Rngs(latent=2)

    encoder = Encoder(
        in_features=3,
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
        out_features=3,
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
        ray.data.read_images("data/oxford_flowers", override_num_blocks=12)
        .map_batches(normalise_batch)
        .random_shuffle(seed=42)
        .train_test_split(test_size=0.1)
    )

    optimiser = nnx.Optimizer(vae, optax.adam(2e-5))
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
            x = jnp.array(batch["image"])
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
            x = jnp.array(train_dataset.take_batch(8, batch_format="numpy")["image"])
            x_hat = vae.reconstruct(x)
            save_sample_images(
                f"imgs/flowers/training_image_{epoch}.png",
                original_images=x,
                reconstructed_images=x_hat,
            )

            x = jnp.array(validation_dataset.take_batch(128, batch_format="numpy")["image"])
            x_hat = vae.reconstruct(x)
            recon_loss = optax.squared_error(predictions=x_hat, targets=x).mean()
            print(f"validation recon_loss: {recon_loss:.4f}")
            save_sample_images(
                f"imgs/flowers/validation_image_{epoch}.png",
                original_images=x[:8],
                reconstructed_images=x_hat[:8],
            )

            ckpt_dir = ocp.test_utils.erase_and_create_empty(Path.cwd() / "models" / "flowers")
            (_, encoder_state), (_, decoder_state) = nnx.split(encoder), nnx.split(decoder)
            with ocp.StandardCheckpointer() as checkpointer:
                checkpointer.save(ckpt_dir / "encoder", encoder_state)
                checkpointer.save(ckpt_dir / "decoder", decoder_state)


if __name__ == "__main__":
    train_mnist(num_epochs=200)

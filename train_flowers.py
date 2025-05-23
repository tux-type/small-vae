# ruff: noqa: E402
import logging

import ray
import ray.data

ray.data.DataContext.get_current().enable_progress_bars = False
logging.getLogger("ray.data").setLevel(logging.WARNING)

ray.init(num_cpus=12, num_gpus=1)

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.image
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from jaxlpips import LPIPS
from tqdm import tqdm

from vae.modules import Decoder, Encoder
from vae.vae import VariationalAutoEncoder


@dataclass(frozen=True)
class HyperParameters:
    # Training
    learning_rate: float = 5e-6
    batch_size: int = 32
    epochs: int = 200

    # Loss
    perceptual_scale: float = 0.5
    kl_scale: float = 0.0001

    # Models
    in_features: int = 3
    num_features: int = 128
    latent_features: int = 4
    out_features: int = 3
    feature_multipliers: tuple[int, ...] = (1, 2, 4)

    # Data
    resolution: int = 64
    test_size: float = 0.1
    prefetch_batches: int = 2

    # Seeds
    init_seed: int = 0
    training_seed: int = 1
    sampling_seed: int = 2
    shuffle_seed: int = 42


def crop_resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
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


def normalise_batch(
    samples: dict[str, np.ndarray], size: tuple[int, int] = (32, 32)
) -> dict[str, np.ndarray]:
    data = np.array([crop_resize(image, size) for image in samples["image"]]).astype(np.float32)
    normalised_data = ((data / 255.0) - 0.5) / 0.5
    samples["image"] = normalised_data
    return samples


def loss_fn(
    model: VariationalAutoEncoder,
    x: jax.Array,
    rngs: nnx.Rngs,
    perceptual_loss_fn: LPIPS,
    perceptual_scale: float,
    kl_scale: float,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    posterior, predictions = model(x, rngs)

    kl = -0.5 * jnp.sum(1 + posterior.logvar - (posterior.mean**2) - posterior.var, axis=(1, 2, 3))
    kl_loss = kl.sum() / x.shape[0]
    recon_loss = optax.squared_error(predictions=predictions, targets=x)
    perceptual_loss = perceptual_loss_fn(x, predictions)

    recon_perc_loss = recon_loss + (perceptual_scale * perceptual_loss)
    recon_perc_loss = recon_perc_loss.sum() / recon_perc_loss.shape[0]

    loss = recon_perc_loss + kl_scale * kl_loss
    return loss, (kl_loss, recon_loss.mean(), perceptual_loss, predictions)


@nnx.jit(static_argnums=6, static_argnames="config")
def train_step(
    model: VariationalAutoEncoder,
    optimiser: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    rngs: nnx.Rngs,
    x: jax.Array,
    perceptual_loss_fn: LPIPS,
    config: HyperParameters,
):
    grad_fn = nnx.value_and_grad(f=loss_fn, has_aux=True)
    (loss, aux_data), grads = grad_fn(
        model,
        x,
        rngs,
        perceptual_loss_fn,
        perceptual_scale=config.perceptual_scale,
        kl_scale=config.kl_scale,
    )
    kl_loss, recon_loss, percep_loss, predictions = aux_data
    metrics.update(loss=loss, kl_loss=kl_loss, recon_loss=recon_loss, percep_loss=percep_loss)
    optimiser.update(grads)


def save_sample_images(path: str, original_images: jax.Array, reconstructed_images: jax.Array):
    original_images = jnp.concatenate(original_images, axis=1)
    reconstructed_images = jnp.concatenate(reconstructed_images, axis=1)
    concat_images = jnp.concatenate((original_images, reconstructed_images), axis=0)
    concat_images = ((concat_images + 1) * 127.5 + 0.5).clip(0, 255)
    concat_images = np.array(concat_images.astype(jnp.uint8))
    matplotlib.image.imsave(path, concat_images)


def train_mnist(config: HyperParameters):
    init_rngs = nnx.Rngs(params=config.init_seed)
    training_rngs = nnx.Rngs(latent=config.training_seed)
    sampling_rngs = nnx.Rngs(latent=config.sampling_seed)

    encoder = Encoder(
        in_features=config.in_features,
        num_features=config.num_features,
        latent_features=config.latent_features,
        rngs=init_rngs,
        resolution=config.resolution,
        feature_multipliers=config.feature_multipliers,
        double_latent_features=True,
    )
    decoder = Decoder(
        latent_features=config.latent_features,
        num_features=config.num_features,
        out_features=config.out_features,
        rngs=init_rngs,
        resolution=config.resolution,
        feature_multipliers=config.feature_multipliers,
    )
    vae = VariationalAutoEncoder(
        encoder=encoder,
        decoder=decoder,
        device=jax.devices("gpu")[0],
    )

    # Model for perceptual loss
    lpips = LPIPS(pretrained_network="vgg16")
    lpips.eval()

    train_dataset, validation_dataset = (
        ray.data.read_images("data/oxford_flowers", override_num_blocks=12)
        .map_batches(normalise_batch, fn_kwargs={"size": (config.resolution, config.resolution)})
        .random_shuffle(seed=config.shuffle_seed)
        .train_test_split(test_size=config.test_size)
    )

    optimiser = nnx.Optimizer(vae, optax.adam(config.learning_rate))
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        kl_loss=nnx.metrics.Average("kl_loss"),
        recon_loss=nnx.metrics.Average("recon_loss"),
        percep_loss=nnx.metrics.Average("percep_loss"),
    )

    metrics_history = {"loss": [], "kl_loss": [], "recon_loss": [], "percep_loss": []}

    for epoch in range(config.epochs):
        train_dataset = train_dataset.random_shuffle().materialize()
        progress_bar = tqdm(
            train_dataset.iter_batches(
                prefetch_batches=config.prefetch_batches,
                batch_size=config.batch_size,
                batch_format="numpy",
                drop_last=True,
            )
        )
        for batch in progress_bar:
            x = jnp.array(batch["image"])
            train_step(vae, optimiser, metrics, training_rngs, x, lpips, config=config)

            for metric, value in metrics.compute().items():
                metrics_history[metric].append(value)
            loss = metrics_history["loss"][-1]
            kl_loss = metrics_history["kl_loss"][-1]
            recon_loss = metrics_history["recon_loss"][-1]
            percep_loss = metrics_history["percep_loss"][-1]
            progress_bar.set_description(
                f"loss: {loss:.4f} kl_loss: {kl_loss:.4f} recon_loss: {recon_loss:.4f} percep_loss: {percep_loss:.4f}"
            )

        if epoch % 5 == 0 or epoch == (config.epochs - 1):
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
    config = HyperParameters(
        # Training
        learning_rate=5e-6,
        batch_size=32,
        epochs=200,
        # Loss
        perceptual_scale=0.5,
        kl_scale=0.0001,
        # Models
        in_features=3,
        num_features=128,
        latent_features=4,
        out_features=3,
        feature_multipliers=(1, 2, 4),
        # Data
        resolution=64,
        test_size=0.1,
        prefetch_batches=2,
        # Seeds
        init_seed=0,
        training_seed=1,
        sampling_seed=2,
        shuffle_seed=42,
    )
    train_mnist(config)

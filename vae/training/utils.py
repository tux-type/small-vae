import jax
import jax.numpy as jnp
import matplotlib.image
import numpy as np


def save_sample_images(path: str, original_images: jax.Array, reconstructed_images: jax.Array):
    original_images = jnp.concatenate(original_images, axis=1)
    reconstructed_images = jnp.concatenate(reconstructed_images, axis=1)
    concat_images = jnp.concatenate((original_images, reconstructed_images), axis=0)
    concat_images = ((concat_images + 1) * 127.5 + 0.5).clip(0, 255)
    concat_images = np.array(concat_images.astype(jnp.uint8))
    matplotlib.image.imsave(path, concat_images)

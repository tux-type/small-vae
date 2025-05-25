import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jaxlpips import LPIPS

from vae.loss.discriminator import Discriminator
from vae.vae import VariationalAutoEncoder


def vae_loss_fn(
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


def gan_loss_fn(discriminator: Discriminator, predictions: jax.Array) -> jax.Array:
    logits_fake = discriminator(predictions)
    loss = -jnp.mean(logits_fake)
    return loss


def vae_with_gan_loss_fn(
    model: VariationalAutoEncoder,
    x: jax.Array,
    rngs: nnx.Rngs,
    discriminator: Discriminator,
    perceptual_loss_fn: LPIPS,
    perceptual_scale: float,
    kl_scale: float,
    adversarial_scale: float,
):
    composite_loss, (kl_loss, recon_loss, percep_loss, predictions) = vae_loss_fn(
        model=model,
        x=x,
        rngs=rngs,
        perceptual_loss_fn=perceptual_loss_fn,
        perceptual_scale=perceptual_scale,
        kl_scale=kl_scale,
    )
    # TODO: Add adaptive scaling
    adversarial_loss = gan_loss_fn(discriminator=discriminator, predictions=predictions)
    return composite_loss + adversarial_scale * adversarial_loss


def discriminator_loss_fn(
    model: Discriminator, x: jax.Array, predictions: jax.Array, disc_scale: float
):
    logits_real = model(x)
    logits_fake = model(predictions)

    hinge_loss_real = jnp.mean(nnx.relu(1.0 - logits_real))
    hinge_loss_fake = jnp.mean(nnx.relu(1.0 + logits_fake))
    hinge_loss = 0.5 * (hinge_loss_real + hinge_loss_fake)
    loss = disc_scale * hinge_loss
    return loss

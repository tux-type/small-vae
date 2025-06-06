import jax
from flax import nnx
from jaxlpips import LPIPS

from vae.loss.discriminator import Discriminator
from vae.loss.loss import discriminator_loss_fn, vae_loss_fn, vae_with_gan_loss_fn
from vae.model.vae import VariationalAutoEncoder
from vae.training.hyperparameters import HyperParameters


@nnx.jit(static_argnums=5, static_argnames=("config"))
def train_step(
    model: VariationalAutoEncoder,
    optimiser: nnx.Optimizer,
    rngs: nnx.Rngs,
    x: jax.Array,
    perceptual_loss_fn: LPIPS,
    config: HyperParameters,
) -> tuple[jax.Array, ...]:
    grad_fn = nnx.value_and_grad(f=vae_loss_fn, has_aux=True)
    (loss, aux_data), grads = grad_fn(
        model,
        x=x,
        rngs=rngs,
        perceptual_loss_fn=perceptual_loss_fn,
        perceptual_scale=config.perceptual_scale,
        kl_scale=config.kl_scale,
    )
    kl_loss, recon_loss, perceptual_loss, adversarial_loss, predictions = aux_data
    optimiser.update(grads)
    return loss, kl_loss, recon_loss, perceptual_loss, adversarial_loss, predictions


@nnx.jit(static_argnums=5, static_argnames=("config"))
def train_step_with_gan_loss(
    model: VariationalAutoEncoder,
    optimiser: nnx.Optimizer,
    rngs: nnx.Rngs,
    x: jax.Array,
    perceptual_loss_fn: LPIPS,
    config: HyperParameters,
    discriminator: Discriminator,
) -> tuple[jax.Array, ...]:
    grad_fn = nnx.value_and_grad(f=vae_with_gan_loss_fn, has_aux=True)
    (loss, aux_data), grads = grad_fn(
        model,
        x=x,
        rngs=rngs,
        perceptual_loss_fn=perceptual_loss_fn,
        perceptual_scale=config.perceptual_scale,
        kl_scale=config.kl_scale,
        adversarial_scale=config.adversarial_scale,
        discriminator=discriminator,
        training=True,
    )
    kl_loss, recon_loss, perceptual_loss, adversarial_loss, predictions = aux_data
    optimiser.update(grads)
    return loss, kl_loss, recon_loss, perceptual_loss, adversarial_loss, predictions


@nnx.jit(static_argnums=4, static_argnames=("config"))
def train_step_discriminator(
    discriminator: Discriminator,
    optimiser: nnx.Optimizer,
    x: jax.Array,
    predictions: jax.Array,
    config: HyperParameters,
) -> jax.Array:
    grad_fn = nnx.value_and_grad(f=discriminator_loss_fn, has_aux=False)
    loss, grads = grad_fn(discriminator, x, predictions, config.disc_scale)
    optimiser.update(grads)
    return loss

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jaxlpips import LPIPS

from vae.loss.discriminator import Discriminator
from vae.modules import GaussianPosterior
from vae.vae import VariationalAutoEncoder


def reconstruction_loss(
    model: VariationalAutoEncoder,
    x: jax.Array,
    rngs: nnx.Rngs,
    perceptual_loss_fn: LPIPS,
    perceptual_scale: float,
) -> tuple[jax.Array, tuple[GaussianPosterior, jax.Array, jax.Array, jax.Array]]:
    posterior, predictions = model(x, rngs)
    recon_loss = optax.squared_error(predictions=predictions, targets=x)

    perceptual_loss = perceptual_loss_fn(x, predictions)

    recon_perc_loss = recon_loss + (perceptual_scale * perceptual_loss)
    recon_perc_loss = recon_perc_loss.sum() / recon_perc_loss.shape[0]
    return recon_perc_loss, (posterior, predictions, recon_loss.mean(), perceptual_loss)


def kl_loss_fn(posterior: GaussianPosterior, x: jax.Array):
    kl = -0.5 * jnp.sum(1 + posterior.logvar - (posterior.mean**2) - posterior.var, axis=(1, 2, 3))
    kl_loss = kl.sum() / x.shape[0]
    return kl_loss


def vae_loss_fn(
    model: VariationalAutoEncoder,
    x: jax.Array,
    rngs: nnx.Rngs,
    perceptual_loss_fn: LPIPS,
    perceptual_scale: float,
    kl_scale: float,
    **kwargs,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    recon_perc_loss, (posterior, predictions, recon_loss, perceptual_loss) = reconstruction_loss(
        model=model,
        x=x,
        rngs=rngs,
        perceptual_loss_fn=perceptual_loss_fn,
        perceptual_scale=perceptual_scale,
    )

    # Regularisation
    kl_loss = kl_loss_fn(posterior, x)

    loss = recon_perc_loss + kl_scale * kl_loss
    return loss, (kl_loss, recon_loss, perceptual_loss, jnp.zeros(()), predictions)


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
    training: bool,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    if training:
        # Filter for gradients on only the last layer
        decoder_filter = nnx.All(
            nnx.PathContains("decoder"), nnx.PathContains("feature_aggregation")
        )
        diff_state = nnx.DiffState(0, decoder_filter)

        recon_grad_fn = nnx.value_and_grad(reconstruction_loss, argnums=diff_state, has_aux=True)
        (recon_perc_loss, aux_data), recon_grad = recon_grad_fn(model, x, rngs)
        (posterior, predictions, recon_loss, perceptual_loss) = aux_data

        # TODO: Trying two passes to see if JIT compiler is smart enough to only do 1 pass
        # https://github.com/google/flax/discussions/3316
        def gan_loss_fn_fwd(
            model: VariationalAutoEncoder, discriminator: Discriminator, x: jax.Array
        ):
            _, predictions = model(x, rngs)
            return gan_loss_fn(discriminator, predictions)

        gan_grad_fn = nnx.value_and_grad(gan_loss_fn_fwd, argnums=diff_state, has_aux=False)
        adversarial_loss, gan_grad = gan_grad_fn(model, x, rngs)

        adversarial_weight = jnp.clip(
            optax.global_norm(recon_grad) / (optax.global_norm(gan_grad) + 1e-4),
            0,
            1e4,
        )
        adversarial_weight = jax.lax.stop_gradient(adversarial_weight)

    else:
        recon_perc_loss, aux_data = reconstruction_loss(
            model, x, rngs, perceptual_loss_fn=perceptual_loss_fn, perceptual_scale=perceptual_scale
        )
        (posterior, predictions, recon_loss, perceptual_loss) = aux_data
        adversarial_loss = gan_loss_fn(discriminator=discriminator, predictions=predictions)
        adversarial_weight = 1.0

    # Regularisation
    kl_loss = kl_loss_fn(posterior, x)

    loss = (
        recon_perc_loss
        + kl_loss * kl_scale
        + adversarial_scale * adversarial_weight * adversarial_loss
    )

    return loss, (kl_loss, recon_loss, perceptual_loss, adversarial_loss, predictions)


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

from dataclasses import dataclass


@dataclass(frozen=True)
class HyperParameters:
    # Training
    learning_rate: float = 5e-6
    batch_size: int = 32
    epochs: int = 200
    gan_start_epoch: int = 20

    # Loss
    perceptual_scale: float = 0.5
    kl_scale: float = 0.0001
    adversarial_scale: float = 0.5
    disc_scale: float = 1.0

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

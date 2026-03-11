from dataclasses import dataclass


@dataclass
class Parameters:
    """Class to hold the parameters for the VAE model."""

    lr: float = 1e-3

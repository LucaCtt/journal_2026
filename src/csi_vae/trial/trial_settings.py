from pydantic_settings import BaseSettings


class TrialSettings(BaseSettings):
    """Trial settings, loaded from environment variables or .env file."""

    study_name: str = "default"
    """Name of the study to which this trial belongs."""
    trial_number: int = 0
    """Unique identifier for the trial within the study."""
    dataset_path: str = "dataset.h5"
    """Path to the dataset to be used for training and evaluation."""
    queue_url: str | None = None
    """URL of the SQS message queue. If set to None, the trial will not send results to a queue."""
    aws_region: str = "us-east-1"
    """AWS region, for configuring the SQS client when used."""
    window_size: int = 450
    """Size of the window to use when segmenting the data."""
    n_subcarriers: int = 256
    """Number of subcarriers in the CSI data."""
    n_activities: int = 12
    """Number of activities (classes) in the dataset."""
    n_antennas: int = 4
    """Number of antennas in the CSI data."""
    stride: int = 50
    """Stride to use when segmenting the data (number of samples to skip between windows)."""
    n_epochs: int = 150
    """Number of epochs to train the autoencoder."""
    collapse_threshold: float = 1e-3
    """Threshold for KL divergence loss to detect posterior collapse."""
    patience: int = 5
    """Number of epochs to wait before raising a collapse error or early stop."""
    plateau_min_delta: float = 1e-4
    """Minimum change in validation loss to qualify as an improvement for early stopping."""

    seed: int = 42
    """Random seed for reproducibility."""
    batch_size: int = 128
    """Batch size for training the autoencoder."""
    lr: float = 2e-3
    """Learning rate for training the autoencoder."""
    latent_dim: int = 2
    """Dimensionality of the latent space in the autoencoder."""

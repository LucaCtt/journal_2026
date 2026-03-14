from typing import NamedTuple, TypeVar

from pydantic_settings import BaseSettings

NumberT = TypeVar("NumberT", int, float)


class ParamRange[NumberT](NamedTuple):
    """Defines a hyperparameter with a name and a range."""

    min: NumberT
    max: NumberT


class ParamCategorical[T](NamedTuple):
    """Defines a hyperparameter with a name and a list of values."""

    values: list[T]


class LauncherSettings(BaseSettings):
    """Application settings, loaded from environment variables or .env file."""

    study_name: str = "default"
    """Name of the Optuna study."""
    journal_path: str | None = f"out/{study_name}/study.log"
    """Path to the Optuna journal file for this study."""
    n_trials: int = 10
    """Total number of Optuna trials to run."""
    starter_seed: int = 42
    """Seed used for generating the trials' seeds"""
    n_seeds_per_trial: int = 10
    """Number of different random seeds to run for each trial configuration."""
    max_pruned_seeds: int = 2
    """Maximum number of seed collapses before pruning the trial."""
    min_accuracy_delta: float = 0.01
    """Minimum improvement in best-trial accuracy required to continue to the next latent dim."""
    aws_job_queue: str = "CSIVAEJobQueue"
    """Name of the AWS Batch job queue to submit trials to."""
    aws_job_definition: str = "CSIVAEJobDefinition"
    """Name of the AWS Batch job definition to use."""
    aws_region: str = "us-east-1"
    """Default AWS region to use for Batch operations."""
    poll_interval: int = 30
    """Seconds to wait between polling AWS Batch for job status."""
    poll_timeout: int = 600
    """Maximum seconds to wait for a batch of trials to complete before giving up."""

    batch_size: ParamRange[int] = ParamRange(min=64, max=256)
    lr: ParamRange[float] = ParamRange(min=1e-4, max=1e-2)
    kl_max: ParamRange[float] = ParamRange(min=1.0, max=4.0)
    latent_dim: ParamRange[int] = ParamRange(min=1, max=10)
    conv_channels: ParamRange[int] = ParamRange(min=8, max=64)
    conv_layers_spec: ParamCategorical[int] = ParamCategorical(values=[*range(4)])

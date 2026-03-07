from pydantic_settings import BaseSettings


class LauncherSettings(BaseSettings):
    """Application settings, loaded from environment variables or .env file."""

    study_name: str = "default"
    """Name of the Optuna study."""
    journal_path: str = f"out/{study_name}/study.log"
    """Path to the Optuna journal file for this study."""
    n_trials: int = 10
    """Total number of Optuna trials to run."""
    trials_batch_size: int = 2
    """Number of trials to submit in each batch to AWS Batch."""
    poll_interval: int = 30
    """Seconds to wait between polling AWS Batch for job status."""
    aws_job_queue: str = "csi_vae_job_queue"
    """Name of the AWS Batch job queue to submit trials to."""
    aws_job_definition: str = "csi_vae_job_def"
    """Name of the AWS Batch job definition to use."""

    param_lr_min: float = 1e-4
    param_lr_max: float = 1e-2

    @property
    def n_batches(self) -> int:
        """Calculate how many batches of jobs to submit based on total trials and batch size."""
        return (self.n_trials + self.trials_batch_size - 1) // self.trials_batch_size

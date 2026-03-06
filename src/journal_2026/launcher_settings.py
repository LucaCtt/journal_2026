from pydantic_settings import BaseSettings


class LauncherSettings(BaseSettings):
    """Application settings, loaded from environment variables or .env file."""

    study_name = "default"
    study_journal_path = f"out/{study_name}/study.log"
    study_n_trials = 10
    study_batch_size = 2
    study_poll_interval = 30

    param_lr_min = 1e-4
    param_lr_max = 1e-2

    aws_region = "us-east-1"
    aws_job_queue = "csi_vae_job_queue"
    aws_job_def = "csi_vae_job_def:1"

    @property
    def n_batches(self) -> int:
        """Calculate how many batches of jobs to submit based on total trials and batch size."""
        return (self.study_n_trials + self.study_batch_size - 1) // self.study_batch_size

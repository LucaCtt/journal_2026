from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings, loaded from environment variables or .env file."""

    study_name = "default"
    study_db_url = "postgresql://user:password@localhost:5432/mydatabase"
    study_n_trials = 100
    study_poll_interval = 30

    param_lr_min = 1e-4
    param_lr_max = 1e-2

    aws_region = "us-east-1"
    aws_job_queue = "csi_vae_job_queue"
    aws_job_def = "csi_vae_job_def:1"

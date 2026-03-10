from pydantic_settings import BaseSettings


class TrialSettings(BaseSettings):
    """Trial settings, loaded from environment variables or .env file."""

    study_name: str = "default"
    """Name of the study to which this trial belongs."""
    trial_number: int = 0
    """Unique identifier for the trial within the study."""
    queue_url: str | None = None
    """URL of the SQS message queue. If set to None, the trial will not send results to a queue."""
    aws_region: str = "us-east-1"
    """AWS region, for configuring the SQS client when used."""

    param_lr: float = 1e-4

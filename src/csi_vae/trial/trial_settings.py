from pydantic_settings import BaseSettings


class TrialSettings(BaseSettings):
    """Trial settings, loaded from environment variables or .env file."""

    study_name: str = "default"
    """Name of the study to which this trial belongs."""
    trial_number: int = 0
    """Unique identifier for the trial within the study."""
    queue_type: str = "local"
    """Type of message queue to use for communication (e.g., SQS or local)."""
    queue_url: str | None = None
    """URL of the SQS message queue, if applicable."""

    param_lr: float = 1e-4

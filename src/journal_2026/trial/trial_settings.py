from pydantic_settings import BaseSettings


class TrialSettings(BaseSettings):
    """Trial settings, loaded from environment variables or .env file."""

    study_name: str = "default"
    study_trial_id: int = 0

    param_lr: float = 1e-4

    queue_url: str = "https://example.com/queue"

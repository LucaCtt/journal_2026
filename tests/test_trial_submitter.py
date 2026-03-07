from unittest.mock import MagicMock, patch

import pytest

from csi_vae.trial.trial_settings import TrialSettings
from csi_vae.trial_submitter import TrialSubmitter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_batch():
    with patch("boto3.client", return_value=MagicMock()) as mock_client:
        yield mock_client.return_value


@pytest.fixture
def submitter(mock_batch):
    return TrialSubmitter(job_queue="test-queue", job_definition="test-def")


@pytest.fixture
def settings():
    return TrialSettings(
        study_name="my-study",
        trial_number=3,
        queue_type="sqs",
        queue_url="https://sqs.fake/queue",
        param_lr=1e-3,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTrialSubmitter:
    def test_returns_job_id(self, submitter, mock_batch, settings):
        mock_batch.submit_job.return_value = {"jobId": "abc-123"}
        result = submitter.submit(settings)
        assert result == "abc-123"

    def test_job_name_format(self, submitter, mock_batch, settings):
        mock_batch.submit_job.return_value = {"jobId": "x"}
        submitter.submit(settings)
        _, kwargs = mock_batch.submit_job.call_args
        assert kwargs["jobName"] == "my-study-trial-3"

    def test_uses_configured_queue_and_definition(self, submitter, mock_batch, settings):
        mock_batch.submit_job.return_value = {"jobId": "x"}
        submitter.submit(settings)
        _, kwargs = mock_batch.submit_job.call_args
        assert kwargs["jobQueue"] == "test-queue"
        assert kwargs["jobDefinition"] == "test-def"

    def test_environment_contains_all_settings_fields(self, submitter, mock_batch, settings):
        mock_batch.submit_job.return_value = {"jobId": "x"}
        submitter.submit(settings)
        _, kwargs = mock_batch.submit_job.call_args
        env = {e["name"]: e["value"] for e in kwargs["containerOverrides"]["environment"]}
        for key, value in settings.model_dump().items():
            assert key in env
            assert env[key] == str(value)

    def test_environment_values_are_strings(self, submitter, mock_batch, settings):
        mock_batch.submit_job.return_value = {"jobId": "x"}
        submitter.submit(settings)
        _, kwargs = mock_batch.submit_job.call_args
        for entry in kwargs["containerOverrides"]["environment"]:
            assert isinstance(entry["value"], str)

    def test_none_queue_url_not_serialised(self, submitter, mock_batch):
        s = TrialSettings(study_name="s", trial_number=0, queue_url=None)
        mock_batch.submit_job.return_value = {"jobId": "x"}
        submitter.submit(s)
        _, kwargs = mock_batch.submit_job.call_args
        env = {e["name"]: e["value"] for e in kwargs["containerOverrides"]["environment"]}
        assert "queue_url" not in env

    def test_default_settings_produce_valid_job(self, submitter, mock_batch):
        s = TrialSettings()
        mock_batch.submit_job.return_value = {"jobId": "y"}
        result = submitter.submit(s)
        assert result == "y"
        _, kwargs = mock_batch.submit_job.call_args
        assert kwargs["jobName"] == "default-trial-0"

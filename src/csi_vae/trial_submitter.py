
import boto3

from csi_vae.trial.trial_settings import TrialSettings


class TrialSubmitter:
    """Submits trials as AWS Batch jobs."""

    def __init__(self, job_queue: str, job_definition: str) -> None:
        """Initialize the submitter with AWS Batch client and job configuration."""
        self.__batch_client = boto3.client("batch")
        self.__job_queue = job_queue
        self.__job_definition = job_definition

    def submit(self, settings: TrialSettings) -> str:
        """Submit a job to AWS Batch with the given environment variables."""
        response = self.__batch_client.submit_job(
            jobName=f"{settings.study_name}-trial-{settings.trial_number}",
            jobQueue=self.__job_queue,
            jobDefinition=self.__job_definition,
            containerOverrides={
                "environment": [{"name": k, "value": str(v)} for k, v in settings.model_dump().items()],
            },
        )
        return response["jobId"]


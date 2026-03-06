import shutil
import subprocess
from typing import Protocol

from botocore.client import BaseClient

from csi_vae.trial.trial_settings import TrialSettings


class TrialJobSubmitter(Protocol):
    """Submits trial jobs to a compute backend."""

    def submit(self, job_name: str, settings: TrialSettings) -> str:
        """Submit a job and return a job ID.

        Arguments:
            job_name: A unique name for the job, e.g. "study-trial-42".
            settings: The trial settings to pass as environment variables.

        """
        ...


class AWSBatchTrialJobSubmitter:
    """Submits trials as AWS Batch jobs."""

    def __init__(self, batch: BaseClient, job_queue: str, job_definition: str) -> None:
        """Initialize the submitter with AWS Batch client and job configuration."""
        self.__batch = batch
        self.__job_queue = job_queue
        self.__job_definition = job_definition

    def submit(self, job_name: str, settings: TrialSettings) -> str:
        """Submit a job to AWS Batch with the given environment variables."""
        response = self.__batch.submit_job(
            jobName=job_name,
            jobQueue=self.__job_queue,
            jobDefinition=self.__job_definition,
            containerOverrides={
                "environment": [{"name": k, "value": str(v)} for k, v in settings.model_dump().items()],
            },
        )
        return response["jobId"]


class LocalProcessTrialJobSubmitter:
    """Submits trials as local subprocesses."""

    def submit(self, _: str, settings: TrialSettings) -> str:
        """Submit a local subprocess with the given environment variables."""
        python_path = shutil.which("python") or "python"
        proc = subprocess.Popen(  # noqa: S603
            [python_path, "-m", "csi_vae.trial.trial"],
            env={k: str(v) for k, v in settings.model_dump().items()},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return str(proc.pid)

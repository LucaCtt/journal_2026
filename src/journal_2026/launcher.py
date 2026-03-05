import logging
import time

import boto3
import optuna
from botocore.client import BaseClient
from rich.logging import RichHandler

from journal_2026.settings import Settings

settings = Settings()

# Logging config
handler = RichHandler(level=logging.INFO, show_path=False)
logging.basicConfig(level=logging.INFO, handlers=[handler], format="%(message)s")
logger = logging.getLogger("rich")

# Route Optuna logs through app logger
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()


def _submit_trial(
    batch: BaseClient,
    job_queue: str,
    job_def: str,
    study_name: str,
    storage: str,
    trial_id: int,
) -> str:
    """Submit a single Batch job for one Optuna trial."""
    response = batch.submit_job(
        jobName=f"{study_name}-trial-{trial_id}",
        jobQueue=job_queue,
        jobDefinition=job_def,
        containerOverrides={
            "environment": [
                {"name": "OPTUNA_STORAGE", "value": storage},
                {"name": "OPTUNA_STUDY", "value": study_name},
                {"name": "OPTUNA_TRIAL_ID", "value": str(trial_id)},
            ],
        },
    )
    job_id = response["jobId"]
    logger.info("Submitted trial %d → Batch job %d", trial_id, job_id)
    return job_id


def _wait_for_jobs(batch: BaseClient, job_ids: list[str], poll_interval: int) -> None:
    """Poll until all jobs reach a terminal state."""
    pending = set(job_ids)

    while pending:
        time.sleep(poll_interval)

        # describe_jobs accepts up to 100 IDs at once
        chunks = [list(pending)[i : i + 100] for i in range(0, len(pending), 100)]
        for chunk in chunks:
            resp = batch.describe_jobs(jobs=chunk)

            for job in resp["jobs"]:
                status = job["status"]
                jid = job["jobId"]
                if status in ("SUCCEEDED", "FAILED"):
                    logger.info("Batch job %d → %s", jid, status)
                    pending.discard(jid)


def main() -> None:
    """Create Optuna study, submit Batch jobs, and wait for results."""
    study = optuna.create_study(
        study_name=settings.study_name,
        storage=settings.study_db_url,
        direction="maximize",  # Maximize validation accuracy
    )

    batch = boto3.client("batch", region_name=settings.aws_region)
    job_ids = []

    for _ in range(settings.study_n_trials):
        # Ask Optuna to create a trial (assigns a trial_id and samples params)
        trial = study.ask()
        logger.info("Submitting trial %d with params: %s", trial.number, trial.params)

        job_id = _submit_trial(
            batch,
            settings.aws_job_queue,
            settings.aws_job_def,
            settings.study_name,
            settings.study_db_url,
            trial.number,
        )
        job_ids.append(job_id)

    logger.info("All %d trials submitted, waiting for completion...", len(job_ids))
    _wait_for_jobs(batch, job_ids, settings.study_poll_interval)

    # Print final results
    study = optuna.load_study(study_name=settings.study_name, storage=settings.study_db_url)
    best = study.best_trial

    logger.info("All trials completed.")
    logger.info("Best trial: #%d  accuracy=%.4f", best.number, best.value)
    logger.info("Params: %s", best.params)


if __name__ == "__main__":
    main()

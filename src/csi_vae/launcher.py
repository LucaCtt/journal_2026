import json
import logging
import time

import boto3
import optuna
from botocore.client import BaseClient
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from rich.logging import RichHandler

from csi_vae.launcher_settings import LauncherSettings

settings = LauncherSettings()

# Logging config
handler = RichHandler(level=logging.INFO, show_path=False)
logging.basicConfig(level=logging.INFO, handlers=[handler], format="%(message)s")
logger = logging.getLogger("rich")

# Route Optuna logs through app logger
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()


def _submit_trial(batch: BaseClient, trial: optuna.Trial, queue_url: str) -> None:
    """Submit a single Batch job for one Optuna trial."""
    lr = trial.suggest_float("lr", settings.param_lr_min, settings.param_lr_max, log=True)

    response = batch.submit_job(
        jobName=f"{settings.study_name}-trial-{trial.number}",
        jobQueue=settings.aws_job_queue,
        jobDefinition=settings.aws_job_def,
        containerOverrides={
            "environment": [
                {"name": "STUDY_NAME", "value": settings.study_name},
                {"name": "STUDY_TRIAL_ID", "value": str(trial.number)},
                {"name": "PARAM_LR", "value": str(lr)},
                {"name": "QUEUE_URL", "value": queue_url},
            ],
        },
    )

    job_id = response["jobId"]
    logger.info("Submitted trial %d → Batch job %d", trial.number, job_id)


def _collect_results(
    sqs: BaseClient,
    queue_url: str,
    trial_ids: list[int],
    poll_interval: int,
) -> list[tuple[int, float | None]]:
    """Poll until all jobs reach a terminal state."""
    remaining_trials = set(trial_ids)
    results = []

    while remaining_trials:
        time.sleep(poll_interval)

        # Check SQS for completed jobs
        resp = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=10)
        messages = resp.get("Messages", [])
        for msg in messages:
            body = json.loads(msg["Body"])
            trial_id = body["trialId"]
            if trial_id in remaining_trials:
                status = body["status"]
                accuracy = body["accuracy"] if status == "SUCCEEDED" else None

                if status == "SUCCEEDED":
                    logger.info("Trial %d completed successfully.", trial_id)
                else:
                    logger.info("Trial %d failed during execution.", trial_id)

                results.append((trial_id, accuracy))
                remaining_trials.remove(trial_id)

            sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=msg["ReceiptHandle"])

    return results


def main() -> None:
    """Create Optuna study, submit Batch jobs, and wait for results."""
    study = optuna.create_study(
        study_name=settings.study_name,
        storage=JournalStorage(JournalFileBackend(settings.study_journal_path)),
        direction="maximize",  # Maximize validation accuracy
    )

    batch = boto3.client("batch", region_name=settings.aws_region)
    sqs = boto3.client("sqs", region_name=settings.aws_region)
    queue_url = sqs.create_queue(QueueName=settings.study_name)["QueueUrl"]

    try:
        for i in range(settings.n_batches):
            logger.info("Starting batch %d/%d.", i + 1, settings.n_batches)

            pending_trials: dict[int, optuna.Trial] = {}

            for _ in range(settings.study_batch_size):
                # Ask Optuna to create a trial (assigns a trial_id and samples params)
                trial = study.ask()
                logger.info("Submitting trial %d with params: %s", trial.number, trial.params)

                _submit_trial(batch, trial, queue_url)
                pending_trials[trial.number] = trial

            logger.info("Batch submitted, waiting for completion...")
            results = _collect_results(
                sqs,
                queue_url,
                [trial.number for trial in pending_trials.values()],
                settings.study_poll_interval,
            )
            for trial_id, accuracy in results:
                trial = pending_trials[trial_id]
                if accuracy is not None:
                    study.tell(trial, values=accuracy)
                else:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)

        # Print final results
        best = study.best_trial

        logger.info("All trials completed.")
        logger.info("Best trial: #%d  accuracy=%.4f", best.number, best.value)
        logger.info("Params: %s", best.params)
    finally:
        sqs.delete_queue(QueueUrl=queue_url)


if __name__ == "__main__":
    main()

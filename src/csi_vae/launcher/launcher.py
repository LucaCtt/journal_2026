import logging
import time

import boto3
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from rich.logging import RichHandler

from csi_vae.launcher.launcher_settings import LauncherSettings
from csi_vae.queue import SQSTrialMessagesQueue, TrialMessagesQueue
from csi_vae.submitter import AWSBatchTrialJobSubmitter, TrialJobSubmitter
from csi_vae.trial.trial_settings import TrialSettings

settings = LauncherSettings()

# Logging config
handler = RichHandler(level=logging.INFO, show_path=False)
logging.basicConfig(level=logging.INFO, handlers=[handler], format="%(message)s")
logger = logging.getLogger("rich")

# Route Optuna logs through app logger
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()


def _submit_trial(submitter: TrialJobSubmitter, trial: optuna.Trial, queue_url: str) -> None:
    """Submit a single Batch job for one Optuna trial."""
    lr = trial.suggest_float("lr", settings.param_lr_min, settings.param_lr_max, log=True)

    job_id = submitter.submit(
        f"{settings.study_name}-trial-{trial.number}",
        TrialSettings(
            study_name=settings.study_name,
            study_trial_id=trial.number,
            param_lr=lr,
            queue_url=queue_url,
        ),
    )
    logger.info("Submitted trial %d → Batch job %d", trial.number, job_id)


def _collect_results(
    queue: TrialMessagesQueue,
    trial_ids: list[int],
    poll_interval: int,
) -> list[tuple[int, float | None]]:
    """Poll until all jobs reach a terminal state."""
    remaining_trials = set(trial_ids)
    results = []

    while remaining_trials:
        time.sleep(poll_interval)

        # Check the queue for completed jobs
        messages = queue.pop(max_messages=10)
        for msg in messages:
            trial_id = msg.get("trial_id", None)
            if trial_id and trial_id in remaining_trials:
                status = msg.get("status", None)
                accuracy = msg.get("accuracy", None) if status == "SUCCEEDED" else None

                if status == "SUCCEEDED":
                    logger.info("Trial %d completed successfully.", trial_id)
                else:
                    logger.info("Trial %d failed during execution.", trial_id)

                results.append((trial_id, accuracy))
                remaining_trials.remove(trial_id)

    return results


def run_study(submitter: TrialJobSubmitter, queue: TrialMessagesQueue) -> None:
    """Create Optuna study, submit Batch jobs, and wait for results."""
    study = optuna.create_study(
        study_name=settings.study_name,
        storage=JournalStorage(JournalFileBackend(settings.study_journal_path)),
        direction="maximize",  # Maximize validation accuracy
    )

    for i in range(settings.n_batches):
        logger.info("Starting batch %d/%d.", i + 1, settings.n_batches)

        pending_trials: dict[int, optuna.Trial] = {}

        for _ in range(settings.study_batch_size):
            # Ask Optuna to create a trial (assigns a trial_id and samples params)
            trial = study.ask()
            pending_trials[trial.number] = trial
            logger.info("Submitting trial %d with params: %s", trial.number, trial.params)

            _submit_trial(submitter, trial, queue.url)

        logger.info("Batch submitted, waiting for completion...")
        results = _collect_results(
            queue,
            list(pending_trials.keys()),
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


def main() -> None:
    """Create Optuna study, submit Batch jobs, and wait for results."""
    batch = boto3.client("batch", region_name=settings.aws_region)
    submitter = AWSBatchTrialJobSubmitter(batch, settings.aws_job_queue, settings.aws_job_def)

    sqs = boto3.client("sqs", region_name=settings.aws_region)
    queue_url = sqs.create_queue(QueueName=settings.study_name)["QueueUrl"]
    queue = SQSTrialMessagesQueue(sqs, queue_url)

    try:
        run_study(submitter, queue)
    finally:
        sqs.delete_queue(QueueUrl=queue_url)


if __name__ == "__main__":
    main()

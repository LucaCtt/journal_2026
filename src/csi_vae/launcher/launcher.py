import logging
import time

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from rich.logging import RichHandler

from csi_vae.launcher.launcher_settings import LauncherSettings
from csi_vae.messages_queue import MessagesQueue
from csi_vae.trial.trial_settings import TrialSettings
from csi_vae.trial_submitter import TrialSubmitter

# Logging config
handler = RichHandler(level=logging.INFO, show_path=False)
logging.basicConfig(level=logging.INFO, handlers=[handler], format="%(message)s")
logger = logging.getLogger("rich")

# Route Optuna logs through app logger
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()


def _collect_results(
    queue: MessagesQueue,
    pending_trials: list[int],
    poll_interval: int,
    timeout: int = 3600,
) -> list[tuple[int, float | None]]:
    """Poll until all jobs reach a terminal state."""
    remaining_trials = set(pending_trials)
    results = []
    start_time = time.time()

    while remaining_trials:
        if time.time() - start_time > timeout:
            msg = "Timeout reached while waiting for trial results."
            raise TimeoutError(msg)

        time.sleep(poll_interval)

        # Check the queue for completed jobs
        messages = queue.pop(max_messages=10)
        for msg in messages:
            trial_id = msg.get("trial_id", None)
            if trial_id is not None and trial_id in remaining_trials:
                status = msg.get("status", None)
                accuracy = msg.get("accuracy", None) if status == "SUCCEEDED" else None

                if status == "SUCCEEDED":
                    logger.info("Trial %d completed successfully.", trial_id)
                else:
                    logger.info("Trial %d failed during execution.", trial_id)

                results.append((trial_id, accuracy))
                remaining_trials.remove(trial_id)

    return results


def run_study(submitter: TrialSubmitter, queue: MessagesQueue, settings: LauncherSettings) -> None:
    """Create Optuna study, submit Batch jobs, and wait for results."""
    study = optuna.create_study(
        study_name=settings.study_name,
        storage=JournalStorage(JournalFileBackend(settings.journal_path)) if settings.journal_path else None,
        direction="maximize",  # Maximize validation accuracy
    )

    for i in range(settings.n_batches):
        logger.info("Starting batch %d/%d.", i + 1, settings.n_batches)

        pending_trials: dict[int, optuna.Trial] = {}

        for _ in range(settings.trials_batch_size):
            # Ask Optuna to create a trial (assigns a trial_id and samples params)
            trial = study.ask()
            pending_trials[trial.number] = trial
            logger.info("Submitting trial %d with params: %s", trial.number, trial.params)

            lr = trial.suggest_float("lr", settings.param_lr_min, settings.param_lr_max, log=True)

            job_id = submitter.submit(
                TrialSettings(
                    study_name=settings.study_name,
                    trial_number=trial.number,
                    param_lr=lr,
                    queue_url=queue.url,
                ),
            )
            logger.info("Submitted trial %d → Batch job %d", trial.number, job_id)

        logger.info("Batch submitted, waiting for completion...")
        results = _collect_results(
            queue,
            list(pending_trials.keys()),
            settings.poll_interval,
        )
        for trial_id, accuracy in results:
            trial = pending_trials[trial_id]
            if accuracy is not None:
                study.tell(trial, values=accuracy)
            else:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)

    if not any(t.value is not None for t in study.trials):
        logger.warning("No successful trials were completed.")
        return

    # Print final results
    best = study.best_trial

    logger.info("All trials completed.")
    logger.info("Best trial: #%d  accuracy=%.4f", best.number, best.value)
    logger.info("Params: %s", best.params)


def run_launcher(settings: LauncherSettings | None = None) -> None:
    """Create Optuna study, submit Batch jobs, and wait for results."""
    settings = LauncherSettings() if settings is None else settings

    submitter = TrialSubmitter(settings.aws_job_queue, settings.aws_job_definition)
    queue = MessagesQueue()
    queue.create(settings.study_name)

    try:
        run_study(submitter, queue, settings)
    finally:
        queue.destroy()


if __name__ == "__main__":
    run_launcher()

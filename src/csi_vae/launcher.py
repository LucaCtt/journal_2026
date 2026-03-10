import logging
import time
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState
from rich.logging import RichHandler

from csi_vae.messages_queue import MessagesQueue
from csi_vae.settings import LauncherSettings
from csi_vae.trial import MessageType, TrialSettings
from csi_vae.trial_submitter import TrialSubmitter

# Logging config
handler = RichHandler(level=logging.DEBUG, show_path=False)
logging.basicConfig(level=logging.DEBUG, handlers=[handler], format="%(message)s")
logger = logging.getLogger("rich")

# Route Optuna logs through app logger
optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()


def _setup_study(study_name: str, journal_path: str | None) -> optuna.Study:
    """Create or load an Optuna study with optional journal storage.

    Arguments:
        study_name: Name of the Optuna study to create or load.
        journal_path: Optional file path for Optuna journal storage.
            If provided, the study will be saved to this file, allowing it to be resumed later.
            If None, the study will only exist in memory and cannot be resumed after the program exits.

    Returns:
        An Optuna Study object, either newly created or loaded from the journal storage.

    """
    if journal_path:
        logger.info("Using journal storage at: %s", journal_path)
        Path(journal_path).parent.mkdir(parents=True, exist_ok=True)

    return optuna.create_study(
        study_name=study_name,
        storage=JournalStorage(JournalFileBackend(journal_path)) if journal_path else None,
        load_if_exists=True,
        direction="maximize",  # Maximize validation accuracy
    )


def _collect_results(
    queue: MessagesQueue,
    pending_trials: list[int],
    poll_interval: int,
    timeout: int,
) -> list[tuple[int, float | None]]:
    """Wait for results of pending trials by polling the message queue.

    Arguments:
        queue: MessagesQueue instance to poll for trial results.
        pending_trials: List of trial IDs that are still pending and for which we are waiting for results.
        poll_interval: Time in seconds to wait between polling attempts.
        timeout: Maximum time in seconds to wait for results before giving up and raising a TimeoutError.

    Returns:
        A list of tuples containing trial IDs and their corresponding accuracies (or None if the trial failed).

    Raises:
        TimeoutError: If the timeout is reached while waiting for trial results.

    """
    remaining_trials = set(pending_trials)
    results: list[tuple[int, float | None]] = []
    start_time = time.time()

    while remaining_trials:
        if time.time() - start_time > timeout:
            msg = "Timeout reached while waiting for trial results."
            raise TimeoutError(msg)

        time.sleep(poll_interval)

        # Check the queue for completed jobs
        messages = queue.pop()
        for msg in messages:
            # Reset the timer on each received message to allow for long-running trials
            start_time = time.time()

            trial_id = msg["trial_id"]
            if trial_id not in remaining_trials:
                continue

            status = MessageType(msg["status"])

            match status:
                case MessageType.STARTING:
                    logger.debug("Trial %d is starting...", trial_id)
                case MessageType.SUCCESS:
                    logger.info("Trial %d completed with accuracy %.4f.", trial_id, msg["accuracy"])
                    results.append((trial_id, msg["accuracy"]))
                    remaining_trials.remove(trial_id)
                case MessageType.ERROR:
                    logger.error("Trial %d failed with error: %s", trial_id, msg.get("error", "Unknown error"))
                    results.append((trial_id, None))
                    remaining_trials.remove(trial_id)
                case _:
                    logger.warning("Received unknown status '%s' for trial %d", status, trial_id)

    return results


def run_study(submitter: TrialSubmitter, queue: MessagesQueue, settings: LauncherSettings) -> None:
    """Create Optuna study, submit Batch jobs, and wait for results.

    Arguments:
        submitter: TrialSubmitter instance to submit trials as Batch jobs.
        queue: MessagesQueue instance to poll for trial results.
        settings: LauncherSettings object containing configuration for the study and trials.

    """
    study = _setup_study(settings.study_name, settings.journal_path)

    # Get any pending trials if the study already exists (e.g. from a previously stopped run)
    previous_trials = [
        trial.number for trial in study.trials if trial.state in [TrialState.WAITING, TrialState.RUNNING]
    ]
    if previous_trials:
        logger.info("Found %d pending trials: %s, trying to collect results...", len(previous_trials), previous_trials)
        results = _collect_results(queue, previous_trials, settings.poll_interval, settings.poll_timeout)
        for trial_id, accuracy in results:
            study.tell(trial_id, values=accuracy)

    # Run new batches of trials
    for i in range(settings.n_batches):
        pending_trials: list[int] = []

        logger.info("Starting batch %d/%d.", i + 1, settings.n_batches)
        for _ in range(settings.trials_batch_size):
            # Ask Optuna to create a trial (assigns a trial_id and samples params)
            trial = study.ask()
            pending_trials.append(trial.number)

            lr = trial.suggest_float("lr", settings.param_lr_min, settings.param_lr_max, log=True)

            job_id = submitter.submit(
                TrialSettings(
                    study_name=settings.study_name,
                    trial_number=trial.number,
                    param_lr=lr,
                    queue_url=queue.url,
                    aws_region=queue.region_name,
                ),
            )
            logger.info("Submitted trial %d → Batch job %s, with params: %s", trial.number, job_id, trial.params)

        results = _collect_results(queue, pending_trials, settings.poll_interval, settings.poll_timeout)
        for trial_id, accuracy in results:
            study.tell(trial_id, values=accuracy)

    if all(t.value is None for t in study.trials):
        logger.warning("All trials failed.")
        return

    # Print final results
    best = study.best_trial

    logger.info("All trials completed.")
    logger.info("Best trial: #%d  accuracy=%.4f", best.number, best.value)
    logger.info("Params: %s", best.params)


def run_launcher(settings: LauncherSettings | None = None) -> None:
    """Create Optuna study, submit Batch jobs, and wait for results.

    Arguments:
        settings: Optional LauncherSettings object. If None, default settings will be used.

    """
    settings = LauncherSettings() if settings is None else settings

    submitter = TrialSubmitter(settings.aws_job_queue, settings.aws_job_definition, settings.aws_region)
    queue = MessagesQueue(settings.aws_region)
    queue.create(settings.study_name)

    try:
        run_study(submitter, queue, settings)
    finally:
        queue.destroy()


if __name__ == "__main__":
    run_launcher()

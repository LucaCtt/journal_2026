import logging
import math
import random
import statistics
import time
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.terminator import (
    EMMREvaluator,
    Terminator,
    TerminatorCallback,
    report_cross_validation_scores,
)
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


def _generate_seeds(starter_seed: int, n_seeds: int) -> list[int]:
    """Generate a fixed list of seeds from a starter seed.

    Arguments:
        starter_seed: The seed to use for the random number generator.
        n_seeds: The number of seeds to generate.

    Returns:
        A list of n_seeds integers to use as seeds for the trials.

    """
    rng = random.Random(starter_seed)
    return [rng.randint(0, 2**31 - 1) for _ in range(n_seeds)]


def _make_study(study_name: str, journal_path: str | None) -> optuna.Study:
    """Create (or load) an Optuna study backed by a journal file.

    Arguments:
        study_name: The name of the study to create or load.
        journal_path: The path to the journal file to use for storage.
            If None, the study will be created without persistent storage.

    Returns:
        An Optuna Study object.

    """
    if journal_path:
        Path(journal_path).parent.mkdir(parents=True, exist_ok=True)

    return optuna.create_study(
        study_name=study_name,
        storage=JournalStorage(JournalFileBackend(journal_path)) if journal_path else None,
        direction="maximize",
        load_if_exists=True,
    )


def _poll_results(
    queue: MessagesQueue,
    seeds: list[int],
    trial_number: int,
    max_pruned_seeds: int,
    poll_timeout: float,
    poll_interval: float,
) -> list[tuple[float, float]]:
    """Poll the messages queue for results from the given trial until all seeds have reported or too many collapses.

    Arguments:
        queue: The MessagesQueue to poll for results.
        seeds: The list of seeds that were run for this trial (used to track which results are still pending).
        trial_number: The Optuna trial number (used to filter messages for this trial).
        max_pruned_seeds: The maximum number of seed collapses allowed before pruning the trial.
        poll_timeout: The maximum number of seconds to wait for results before giving up.
        poll_interval: The number of seconds to wait between polling attempts.

    Returns:
        A list of (accuracy, kl_loss) tuples for each successful seed.

    """
    results: list[tuple[float, float]] = []
    collapses = 0
    deadline = time.monotonic() + poll_timeout

    while len(results) + collapses < len(seeds):
        if time.monotonic() > deadline:
            logger.warning("[Trial %d] Timed out waiting for results.", trial_number)
            msg = "Timed out waiting for seed results."
            raise TimeoutError(msg)

        messages = queue.pop(max_messages=len(seeds))

        for message in messages:
            if message.get("trial_number") != trial_number:
                continue

            msg_type = message.get("type")
            seed = message.get("seed")

            if msg_type == MessageType.SUCCESS:
                logger.debug(
                    "[Trial %d] seed=%d succeeded: accuracy=%.4f, kl_loss=%.4f",
                    trial_number,
                    seed,
                    message["accuracy"],
                    message["kl_loss"],
                )
                results.append((message["accuracy"], message["kl_loss"]))

            elif msg_type == MessageType.COLLAPSE:
                collapses += 1
                logger.warning("[Trial %d] seed=%d collapsed (%d total).", trial_number, seed, collapses)
                if collapses > max_pruned_seeds:
                    logger.warning("[Trial %d] Too many collapses, pruning trial.", trial_number)
                    msg = f"More than {max_pruned_seeds} seeds collapsed (got {collapses})."
                    raise optuna.TrialPruned(msg)

            elif msg_type == MessageType.ERROR:
                msg = f"Seed {seed} failed with error: {message.get('error', 'Unknown error')}"
                raise RuntimeError(msg)

            else:
                logger.warning("[Trial %d] Unknown message type: %s", trial_number, msg_type)

        if not messages:
            time.sleep(poll_interval)

    return results


def _run_trial(
    trial: optuna.Trial,
    latent_dim: int,
    seeds: list[int],
    settings: LauncherSettings,
    submitter: TrialSubmitter,
    queue: MessagesQueue,
) -> float:
    """Run a single Optuna trial."""
    # lr: log-uniform float
    lr = trial.suggest_float("lr", settings.lr.min, settings.lr.max, log=True)

    # kl_max: uniform float
    kl_max = trial.suggest_float("kl_max", settings.kl_max.min, settings.kl_max.max)

    # batch_size: powers of 2 — sample exponent, then map back
    bs_exp_min = int(math.log2(settings.batch_size.min))
    bs_exp_max = int(math.log2(settings.batch_size.max))
    batch_size = 2 ** trial.suggest_int("batch_size_exp", bs_exp_min, bs_exp_max)

    # conv_channels: multiples of 8 — sample multiplier, then map back
    cc_mult_min = settings.conv_channels.min // 8
    cc_mult_max = settings.conv_channels.max // 8
    conv_channels = 8 * trial.suggest_int("conv_channels_mult", cc_mult_min, cc_mult_max)

    logger.info("[Trial %d] latent_dim=%d, lr=:%.2e, submitting %d jobs...", trial.number, latent_dim, lr, len(seeds))

    # Submit one job per seed
    for seed in seeds:
        trial_settings = TrialSettings(
            study_name=settings.study_name,
            trial_number=trial.number,
            queue_url=queue.url,
            aws_region=settings.aws_region,
            latent_dim=latent_dim,
            seed=seed,
            lr=lr,
            kl_max=kl_max,
            batch_size=batch_size,
            conv_channels=conv_channels,
        )
        job_id = submitter.submit(trial_settings)
        logger.debug("[Trial %d] Submitted job %s for seed=%d", trial.number, seed, job_id)

    results = _poll_results(
        queue,
        seeds,
        trial.number,
        settings.max_pruned_seeds,
        settings.poll_timeout,
        settings.poll_interval,
    )

    accuracies = [r[0] for r in results]
    median_accuracy = statistics.median(accuracies)
    median_kl = statistics.median(r[1] for r in results)

    report_cross_validation_scores(trial, accuracies)

    logger.info("[Trial %d] Done — median accuracy=%.4f, median kl_loss=%.4f", trial.number, median_accuracy, median_kl)

    return median_accuracy


def _run_study(
    latent_dim: int,
    seeds: list[int],
    settings: LauncherSettings,
    submitter: TrialSubmitter,
    queue: MessagesQueue,
) -> float:
    """Run all trials for a given latent_dim. Returns the best accuracy achieved."""
    study_name = f"{settings.study_name}-l{latent_dim}"
    study = _make_study(study_name, settings.journal_path)

    already_done = len([t for t in study.trials if t.state == TrialState.COMPLETE])
    remaining = settings.n_trials - already_done
    if remaining <= 0:
        logger.info("[latent_dim=%d] Study already complete, skipping.", latent_dim)
        return study.best_value

    logger.info("[latent_dim=%d] Starting study '%s' (%d trials remaining).", latent_dim, study_name, remaining)

    def objective(trial: optuna.Trial) -> float:
        return _run_trial(trial, latent_dim, seeds, settings, submitter, queue)

    terminator = Terminator(EMMREvaluator())
    study.optimize(objective, n_trials=remaining, callbacks=[TerminatorCallback(terminator)])

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        logger.warning("[latent_dim=%d] No completed trials.", latent_dim)
        return 0.0

    best = study.best_trial
    logger.info(
        "[latent_dim=%d] Best trial: #%d accuracy=%.4f, params=%s",
        latent_dim,
        best.number,
        best.value,
        best.params,
    )
    return best.value if best.value is not None else 0.0


def run_launcher(settings: LauncherSettings | None = None) -> None:
    """Run the launcher with the given settings (or defaults if None)."""
    settings = LauncherSettings() if settings is None else settings

    submitter = TrialSubmitter(settings.aws_job_queue, settings.aws_job_definition, settings.aws_region)
    queue = MessagesQueue(settings.aws_region)
    queue.create(settings.study_name)

    seeds = _generate_seeds(settings.starter_seed, settings.n_seeds_per_trial)
    logger.info("Generated %d seeds from starter seed %d.", len(seeds), settings.starter_seed)

    try:
        previous_best_accuracy: float | None = None

        for latent_dim in range(settings.latent_dim.min, settings.latent_dim.max + 1):
            best_accuracy = _run_study(latent_dim, seeds, settings, submitter, queue)

            if previous_best_accuracy is not None:
                delta = best_accuracy - previous_best_accuracy
                logger.info(
                    "[latent_dim=%d] Accuracy delta vs previous: %+.4f (threshold: %f)",
                    latent_dim,
                    delta,
                    settings.min_accuracy_delta,
                )

                if delta < settings.min_accuracy_delta:
                    logger.info("[latent_dim=%d] Accuracy did not improve sufficiently. Stopping search.", latent_dim)
                    break

            previous_best_accuracy = best_accuracy
    finally:
        queue.destroy()


if __name__ == "__main__":
    run_launcher()

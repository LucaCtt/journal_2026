import logging
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from csi_vae.messages_queue import MessagesQueue, MessageType
from csi_vae.trial import fusion, vae
from csi_vae.trial.dataset import AntennaDataset, load_datasets
from csi_vae.trial.evaluator import Evaluator
from csi_vae.trial.queue_handler import QueueHandler
from csi_vae.trial.trial_settings import TrialSettings

logger = logging.getLogger(__name__)


def _init_seeds(seed: int) -> None:
    """Initialize random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


def _train_and_eval(settings: TrialSettings) -> float:
    """Train the autoencoder and classifier, then evaluate the accuracy on the test set."""
    torch.set_float32_matmul_precision("high")

    train_ds, test_ds = load_datasets(
        dataset_path=Path(settings.dataset_path),
        window_size=settings.window_size,
        n_activities=settings.n_activities,
        stride=settings.stride,
    )

    gaussians = []
    for antenna_select in range(settings.n_antennas):
        antenna_ds = AntennaDataset(train_ds, antenna_select)
        train_dl = DataLoader(antenna_ds, batch_size=settings.batch_size, shuffle=True)

        gaussian = vae.SingleAntenna(settings.window_size, settings.n_subcarriers, settings.latent_dim)
        trainer = vae.Trainer(gaussian, train_dl, settings.lr, settings.collapse_threshold, settings.collapse_patience)

        try:
            trainer.train(settings.n_epochs)
        except vae.PosteriorCollapseError:
            logger.exception(
                {
                    "study_name": settings.study_name,
                    "trial_id": settings.trial_number,
                    "settings": settings.model_dump(),
                    "status": MessageType.COLLAPSE,
                },
            )
            raise

        gaussians.append(gaussian)

    train_dl = DataLoader(train_ds, batch_size=settings.batch_size, shuffle=True)
    delayed_fusion = fusion.Delayed(gaussians, settings.latent_dim, settings.n_activities)
    trainer = fusion.Trainer(delayed_fusion, train_dl, settings.lr)
    trainer.train(settings.n_epochs)

    test_dl = DataLoader(test_ds, batch_size=settings.batch_size, shuffle=False)
    evaluator = Evaluator(delayed_fusion, test_dl)

    return evaluator.evaluate()


def run_trial(settings: TrialSettings | None = None) -> None:
    """Run a single trial of training and evaluating the autoencoder and classifier."""
    settings = TrialSettings() if settings is None else settings
    _init_seeds(settings.seed)

    if settings.queue_url:
        queue = MessagesQueue.from_url(settings.queue_url, settings.aws_region)
        logger.addHandler(QueueHandler(queue))

    logger.info(
        {
            "study_name": settings.study_name,
            "trial_id": settings.trial_number,
            "settings": settings.model_dump(),
            "status": MessageType.STARTING,
        },
    )

    try:
        accuracy = _train_and_eval(settings)
    except Exception:
        logger.exception(
            {
                "study_name": settings.study_name,
                "trial_id": settings.trial_number,
                "settings": settings.model_dump(),
                "status": MessageType.ERROR,
            },
        )
        raise

    logger.info(
        {
            "study_name": settings.study_name,
            "trial_id": settings.trial_number,
            "settings": settings.model_dump(),
            "accuracy": accuracy,
            "status": MessageType.SUCCESS,
        },
    )


if __name__ == "__main__":
    run_trial()

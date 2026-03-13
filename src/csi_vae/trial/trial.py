import logging
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from csi_vae.messages_queue import MessagesQueue, MessageType
from csi_vae.trial import dataset, fusion, vae
from csi_vae.trial.dataset import AntennaDataset
from csi_vae.trial.evaluator import Evaluator
from csi_vae.trial.queue_handler import QueueHandler
from csi_vae.trial.trial_settings import TrialSettings

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def _init_seeds(seed: int) -> None:
    """Initialize random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


def _make_dataloader(ds: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )


def _train_and_eval(settings: TrialSettings) -> tuple[float, float]:
    """Train the autoencoder and classifier, then evaluate the accuracy on the test set."""
    torch.set_float32_matmul_precision("high")

    train_ds, val_ds, test_ds = dataset.load(
        dataset_path=Path(settings.dataset_path),
        window_size=settings.window_size,
        n_activities=settings.n_activities,
        stride=settings.stride,
    )

    gaussians = []
    total_kl_loss = 0.0

    for antenna_select in range(settings.n_antennas):
        antenna_train_ds = AntennaDataset(train_ds, antenna_select)
        antenna_val_ds = AntennaDataset(val_ds, antenna_select)

        train_dl = _make_dataloader(antenna_train_ds, settings.batch_size, shuffle=True)
        val_dl = _make_dataloader(antenna_val_ds, settings.batch_size, shuffle=False)

        gaussian = vae.SingleAntenna(settings.window_size, settings.n_subcarriers, settings.latent_dim)
        gaussian.compile(fullgraph=True)
        trainer = vae.Trainer(
            gaussian,
            train_dl,
            val_dl,
            settings.lr,
            settings.patience,
            settings.collapse_threshold,
            settings.plateau_min_delta,
        )

        _, _, kl_loss = trainer.train(settings.n_epochs)

        gaussians.append(gaussian)
        total_kl_loss += kl_loss

    train_dl = _make_dataloader(train_ds, settings.batch_size, shuffle=True)
    delayed_fusion = fusion.Delayed(gaussians, settings.latent_dim, settings.n_activities)
    delayed_fusion.compile(fullgraph=True)
    trainer = fusion.Trainer(delayed_fusion, train_dl, settings.lr)
    trainer.train(settings.n_epochs)

    test_dl = _make_dataloader(test_ds, settings.batch_size, shuffle=False)
    evaluator = Evaluator(delayed_fusion, test_dl)

    return evaluator.evaluate(), total_kl_loss / settings.n_antennas


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
        accuracy, kl_loss = _train_and_eval(settings)
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
            "kl_loss": kl_loss,
            "status": MessageType.SUCCESS,
        },
    )


if __name__ == "__main__":
    run_trial()

import logging
from enum import Enum

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from csi_vae.messages_queue import MessagesQueue
from csi_vae.trial.trial_settings import TrialSettings

torch.set_float32_matmul_precision("high")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/tmp/mnist"  # noqa: S108

logger = logging.getLogger(__name__)


class TrialStatus(Enum):
    """Enumeration of possible trial statuses."""

    STARTING = "STARTING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class Autoencoder(nn.Module):
    """Autoencoder network for dimensionality reduction.

    Encodes MNIST images to a latent representation and decodes them back.
    """

    def __init__(self, latent_dim: int) -> None:
        """Initialize the autoencoder.

        Arguments:
            latent_dim : Dimension of the latent representation.

        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the autoencoder.

        Arguments:
            x : Input tensor.

        Returns:
            Reconstructed images and latent representation.

        """
        z = self.encoder(x)
        return self.decoder(z), z


class Classifier(nn.Module):
    """Simple MLP classifier on the latent space."""

    def __init__(self, latent_dim: int, hidden: int) -> None:
        """Initialize the classifier.

        Arguments:
            latent_dim : Dimension of the latent representation.
            hidden : Dimension of the hidden layer.

        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classifier.

        Arguments:
            z : Latent representation tensor.

        Returns:
            Logits for the 10 classes.

        """
        return self.net(z)


def train_autoencoder(
    model: Autoencoder,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    epochs: int,
) -> None:
    """Train the autoencoder.

    Arguments:
        model : The autoencoder model to train.
        loader : DataLoader for training data.
        optimizer : Optimizer for updating model parameters.
        epochs : Number of training epochs.

    """
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)  # noqa: PLW2901
            recon, _ = model(batch_x.to(DEVICE))
            loss = criterion(recon, batch_x.view(batch_x.size(0), -1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train_classifier(
    ae: Autoencoder,
    clf: Classifier,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    epochs: int,
) -> None:
    """Train the classifier on encoded representations.

    Arguments:
        ae : Trained autoencoder for encoding data.
        clf : Classifier to train.
        loader : DataLoader for training data.
        optimizer : Optimizer for updating classifier parameters.
        epochs : Number of training epochs.

    """
    criterion = nn.CrossEntropyLoss()
    ae.eval()
    clf.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)  # noqa: PLW2901
            with torch.no_grad():
                z = ae.encoder(batch_x)
            logits = clf(z)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def evaluate(
    ae: Autoencoder,
    clf: Classifier,
    loader: DataLoader,
) -> float:
    """Evaluate the classifier on encoded representations.

    Arguments:
        ae : Trained autoencoder for encoding data.
        clf : Trained classifier.
        loader : DataLoader for evaluation data.

    Returns:
        Accuracy of the classifier on the data.

    """
    ae.eval()
    clf.eval()
    correct = total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)  # noqa: PLW2901
            z = ae.encoder(batch_x)
            preds = clf(z).argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    return correct / total


def run_trial(settings: TrialSettings | None = None) -> None:
    """Run a single trial of training and evaluating the autoencoder and classifier."""
    settings = TrialSettings() if settings is None else settings
    queue = MessagesQueue.from_url(settings.queue_url) if settings.queue_url else None

    logger.info("Starting trial #%d with params: %s", settings.trial_number, settings.model_dump())
    if queue:
        queue.push(
            {
                "study_name": settings.study_name,
                "trial_id": settings.trial_number,
                "params": settings.model_dump(),
                "status": TrialStatus.STARTING,
            },
        )

    try:
        logger.debug("Loading dataset...")
        tf = transforms.ToTensor()
        train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
        test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

        logger.debug("Initializing models...")
        ae = Autoencoder(128).to(DEVICE)
        ae.compile()
        clf = Classifier(128, 128).to(DEVICE)
        clf.compile()

        logger.debug("Training autoencoder and classifier...")
        train_autoencoder(ae, train_loader, optim.Adam(ae.parameters(), lr=settings.param_lr), 10)
        train_classifier(ae, clf, train_loader, optim.Adam(clf.parameters(), lr=settings.param_lr), 10)

        accuracy = evaluate(ae, clf, test_loader)
    except Exception as e:
        logger.exception("Trial #%d failed.", settings.trial_number)
        if queue:
            queue.push(
                {
                    "study_name": settings.study_name,
                    "trial_id": settings.trial_number,
                    "params": settings.model_dump(),
                    "status": TrialStatus.FAILED,
                    "error": str(e),
                },
            )
        raise

    logger.info("Trial #%d completed with accuracy: %.4f", settings.trial_number, accuracy)
    if queue:
        queue.push(
            {
                "study_name": settings.study_name,
                "trial_id": settings.trial_number,
                "params": settings.model_dump(),
                "accuracy": accuracy,
                "status": TrialStatus.SUCCEEDED,
            },
        )


if __name__ == "__main__":
    run_trial()

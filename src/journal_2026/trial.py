import os

import optuna
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Hyperparams come from env vars set by the launcher
STORAGE = os.environ["OPTUNA_STORAGE"]
STUDY_NAME = os.environ["OPTUNA_STUDY"]
TRIAL_ID = int(os.environ["OPTUNA_TRIAL_ID"])

DEVICE = torch.device("cuda")
DATA_DIR = "/tmp/mnist"  # noqa: S108


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


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function to optimize."""
    latent_dim = trial.suggest_int("latent_dim", 8, 128)
    ae_lr = trial.suggest_float("ae_lr", 1e-4, 1e-2, log=True)
    clf_lr = trial.suggest_float("clf_lr", 1e-4, 1e-2, log=True)
    clf_hidden = trial.suggest_categorical("clf_hidden", [64, 128, 256])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    tf = transforms.ToTensor()
    train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    ae = Autoencoder(latent_dim).to(DEVICE)
    clf = Classifier(latent_dim, clf_hidden).to(DEVICE)

    train_autoencoder(ae, train_loader, optim.Adam(ae.parameters(), lr=ae_lr), 10)
    train_classifier(ae, clf, train_loader, optim.Adam(clf.parameters(), lr=clf_lr), 10)

    return evaluate(ae, clf, test_loader)


if __name__ == "__main__":
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    # Tell Optuna to run exactly the trial that was pre-created by the launcher
    study.optimize(objective, n_trials=1)

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from csi_vae.trial import classifier, vae


class Trainer:
    """Trainer class for classifier model using Distributed Data Parallel (DDP)."""

    def __init__(
        self,
        clf: classifier.BasicNN,
        gaussian: vae.SingleAntenna,
        dataloader: DataLoader,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the Classifier Trainer.

        Arguments:
            clf: The classifier model to be trained.
            gaussian: The pre-trained VAE model used for feature extraction.
            dataloader: DataLoader for the training dataset.
            device: The device to use for training.

        """
        self.__device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.__clf = clf.to(self.__device)
        self.__gaussian = gaussian.to(self.__device)
        self.__dataloader = dataloader

        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = optim.Adam(self.__clf.parameters(), lr=1e-3)
        self.__scaler = torch.GradScaler(device=self.__device.type)

    def __run_batch(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a single training batch."""
        self.__optimizer.zero_grad()

        with torch.autocast(device_type=self.__device.type, dtype=torch.bfloat16):
            with torch.no_grad():
                mu, logvar = self.__gaussian.encode(x)

            logits = self.__clf(torch.cat([mu, logvar], dim=1))
            loss = self.__criterion(logits, y)

        self.__scaler.scale(loss).backward()
        self.__scaler.step(self.__optimizer)
        self.__scaler.update()

        with torch.no_grad():
            accuracy = (logits.argmax(dim=1) == y).float().mean()

        return loss.detach(), accuracy.detach()

    def __run_epoch(self) -> tuple[torch.Tensor, torch.Tensor]:
        metrics = torch.zeros(2, device=self.__device)

        for x, y in self.__dataloader:
            loss, accuracy = self.__run_batch(x.to(self.__device), y.to(self.__device))

            metrics[0] += loss
            metrics[1] += accuracy

        metrics /= len(self.__dataloader)

        return metrics[0], metrics[1]

    def train(self, epochs: int) -> tuple[float, float]:
        """Train the classifier model for a specified number of epochs."""
        self.__clf.train()
        self.__gaussian.eval()

        total_metrics = torch.zeros(2, device=self.__device)

        for _ in range(epochs):
            epoch_loss, epoch_accuracy = self.__run_epoch()
            total_metrics += torch.tensor([epoch_loss, epoch_accuracy], device=self.__device)

        total_metrics /= epochs

        return tuple(total_metrics.tolist())

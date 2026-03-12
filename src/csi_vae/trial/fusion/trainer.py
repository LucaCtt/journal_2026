import torch
from torch import nn
from torch.utils.data import DataLoader

from csi_vae.trial import fusion


class Trainer:
    """Trainer class for DelayedFusion model."""

    def __init__(
        self,
        model: fusion.Delayed,
        dataloader: DataLoader,
        lr: float,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the Trainer.

        Arguments:
            model: DelayedFusion model to be trained.
            dataloader: DataLoader for training data.
            lr: Learning rate for the optimizer.
            device: Device to train the model on.

        """
        self.__device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__model = model.to(self.__device)
        self.__dataloader = dataloader
        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr=lr)
        self.__scaler = torch.GradScaler(device=self.__device.type)

    def __run_batch(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a single training batch.

        Arguments:
            x: Input tensor of shape (batch_size, num_antennas, num_features).
            y: Ground truth labels of shape (batch_size,).

        Returns:
            Tuple containing loss and accuracy.

        """
        self.__optimizer.zero_grad()

        with torch.autocast(device_type=self.__device.type, dtype=torch.bfloat16):
            logits = self.__model(x)
            loss = self.__criterion(logits, y)

        self.__scaler.scale(loss).backward()
        self.__scaler.step(self.__optimizer)
        self.__scaler.update()

        with torch.no_grad():
            accuracy = (logits.argmax(dim=1) == y).float().mean()

        return loss.detach(), accuracy.detach()

    def __run_epoch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a single training epoch.

        Returns:
            Tuple containing average loss and accuracy for the epoch.

        """
        metrics = torch.zeros(2, device=self.__device)

        for x, y in self.__dataloader:
            loss, accuracy = self.__run_batch(x.to(self.__device), y.to(self.__device))

            metrics[0] += loss
            metrics[1] += accuracy

        metrics /= len(self.__dataloader)
        return metrics[0], metrics[1]

    def train(self, epochs: int) -> tuple[float, float]:
        """Train the DelayedFusion model for a specified number of epochs.

        Arguments:
            epochs: Number of epochs to train.

        Returns:
            Tuple containing average loss and accuracy over all epochs.

        """
        self.__model.train()

        total_metrics = torch.zeros(2, device=self.__device)

        for _ in range(epochs):
            epoch_loss, epoch_accuracy = self.__run_epoch()
            total_metrics[0] += epoch_loss
            total_metrics[1] += epoch_accuracy

        total_metrics /= epochs
        return tuple(total_metrics.tolist())

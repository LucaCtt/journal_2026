import torch
from torch.utils.data import DataLoader

from csi_vae.trial import vae


class PosteriorCollapseError(Exception):
    """Raised when the VAE posterior collapses during training."""


class Trainer:
    """Trainer class for VAE model."""

    def __init__(
        self,
        gaussian: vae.SingleAntenna,
        dataloader: DataLoader,
        lr: float,
        collapse_threshold: float,
        collapse_patience: int,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the Trainer.

        Arguments:
            gaussian: VAE model to be trained.
            dataloader: DataLoader for training data.
            lr: Learning rate for the optimizer.
            collapse_threshold: Threshold for KL divergence loss to detect posterior collapse.
            collapse_patience: Number of epochs to wait before raising a collapse error.
            device: Device to train the model on.

        """
        self.__device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.__gaussian = gaussian.to(self.__device)
        self.__dataloader = dataloader
        self.__optimizer = torch.optim.Adam(self.__gaussian.parameters(), lr=lr)
        self.__scaler = torch.GradScaler(device=self.__device.type)
        self.__collapse_threshold = collapse_threshold
        self.__collapse_patience = collapse_patience

    def __run_batch(
        self,
        x_true: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a single training batch.

        Arguments:
            x_true: Ground truth input tensor.

        Returns:
            Tuple containing total loss, reconstruction loss, KL divergence loss.

        """
        self.__optimizer.zero_grad()

        # Autocast for mixed precision training
        with torch.autocast(device_type=self.__device.type, dtype=torch.float16):
            x_recon, mu, logvar = self.__gaussian(x_true)
            loss, recon_loss, kl_loss = vae.loss(x_recon, x_true, mu, logvar)

        self.__scaler.scale(loss).backward()
        self.__scaler.step(self.__optimizer)
        self.__scaler.update()

        return loss, recon_loss, kl_loss

    def __run_epoch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a single training epoch.

        Arguments:
            epoch: Current epoch number.

        Returns:
            Tuple containing average total loss, reconstruction loss,
            KL divergence loss, and average latent entropy for the epoch.

        """
        metrics = torch.zeros(3, device=self.__device)

        for x_true, _ in self.__dataloader:
            loss, recon_loss, kl_loss = self.__run_batch(x_true.to(self.__device))

            metrics[0] += loss.detach()
            metrics[1] += recon_loss.detach()
            metrics[2] += kl_loss.detach()

        # Average the loss and other metrics
        metrics = metrics / len(self.__dataloader)

        return metrics[0], metrics[1], metrics[2]

    def train(self, epochs: int) -> tuple[float, float, float]:
        """Train the VAE model for a specified number of epochs.

        Arguments:
            epochs: Number of epochs to train.

        Returns:
            Tuple containing average total loss, reconstruction loss,
            and KL divergence loss over all epochs.

        """
        self.__gaussian.train()

        total_metrics = torch.zeros(3, device=self.__device)
        collapse_counter = 0

        for _ in range(epochs):
            epoch_loss, epoch_recon_loss, epoch_kl_loss = self.__run_epoch()

            if epoch_kl_loss < self.__collapse_threshold:
                collapse_counter += 1
                if collapse_counter >= self.__collapse_patience:
                    msg = f"Posterior collapse detected: KL loss {epoch_kl_loss:.6f} \
                        below threshold {self.__collapse_threshold}"
                    raise PosteriorCollapseError(msg)
            else:
                collapse_counter = 0

            # Distributed averaging of metrics
            total_metrics[0] += epoch_loss
            total_metrics[1] += epoch_recon_loss
            total_metrics[2] += epoch_kl_loss

        total_metrics /= epochs

        return tuple(total_metrics.tolist())

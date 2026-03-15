import torch
from torch.utils.data import DataLoader

from csi_vae.trial import vae
from csi_vae.trial.early_stopping import EarlyStopping
from csi_vae.trial.vae.kl_annealer import KLAnnealer


def _is_collapsed(kl_history: list[float], threshold: float) -> bool:
    """Determine if the VAE has collapsed based on recent KL loss history."""
    return all(kl < threshold for kl in kl_history) or all(
        abs(kl_history[i] - kl_history[i - 1]) < threshold for i in range(1, len(kl_history))
    )


class PosteriorCollapseError(Exception):
    """Raised when the VAE posterior collapses during training."""

    def __init__(self) -> None:
        """Initialize the error with a default message."""
        super().__init__("Posterior collapse detected.")


class Trainer:
    """Trainer class for VAE model."""

    def __init__(
        self,
        gaussian: vae.SingleAntenna,
        train_dl: DataLoader,
        val_dl: DataLoader,
        lr: float,
        patience: int,
        warmup_epochs: int,
        collapse_threshold: float,
        plateau_min_delta: float,
        kl_max: float,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the Trainer with model, data, and training parameters.

        Arguments:
            gaussian: VAE model to be trained.
            train_dl: DataLoader for training data.
            val_dl: DataLoader for validation data.
            lr: Learning rate for the optimizer.
            patience: Patience for both early stopping and collapse detection.
            warmup_epochs: Number of epochs to warm up the learning rate.
            collapse_threshold: KL loss below this triggers collapse detection.
            plateau_min_delta: Minimum improvement in val loss to reset early stopping.
            kl_max: Maximum KL divergence weight.
            device: Target device; defaults to CUDA if available.

        """
        self.__device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__gaussian = gaussian.to(self.__device)
        self.__train_dl = train_dl
        self.__val_dl = val_dl
        self.__optimizer = torch.optim.Adam(self.__gaussian.parameters(), lr=lr)
        self.__scaler = torch.GradScaler(device=self.__device.type)
        self.__collapse_threshold = collapse_threshold
        self.__plateau_min_delta = plateau_min_delta
        self.__kl_max = kl_max
        self.__patience = patience
        self.__early_stopping = EarlyStopping(self.__gaussian, patience, warmup_epochs)

    def __run_batch(self, x_true: torch.Tensor, kl_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.__optimizer.zero_grad()

        with torch.autocast(device_type=self.__device.type, dtype=torch.float16):
            x_recon, mu, logvar = self.__gaussian(x_true)
            loss, recon_loss, kl_loss = vae.loss(x_recon, x_true, mu, logvar, kl_weight)

        self.__scaler.scale(loss).backward()
        self.__scaler.step(self.__optimizer)
        self.__scaler.update()

        return loss, recon_loss, kl_loss

    def __run_epoch(self, kl_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.__gaussian.train()

        metrics = torch.zeros(3, device=self.__device)

        for x_true, _ in self.__train_dl:
            loss, recon_loss, kl_loss = self.__run_batch(x_true.to(self.__device), kl_weight)
            metrics[0] += loss.detach()
            metrics[1] += recon_loss.detach()
            metrics[2] += kl_loss.detach()

        metrics /= len(self.__train_dl)

        return metrics[0], metrics[1], metrics[2]

    @torch.no_grad()
    def __run_val_epoch(self, kl_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.__gaussian.eval()

        metrics = torch.zeros(3, device=self.__device)

        for x_true_cpu, _ in self.__val_dl:
            x_true = x_true_cpu.to(self.__device)

            with torch.autocast(device_type=self.__device.type, dtype=torch.float16):
                x_recon, mu, logvar = self.__gaussian(x_true)
                loss, recon_loss, kl_loss = vae.loss(x_recon, x_true, mu, logvar, kl_weight)

            metrics[0] += loss.detach()
            metrics[1] += recon_loss.detach()
            metrics[2] += kl_loss.detach()

        metrics /= len(self.__val_dl)

        return metrics[0], metrics[1], metrics[2]

    def train(self, epochs: int) -> tuple[float, float, float]:
        """Train the VAE model for a specified number of epochs.

        Arguments:
            epochs: Maximum number of epochs to train.

        Returns:
            Tuple of (average total loss, average recon loss, average KL loss).

        """
        total_metrics = torch.zeros(3, device=self.__device)
        kl_history = []
        epochs_run = 0
        annealer = KLAnnealer(epochs, kl_max=self.__kl_max)

        for _ in range(epochs):
            epoch_loss, epoch_recon_loss, epoch_kl_loss = self.__run_epoch(annealer.weight)
            val_loss, _, _ = self.__run_val_epoch(annealer.weight)
            annealer.step()

            # Accumulate before any early-exit so counts stay consistent
            total_metrics[0] += epoch_loss
            total_metrics[1] += epoch_recon_loss
            total_metrics[2] += epoch_kl_loss
            epochs_run += 1
            kl_history.append(epoch_kl_loss)
            kl_history = kl_history[-self.__patience :]

            if annealer.weight >= 1.0 and _is_collapsed(kl_history, self.__collapse_threshold):
                raise PosteriorCollapseError

            self.__early_stopping.step_loss(val_loss, delta=self.__plateau_min_delta)
            if self.__early_stopping.should_stop:
                break

        self.__early_stopping.restore_best_weights()

        loss, recon_loss, kl_loss = (total_metrics / epochs_run).tolist()
        return loss, recon_loss, kl_loss

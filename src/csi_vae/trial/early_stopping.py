import io

import torch
from torch import nn

Metric = float | torch.Tensor


class EarlyStopping:
    """Tracks a validation metric and saves/restores best model weights.

    Supports accuracy-based (higher is better) and loss-based
    (lower is better, with optional min delta) stepping.

    Raises:
        RuntimeError: If restore_best_weights is called before any improvement
            has been recorded.

    """

    def __init__(self, model: nn.Module, patience: int) -> None:
        """Initialize the EarlyStopping instance.

        Arguments:
            model: Model whose weights are tracked.
            patience: Epochs without improvement before early stopping triggers.

        """
        self.__model = model
        self.__patience = patience
        self.__best_accuracy: torch.Tensor = torch.tensor(float("-inf"))
        self.__best_loss: torch.Tensor = torch.tensor(float("inf"))
        self.__plateau_counter = 0
        self.__best_weights: bytes | None = None

    @property
    def should_stop(self) -> bool:
        """Whether training should stop due to lack of improvement."""
        return self.__plateau_counter >= self.__patience

    def step(self, val_accuracy: Metric) -> None:
        """Step using accuracy (higher is better).

        Arguments:
            val_accuracy: Validation accuracy from the most recent epoch.

        """
        val_accuracy = torch.as_tensor(val_accuracy)
        improved = val_accuracy > self.__best_accuracy
        self.__update(improved)
        if improved:
            self.__best_accuracy = val_accuracy

    def step_loss(self, val_loss: Metric, delta: float = 0.0) -> None:
        """Step using loss (lower is better).

        Arguments:
            val_loss: Validation loss from the most recent epoch.
            delta: Minimum improvement required to reset the plateau counter.

        """
        val_loss = torch.as_tensor(val_loss)
        improved = val_loss < self.__best_loss - delta
        self.__update(improved)
        if improved:
            self.__best_loss = val_loss

    def __update(self, improved: torch.Tensor) -> None:
        if improved:
            self.__plateau_counter = 0
            buf = io.BytesIO()
            torch.save(self.__model.state_dict(), buf)
            self.__best_weights = buf.getvalue()
        else:
            self.__plateau_counter += 1

    def restore_best_weights(self) -> None:
        """Load the best recorded weights back into the model."""
        if self.__best_weights is None:
            msg = "No checkpoint saved; restore called before any step."
            raise RuntimeError(msg)

        self.__model.load_state_dict(torch.load(io.BytesIO(self.__best_weights), weights_only=True))

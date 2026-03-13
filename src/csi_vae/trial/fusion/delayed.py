import torch
from torch import nn

from csi_vae.trial.vae.gaussian import SingleAntenna


class Delayed(nn.Module):
    """Delayed fusion module for multi-antenna CSI data."""

    def __init__(self, antennas: list[SingleAntenna], latent_dim: int, n_activities: int) -> None:
        """Initialize the delayed fusion module."""
        super().__init__()

        self.__antennas = nn.ModuleList(antennas)
        for param in self.__antennas.parameters():
            param.requires_grad = False

        self.__fc = nn.Sequential(
            nn.Linear(latent_dim * 2 * len(antennas), 32),
            nn.GELU(),
            nn.Linear(32, n_activities),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the delayed fusion module.

        Arguments:
            x: Input tensor of shape (batch_size, num_antennas, num_features).

        Returns:
            Output tensor of shape (batch_size, n_activities).

        """
        z = []

        for i, antenna in enumerate(self.__antennas):
            _, mu_i, logvar_i = antenna(x[:, i])
            z.append(torch.cat((mu_i, logvar_i), dim=1))

        z = torch.cat(z, dim=1)
        return self.__fc(z)

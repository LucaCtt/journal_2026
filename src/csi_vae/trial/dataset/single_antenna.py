import torch
from torch.utils.data import Dataset

from csi_vae.trial.dataset.multi_antenna import MultiAntenna


class SingleAntenna(Dataset):
    """Dataset for selecting a single antenna from the CSI data.

    This class wraps around a CSIDataset and selects the specified antenna from the CSI data,
    to avoid loading the whole dataset multiple times when training/evaluating models on different antennas.
    """

    def __init__(self, dataset: MultiAntenna, antenna_select: int) -> None:
        """Initialize the AntennaDataset."""
        self.__dataset = dataset
        self.__antenna_select = antenna_select

    def __len__(self) -> int:
        return len(self.__dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x, y = self.__dataset[idx]
        x = x[self.__antenna_select, :, :]
        return x, y

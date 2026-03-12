import numpy as np
import torch
from torch.utils.data import Dataset


class CSIDataset(Dataset):
    """CSI Dataset for PyTorch."""

    def __init__(
        self,
        csi_mats: list[np.ndarray],
        window_size: int,
        stride: int,
    ) -> None:
        """Initialize the CSI dataset.

        Arguments:
            csi_mats: List of CSI matrices, one for each activity. Each matrix should have shape
                [n_antennas, n_samples, n_subcarriers].
            window_size: Size of the sliding window to extract from each sample.
            stride: Stride of the sliding window.

        """
        self.__window_size = window_size

        self.__data: list[np.ndarray] = []
        self.__labels: list[int] = []
        self.__index_map: list[tuple[int, int]] = []

        # Load files once, build index map
        for label, csi in enumerate(csi_mats):
            start = len(self.__data)
            self.__data.append(csi)
            self.__labels.append(label)

            # Build lazy sliding-window index
            for offset in range(0, csi.shape[1] - window_size + 1, stride):
                self.__index_map.append((start, offset))

    def __len__(self) -> int:
        return len(self.__index_map)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        file_id, start = self.__index_map[idx]
        csi = self.__data[file_id]

        window = csi[:, start : start + self.__window_size, :]

        x = torch.from_numpy(window)
        y = self.__labels[file_id]

        return x, y

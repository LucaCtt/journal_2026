from pathlib import Path
from string import ascii_uppercase
from typing import cast

import h5py
import numpy as np

from csi_vae.trial.dataset.csi_dataset import CSIDataset


def load(
    dataset_path: Path,
    window_size: int,
    n_activities: int,
    stride: int,
) -> tuple[CSIDataset, CSIDataset, CSIDataset]:
    """Load the CSI train/test datasets.

    Arguments:
        dataset_path: Path to the dataset directory.
        window_size: Window size for CSI samples.
        n_activities: Number of activities (files) to load from the dataset.
        stride: Stride of the sliding window.

    Returns:
        A tuple containing the train, validation, and test datasets.

    """
    train_mats = []
    val_mats = []
    test_mats = []
    with h5py.File(dataset_path, "r") as f:
        train_group = cast("h5py.Group", f["train"])
        val_group = cast("h5py.Group", f["val"])
        test_group = cast("h5py.Group", f["test"])

        for i in range(n_activities):
            activity_key = ascii_uppercase[i]

            train_mats.append(np.array(train_group[activity_key]))
            val_mats.append(np.array(val_group[activity_key]))
            test_mats.append(np.array(test_group[activity_key]))

    # Shape of dataset samples: (n_antennas, window_size, n_subcarriers)
    train_dataset = CSIDataset(csi_mats=train_mats, window_size=window_size, stride=stride)
    val_dataset = CSIDataset(csi_mats=val_mats, window_size=window_size, stride=stride)
    test_dataset = CSIDataset(csi_mats=test_mats, window_size=window_size, stride=stride)

    return train_dataset, val_dataset, test_dataset

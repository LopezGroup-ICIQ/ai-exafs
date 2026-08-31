"""
Copyright (c) 2026 - ICIQ (Institute of Chemical Research of Catalonia)

Developed at ICIQ and funded by NCCR Catalysis.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pickle
import lmdb

import numpy as np

import torch
from torch.utils.data import random_split
from torch_geometric.data import InMemoryDataset

from ai_exafs.preprocessing.normalization import (
    ElementScalers,
    NormalizeTargetByElement,
)


class LmdbGraphDataset(InMemoryDataset):
    """
    PyG Dataset class to load LMDB file containing graph representations.

    Args:
        root    : Where the processed folder is going to be created.
        db_path : Where the lmdb file is stored.

    Returns:
        PyG Dataset object to be converted into DataLoader.
    """

    def __init__(self, root: str, db_path: str):
        self.root = root
        self.db_path = db_path
        self.db = self._connect_db()
        self.length = self._get_additional_info(info="length")
        self.norm_meta = self._get_additional_info(
            info="normalization", default=None
        )

        super().__init__(root)
        self.data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False, map_location="cpu"
        )

    def _connect_db(self):
        db = lmdb.open(
            str(self.db_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        return db

    def _get_additional_info(self, info, default=...):
        """To retrieve specific information from the lmdb file"""
        value = self.db.begin().get(str(info).encode("ascii"))
        if value is None:
            if default is ...:
                raise KeyError(f"'{info}' not found in {self.db_path}")
            return default
        return pickle.loads(value)

    @property
    def raw_file_names(self):
        return self.db_path

    @property
    def processed_file_names(self):
        return "data.pt"

    def process(self):
        """Method to build list of data objects, collate and save"""
        # Get data objects from lmdb
        data_list = []
        for idx in range(self.length):
            with self.db.begin() as txn:
                data_compress = txn.get(f"{idx}".encode("ascii"))
                graph = pickle.loads(data_compress)
                # Append
                data_list.append(graph)

        # Collate and save
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def get_dataset_splits(
    dataset,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
):
    """Helper function to split PyG dataset into train/val/testing"""
    dataset_length = len(dataset)
    val_split = int(np.ceil(val_ratio * dataset_length))
    test_split = int(np.ceil(test_ratio * dataset_length))
    train_split = int(np.floor(dataset_length - (val_split + test_split)))
    train_data, val_data, test_data = random_split(
        dataset,
        [train_split, val_split, test_split],
        generator=torch.Generator().manual_seed(random_state),
    )
    return train_data, val_data, test_data


def fit_scalers_on_train(
    dataset,
    train_data,
    lo: float = 0.1,
    hi: float = 0.9,
):
    """
    Fit the Min-Max scalers on the training split only and attach
    them to the dataset as transform.

    The transform lives on the parent dataset, so the very same
    statistics are applied to the training, validation and test subsets,
    and later to experimental spectra at inference time.

    Args:
        dataset   : The full LmdbGraphDataset (raw spectra)
        train_data: The training Subset returned by get_dataset_splits.
        lo, hi    : Normalized range of the training data. The margins
        [0, lo) and (hi, 1] are the headroom left for spectra exceeding
        the training amplitude range.

    Returns:
        The fitted ElementScalers.
    """
    if dataset.transform is not None:
        raise RuntimeError(
            "dataset.transform is already set: the scalers must be fitted "
            "on the raw training targets."
        )
    scalers = ElementScalers(lo=lo, hi=hi).fit_from_subset(train_data)
    dataset.transform = NormalizeTargetByElement(scalers)
    return scalers

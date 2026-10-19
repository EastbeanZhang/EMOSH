# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)
import lmdb
import numpy as np
import torch
from torchvision.io import ImageReadMode, decode_image


class LMDBEngine:
    def __init__(self, lmdb_path, write=False):
        if write:
            raise ValueError("Use prepare_dataset.py to create training records.")
        self.environment = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)

    def __getitem__(self, key):
        with self.environment.begin() as transaction:
            payload = transaction.get(key.encode())
        if payload is None:
            raise KeyError(key)
        image = torch.from_numpy(np.frombuffer(payload, dtype=np.uint8).copy())
        return decode_image(image, mode=ImageReadMode.RGB)

    def close(self):
        if self.environment is not None:
            self.environment.close()
            self.environment = None

    def __del__(self):
        if hasattr(self, "environment"):
            self.close()

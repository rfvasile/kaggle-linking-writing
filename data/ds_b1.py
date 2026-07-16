from types import SimpleNamespace
from typing import Literal

from pandas import DataFrame
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, df: DataFrame, cfg: SimpleNamespace, mode: Literal["train", "val"]):
        self.df = df
        self.cfg = cfg
        self.mode = mode

    def __len__(self):
        return len(self.df)


def tr_collate_fn():
    pass


def val_collate_fn():
    pass


def batch_to_device():
    pass

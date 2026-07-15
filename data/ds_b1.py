from types import SimpleNamespace

from pandas import DataFrame


class CustomDataset:
    def __init__(self, df: DataFrame, cfg: SimpleNamespace, mode: str):
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

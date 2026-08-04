# %%
from types import SimpleNamespace
from typing import Any, Literal

import torch
from pandas import DataFrame
from torch import tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

ins = None
item = None
g_out = None


def collate_fn(batch: list[Any]) -> dict[str, Any]:
    """Produces a batch of items in the required training format.

    Args:
        batch: the items produced by __getitem__ from the dataset.

    Returns:
        A batch of items of dim: BxTxF.
    """
    global ins
    ins = batch

    out_dict = {
        "input": pad_sequence(
            [b["input"] for b in batch], batch_first=True
        ),  # output: BxTx*, where T is the length of the longest sequence
        "attention_mask": pad_sequence([b["attention_mask"] for b in batch], batch_first=True),
        "idx": torch.stack([b["idx"] for b in batch]),
    }

    if "target" in batch[0].keys():
        # Produces a tensor of an array
        out_dict.update({"target": torch.stack([b["target"] for b in batch])})

    return out_dict


tr_collate_fn = collate_fn
val_collate_fn = collate_fn


def batch_to_device(batch: dict[str, Any], device: str):
    """Moves data to a GPU device."""
    out = {key: val.to(device) for key, val in batch.items()}
    return out


class CustomDataset(Dataset):
    def __init__(self, df: DataFrame, cfg: SimpleNamespace, mode: Literal["train", "val"]):
        self.cfg = cfg
        self.mode = mode
        # Preserve temporal order of the events via a "stable" sort. This means that .loc resolves
        # to a slice instead of scanning the whole dataset when checking temporal order:
        # O(log N) vs O(N))
        self.df = df.set_index("id", drop=False).sort_index(kind="stable")

        # Working with temporal sequences, so the order must be monotone
        assert self.df.groupby(level="id")["event_id"].diff().dropna().gt(0).all(), "events out of order"

        self.ids = self.df["id"].unique()
        self.indices = self.df.index.unique()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Prepares a specific dataset item for training.

        Args:
            index: The index for a specific item.

        Returns:
            Outputs a dictionary, prepared for the collate function.
        """
        global item

        # The index represents a specific essay/user, so the  dimension of the retrieved
        # items is TxF, where T is the time dimension, while F are the train features.
        item = self.df.loc[self.indices[index]]

        feats = item[["action_time", "cursor_position", "up_time"]]
        g_out = {
            "input": tensor(feats.to_numpy(dtype="float32")),
            "attention_mask": torch.ones([len(feats)]),
            "idx": tensor(item["idx"].iloc[0]),  # useful for oof[batch["idx"]] = preds
        }

        if "score" in item:
            # The score is per essay, so it is constant across multiple event
            g_out.update({"target": torch.tensor(item["score"].iloc[0], dtype=torch.float32)})

        return g_out


# TODO
# import pandas
# from torch.utils.data import DataLoader

# df = pandas.read_parquet("datamount/train_folds.parquet")
# cust_ds = CustomDataset(df, SimpleNamespace(**{}), "train")
# data_loader = DataLoader(cust_ds, batch_size=5, collate_fn=tr_collate_fn)
# it = iter(data_loader)
# next(it)

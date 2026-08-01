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
        A batch of items of dim. BxFxN.
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
    out = {batch[key].to(device) for key in ["input", "idx", "target", "attention_mask"]}
    return out


class CustomDataset(Dataset):
    def __init__(self, df: DataFrame, cfg: SimpleNamespace, mode: Literal["train", "val"]):
        self.df = df
        self.cfg = cfg
        self.mode = mode
        self.indices = df["idx"].unique()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Prepares a specific dataset item for training.

        Args:
            index: The index for a specific item.

        Returns:
            Outputs a dictionary with collate function fields.
        """
        global item
        item = self.df.loc[self.indices[index]]
        feats = item[
            [key for key in item.keys() if key not in ["id", "score", "idx"] and not isinstance(item[key], str)]
        ]
        g_out = {"input": tensor(feats), "attention_mask": tensor([2]), "idx": tensor([1])}

        if "score" in item:
            g_out.update({"target": torch.tensor(item["score"])})

        return g_out


# TODO
# df = pandas.read_parquet("datamount/train_folds.parquet")
# cust_ds = CustomDataset(df, SimpleNamespace(**{}), "train")
# data_loader = DataLoader(cust_ds, batch_size=5, collate_fn=tr_collate_fn)
# it = iter(data_loader)
# next(it)

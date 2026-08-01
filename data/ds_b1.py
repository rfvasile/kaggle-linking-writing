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
        # Indexed by essay so that .loc gathers all the keystroke events of one essay
        self.df = df.set_index("idx")
        self.cfg = cfg
        self.mode = mode
        self.indices = self.df.index.unique()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Prepares a specific dataset item for training.

        Args:
            index: The index for a specific item.

        Returns:
            Outputs a dictionary with collate function fields.
        """
        global item
        item = self.df.loc[self.indices[index]]  # TxF: one row/event
        feats = item.select_dtypes("number").drop(columns=["score", "fold"], errors="ignore")
        g_out = {"input": tensor(feats.to_numpy(dtype="float32")), "attention_mask": tensor([2]), "idx": tensor([1])}

        if "score" in item:
            # The score is per essay, so it is constant across multiple event
            g_out.update({"target": torch.tensor(item["score"].iloc[0])})

        return g_out


# TODO
# df = pandas.read_parquet("datamount/train_folds.parquet")
# cust_ds = CustomDataset(df, SimpleNamespace(**{}), "train")
# data_loader = DataLoader(cust_ds, batch_size=5, collate_fn=tr_collate_fn)
# it = iter(data_loader)
# next(it)

# %%
from types import SimpleNamespace
from typing import Any, Literal

import torch
from pandas import DataFrame
from torch import tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer, DebertaV2TokenizerFast

ins = None
item = None
g_out = None


model_id = "microsoft/deberta-v3-base"
tokenizer: DebertaV2TokenizerFast = AutoTokenizer.from_pretrained(model_id)


def replay_with_owner(events: DataFrame) -> tuple[str, list[int]]:
    """Replay the log, carrying a parallel per-character owner array."""
    text = ""
    owner: list[int] = []

    cols = events[["activity", "cursor_position", "text_change"]]
    for i, (activity, cursor, change) in enumerate(cols.itertuples(index=False)):
        if activity == "Replace":
            old, new = change.split(" => ")
            start = cursor - len(new)
            text = text[:start] + new + text[start + len(old) :]
            owner = owner[:start] + [i] * len(new) + owner[start + len(old) :]
        elif activity == "Remove/Cut":
            text = text[:cursor] + text[cursor + len(change) :]
            owner = owner[:cursor] + owner[cursor + len(change) :]
        elif "M" in activity:  # "Move From [a, b] To [c, d]"
            lhs, rhs = activity[10:].split(" To ")
            a, b = (int(v.strip("[] ")) for v in lhs.split(", "))
            c, d = (int(v.strip("[] ")) for v in rhs.split(", "))
            if a != c:
                if a < c:
                    text = text[:a] + text[b:d] + text[a:b] + text[d:]
                    owner = owner[:a] + owner[b:d] + owner[a:b] + owner[d:]
                else:
                    text = text[:c] + text[a:b] + text[c:a] + text[b:]
                    owner = owner[:c] + owner[a:b] + owner[c:a] + owner[b:]
        else:  # Input, Paste
            start = cursor - len(change)
            text = text[:start] + change + text[start:]
            owner = owner[:start] + [i] * len(change) + owner[start:]

        assert len(owner) == len(text), f"desync at event {i}: {len(owner)} != {len(text)}"

    return text, owner


def gen_token_events(essay: str, events: DataFrame, owner: list[int]) -> DataFrame:
    enc = tokenizer(essay, return_tensors="pt", return_offsets_mapping=True)
    rows = []
    for s, f in enc["offset_mapping"][0].tolist():
        # Extract overlapping events
        idx = sorted(set(owner[s:f]))

        if not idx:  # [CLS] / [SEP]
            rows.append({"up_time": 0, "cursor_position": 0, "action_time": 0, "n_events": 0})
            continue

        refs = events.iloc[idx]
        rows.append(
            {
                "up_time": refs["up_time"].iloc[-1],  # time at completion
                "cursor_position": refs["cursor_position"].iloc[-1],  # final position
                "action_time": refs["action_time"].sum(),  # a duration as a sum
                "n_events": len(idx),
            }
        )
    return DataFrame(rows)


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
        "input_sf": pad_sequence(
            [b["input_sf"] for b in batch], batch_first=True
        ),  # output: BxTx*, where T is the length of the longest sequence
        "input_deb": pad_sequence(
            [b["input_deb"] for b in batch], batch_first=True
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

        events = item[["activity", "cursor_position", "text_change", "up_time", "action_time"]]
        events = events[events.activity != "Nonproduction"]

        text, owner = replay_with_owner(events)
        token_events = gen_token_events(text, events, owner)

        enc = tokenizer(text, return_tensors="pt")
        assert len(token_events) == enc["input_ids"].shape[1], "streams on different grids"

        g_out = {
            "input_sf": tensor(
                token_events[["action_time", "cursor_position", "up_time"]].to_numpy(dtype="float32")
            ),  # squeezeformer input
            "input_deb": enc["input_ids"][0],  # deberta model input
            "attention_mask": enc["attention_mask"][0],  # same mask for both
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
# data_loader = DataLoader(cust_ds, batch_size=2, collate_fn=tr_collate_fn)
# it = iter(data_loader)
# batch = next(it)

# print({k: tuple(v.shape) for k, v in batch.items()})

# self = cust_ds

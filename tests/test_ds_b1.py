# %%
from pathlib import Path

import pandas
import pytest
import torch
from torch.utils.data import DataLoader

from configs.cfg_b1 import cfg
from data.ds_b1 import CustomDataset, tr_collate_fn


@pytest.mark.slow
@pytest.mark.skipif(
    not Path("datamount/train_folds.parquet").exists(),
    reason="datamount/*.parquet is gitignored, so the file is absent in CI",
)
def test_dataset_smoke():
    # Smoke Test
    df = pandas.read_parquet("datamount/train_folds.parquet")
    cust_ds = CustomDataset(df, cfg, "train")
    data_loader = DataLoader(cust_ds, batch_size=1, collate_fn=tr_collate_fn)
    it = iter(data_loader)
    batch = next(it)
    missing = [k for k in "input_sf input_deb attention_mask idx target".split() if k not in batch]
    assert not missing, f"missing keys: {missing}"

    FEATS = ["action_time", "cursor_position", "up_time"]

    rows_sf = []
    for x in cust_ds:
        rows_sf.append(x["input_sf"])

    sf = pandas.DataFrame(torch.cat(rows_sf).numpy(), columns=FEATS)

    print(sf.agg(["max", "min", "mean"]))

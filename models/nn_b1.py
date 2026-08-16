# %%
from types import SimpleNamespace
from typing import Any, Literal

import torch
from torch import concat, nn
from transformers import AutoConfig, AutoModel, DebertaV2Model

from data.ds_b1 import CustomDataset


class SqueezeFormer(nn.Module):
    def __init__(self):
        super(SqueezeFormer, self).__init__()
        self.nn = nn.Linear(3, 12)

    def forward(self, batch: Any) -> dict[str, Any]:

        output = self.nn(batch["input_sf"])

        return output


class Net(nn.Module):
    def __init__(self, dataset: CustomDataset, cfg: SimpleNamespace, mode: Literal["train", "val"]):
        super(Net, self).__init__()

        self.dataset = dataset
        self.cfg = cfg
        self.mode = mode

        config = AutoConfig.from_pretrained(cfg.backbone, **cfg.backbone_cfg)
        self.deberta: DebertaV2Model = AutoModel.from_pretrained(cfg.backbone, config=config)
        self.squeezeformer = SqueezeFormer()
        self.fc = nn.Linear(780, 1)
        self.criterion = nn.MSELoss()

        if self.cfg.gradient_checkpointing:
            self.deberta.gradient_checkpointing_enable()

    def len(self) -> int:
        return len(self.dataset)

    def forward(self, batch: Any) -> dict[str, Any]:
        # Deberta: (B,T) -> (B,T,768)
        out_deb = self.deberta(input_ids=batch["input_deb"], attention_mask=batch["attention_mask"])

        # Squeeze former: (B,T,3) -> (B,T,12)
        out_sq = self.squeezeformer(batch)

        # Concatenate
        composed = concat((out_deb.last_hidden_state, out_sq), dim=2)
        assert composed.shape[2] == 780  # 780=768+12

        # Pads carry non-zero hidden states, so set them to 0 before summing
        mask = batch["attention_mask"].unsqueeze(-1)  # (B,T) -> (B,T,1)
        pooled = (composed * mask).sum(dim=1) / mask.sum(dim=1)  # (B,T,1) -> (B,780)
        logits = self.fc(pooled).squeeze(-1)  # (B,)

        loss = torch.sqrt(self.criterion(logits, batch["target"]))

        return {"loss": loss, "preds": logits}


"""
import pandas as pd
from torch.utils.data import DataLoader

from data.ds_b1 import CustomDataset, collate_fn

cfg = SimpleNamespace()
cfg.backbone = "microsoft/deberta-v3-base"
cfg.backbone_cfg = {}
cfg.gradient_checkpointing = False

df = pd.read_parquet("datamount/train_folds.parquet")
ds = CustomDataset(df=df, cfg=cfg, mode="train")
loader = DataLoader(dataset=ds, collate_fn=collate_fn, batch_size=2)
batch = next(iter(loader))

net = Net(dataset=ds, cfg=cfg, mode="train")
out = net(batch)
assert len(out["preds"].shape) == 1 and out["preds"].shape[0] == 2, f"Missmatch: {out['preds'].shape}"
"""


class FeatureExtractor(nn.Module):
    """
    This class projects the in_feats to out_feats, mixing across ksize//2 in the T dim to have invariant T,
    then ignores padded positions in the batch norm, otherwise padded values would skew the mean and variance.
    """

    def __init__(self, in_feats: int, out_feats: int, ksize: int):
        super(FeatureExtractor, self).__init__()
        assert ksize % 2 == 1
        self.c1 = nn.Conv1d(in_channels=in_feats, out_channels=out_feats, kernel_size=ksize, padding=(ksize - 1) // 2)
        self.norm = nn.BatchNorm1d(out_feats)

    def forward(self, batch: Any, mask: Any):
        """Note that in the mask 1s represent real values and 0s padded values."""
        # The convolution layer accepts input as BxFxT, but input is BxTxF
        batch = batch.permute(0, 2, 1)  # BxTxF -> BxFxT
        out = self.c1(batch)
        out = out.permute(0, 2, 1)

        # Before normalization, flatten the input and mask padded values
        # otherwise the padded values bias the output
        flat_out = out.flatten(0, 1)
        flat_mask = mask.flatten(0, 1).bool()  # 1s for real positions
        real_rows = flat_out[flat_mask]
        normed = self.norm(real_rows)

        # The destination tensor must contain non-zero values in real positions only
        result = torch.zeros_like(flat_out)
        result[flat_mask] = normed  # real values at non-zero positions
        result = result.view(out.shape)
        return result


x = torch.randn(2, 20, 3)  # BxTxF  -> 2,3,20
mask = torch.ones(2, 20)
mask[1, 15:] = 0

net = FeatureExtractor(out_feats=256, in_feats=3, ksize=9)
out = net(x, mask)
# Note that real_rows.shape = (35, 256) -- 20 real T values for item 0, and 15 for item 1, 5 values are padded

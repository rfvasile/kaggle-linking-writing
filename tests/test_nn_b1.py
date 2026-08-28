# %%
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from configs.cfg_b1 import cfg
from data.ds_b1 import CustomDataset, collate_fn
from models.nn_b1 import Attention, FeatureExtractor, Net, SqueezeFormer

mode = "train"

if not Path("datamount/train_folds_test.parquet").exists():
    df = pd.read_parquet("datamount/train_folds.parquet")
    df.head(5).to_parquet("datamount/train_folds5.parquet")


@pytest.fixture
def dsl():
    ds = pd.read_parquet("datamount/train_folds5.parquet")
    dsw = CustomDataset(ds, cfg, mode=mode)
    dsl = DataLoader(dsw)
    return dsl


def test_net_forward(dsl: DataLoader):
    batch = next(iter(dsl))
    net = Net(batch, cfg, mode=mode)
    out = net(batch)
    assert out.shape == 3 and out.shape[2] == 256


def test_sf_forward(dsl: DataLoader):
    batch = next(iter(dsl))
    net = SqueezeFormer(cfg)
    out = net(batch)


def test_feature_extractor_experiment():
    # Feature extractor experiment
    x2 = torch.randn(2, 20, 3)  # BxTxF
    mask = torch.ones(2, 20)
    mask[1, 15:] = 0

    net = FeatureExtractor(out_feats=256, in_feats=3, ksize=9)
    out = net(x2, mask)
    # real_rows.shape = (35, 256) -- 20 real T values for item 0, and 15 for item 1, 5 values are padded
    print(out.shape)


def test_attention_experiment():
    # ---------------------
    # ----- Attention -----
    # ---------------------
    # Experiment 1: attention
    x1 = torch.randn(2, 20, 256)
    W_q = nn.Linear(x1.shape[2], 256)
    W_k = nn.Linear(x1.shape[2], 256)
    W_v = nn.Linear(x1.shape[2], 256)

    ## Projections
    q1 = W_q(x1)
    k1 = W_k(x1)
    v1 = W_v(x1)

    q1.shape
    q1.T.shape
    k1.shape

    ## Scores
    scores1 = q1 @ k1.mT  # (2, 20, 256) x (2, 256, 20)

    ### These fail:
    ### scores1 = q1 @ k1  # (2, 20, 256) x (2, 20, 256)
    ### scores1 = q1 @ k1.T  # (2, 20, 256) x (256, 20, 2)

    ### ----
    perm = torch.randperm(x1.shape[1])

    x2 = x1[:, perm, :]

    q2 = W_q(x2)
    k2 = W_k(x2)
    v2 = W_v(x2)

    scores2 = q2 @ k2.mT  # (BxTx256) x (Bx256xT) -> (BxTxT)

    ### Permuting the T dim. yields values that already exist in the initial tensor, so
    ### set(scores1[0, perm[i]]) == set(scores2[0, i])
    sorted1, _ = torch.sort(scores1[0, perm[0]].flatten())
    sorted2, _ = torch.sort(scores2[0, 0].flatten())
    assert torch.equal(sorted1, sorted2)

    ### softmax then sum over the same axis is always 1.0. dim=1 here is the query
    ### axis, not the key axis that Attention.forward actually normalizes over.
    torch.softmax(scores1, dim=1).sum(dim=1)


def test_attention_rope_experiment():
    cfg.apply_RoPE = True
    cfg.slow_RoPE = False
    # Attention smoke test
    attn_mask = torch.ones(2, 20)
    attn_mask[1, 15:] = 0
    batch = torch.rand(2, 20, 256)
    net = Attention(dim=256, num_heads=4, cfg=cfg)
    net(batch, attn_mask)

    # --- Rotary embeddings ---
    perm = torch.randperm(20)
    full = torch.ones(2, 20)

    # Correctness test: with positional information, the assertion must pass
    assert torch.allclose(net(batch[:, perm, :], full), net(batch, full)[:, perm, :], atol=1e-5) == False, (
        "If RoPE enabled, this must pass"
    )


@pytest.mark.skipif(
    not Path("datamount/train_folds.parquet").exists(),
    reason="datamount/*.parquet is gitignored, so the file is absent in CI",
)
def test_net_experiment():
    df = pd.read_parquet("datamount/train_folds.parquet")
    ds = CustomDataset(df=df, cfg=cfg, mode="train")
    loader = DataLoader(dataset=ds, collate_fn=collate_fn, batch_size=2)
    batch = next(iter(loader))

    net = Net(dataset=ds, cfg=cfg, mode="train")
    out = net(batch)
    assert len(out["preds"].shape) == 1 and out["preds"].shape[0] == 2, f"Missmatch: {out['preds'].shape}"

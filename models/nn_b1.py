# %%
import math
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
        # input_sf.shape = BxTxF
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
        self.fc = nn.Linear(1024, 1)
        self.criterion = nn.MSELoss()

        if self.cfg.gradient_checkpointing:
            self.deberta.gradient_checkpointing_enable()

    def len(self) -> int:
        return len(self.dataset)

    def forward(self, batch: Any) -> dict[str, Any]:
        # Deberta: (B,T) -> (B,T,768)
        out_deb = self.deberta(input_ids=batch["input_deb"], attention_mask=batch["attention_mask"])

        # Squeeze former: (B,T,3) -> (B,T,256)
        out_sq = self.squeezeformer(batch)

        # Concatenate
        composed = concat((out_deb.last_hidden_state, out_sq), dim=2)
        assert composed.shape[2] == 1024  # 768+256=1024

        # Pads carry non-zero hidden states, so set them to 0 before summing
        mask = batch["attention_mask"].unsqueeze(-1)  # (B,T) -> (B,T,1)
        pooled = (composed * mask).sum(dim=1) / mask.sum(dim=1)  # (B,T,1024) -> (B,1024)
        logits = self.fc(pooled).squeeze(-1)  # (B,)

        loss = torch.sqrt(self.criterion(logits, batch["target"]))

        return {"loss": loss, "preds": logits}


"""
import pandas as pd
from torch.utils.data import DataLoader

from configs.cfg_b1 import cfg
from data.ds_b1 import CustomDataset, collate_fn

df = pd.read_parquet("datamount/train_folds.parquet")
ds = CustomDataset(df=df, cfg=cfg, mode="train")
loader = DataLoader(dataset=ds, collate_fn=collate_fn, batch_size=2)
batch = next(iter(loader))

net = Net(dataset=ds, cfg=cfg, mode="train")
out = net(batch)
assert len(out["preds"].shape) == 1 and out["preds"].shape[0] == 2, f"Missmatch: {out['preds'].shape}"
"""

# %%


class FeatureExtractor(nn.Module):
    """
    This class projects the in_feats to out_feats, mixing across ksize//2 in the T dim to have invariant T,
    then ignores padded positions in the batch norm, otherwise padded values would skew the mean and variance.
    """

    def __init__(self, in_feats: int, out_feats: int, ksize: int):
        super(FeatureExtractor, self).__init__()
        assert ksize % 2 == 1
        # the padding value ensures the T dim preserves size
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


# Feature extractor experiment
x2 = torch.randn(2, 20, 3)  # BxTxF
mask = torch.ones(2, 20)
mask[1, 15:] = 0

net = FeatureExtractor(out_feats=256, in_feats=3, ksize=9)
out = net(x2, mask)
# real_rows.shape = (35, 256) -- 20 real T values for item 0, and 15 for item 1, 5 values are padded
print(out.shape)

# %%
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


# %%
## -----------------------------------------------------------------------------
## --- add positional embeddings
## -----------------------------------------------------------------------------


def rotation_slow(m: int, head_dim: int) -> torch.Tensor:
    """Simplest implementation of the rotation matrix. Reimplements figure 15 from docs/rope.pdf"""
    d = head_dim

    # The variables used within sin/cos
    i = torch.tensor([idx for idx in range(1, d // 2 + 1)])
    theta = torch.pow(10_000, (-2 * (i - 1) / d))

    # the sin/cos inside the rotation matrix
    cos = torch.cos(torch.tensor(m) * theta)
    sin = torch.sin(torch.tensor(m) * theta)

    # Build the rotation matrix
    R = torch.zeros(d * d).view((d, d))
    for i in range(d):
        R[i, i] = cos[i // 2]  # diagonal elements

        # off-diagonal elements
        if i % 2 == 0:
            R[i, i + 1] = -sin[i // 2]
        else:
            R[i, i - 1] = sin[i // 2]

    return R  # (d, d)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Optimized RoPE convetion. It removes the off-diagonal half of the rotation matrix."""
    # Swap each (2i, 2i+1) pair and negate one member
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return torch.stack((odd, -even), dim=-1).flatten(-2)


class Attention(nn.Module):
    # Fix ty warnings
    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, dim: int, num_heads: int, cfg: SimpleNamespace):
        super(Attention, self).__init__()
        self.cfg = cfg
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_o = nn.Linear(dim, dim, bias=False)

        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = self.dim // self.num_heads

        # The variables used within sin/cos
        d = self.head_dim
        i = torch.tensor([idx for idx in range(1, d // 2 + 1)])
        theta = torch.pow(10_000, (-2 * (i - 1) / d))

        # The position between distinct vectors becomes an axis
        max_length = 12876
        m_axis = torch.arange(max_length).unsqueeze(-1).to(torch.float32)
        theta = theta.unsqueeze(0)

        # the sin/cos inside the rotation matrix
        cos = torch.cos(m_axis @ theta)  # (12876 x head_dim)
        sin = torch.sin(m_axis @ theta)

        sin = sin.repeat_interleave(2, dim=-1)
        cos = cos.repeat_interleave(2, dim=-1)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def forward(self, batch, mask):
        """Attention module for the SqueezeFormer.

        Args:
            batch (B,T,256): input from the FeatureExtractor
            attention_mask (B,T): 1 = real, 0 = pad

        Returns:
            Tensor of dimension (B, T, 256)
        """
        # Projections: (BxTx256) x (256,256) -> (BxTx256)
        T = batch.shape[1]
        q = self.W_q(batch)  # what am i looking for
        k = self.W_k(batch)  # what do i offer (probs sum to 1)
        v = self.W_v(batch)  # what i actually hand over

        ## Separate input for multi-head attention
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim)  # (B,T,4,64)
        q = q.permute(0, 2, 1, 3)  # (B,4,T,64)

        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)

        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3)

        # Scores
        if self.cfg.apply_RoPE:

            def slow_impl():
                # Baseline impl
                Rs = []
                for r in range(T):
                    Rs.append(rotation_slow(r, self.head_dim))

                ## Get rotation matrix
                R = torch.stack(Rs)  # TxFxF
                return R

            assert self.head_dim % 2 == 0

            if self.cfg.slow_RoPE:
                # Straight-forward implementation but slow, costs B*H*T*head_dim**2
                R = slow_impl()  # TxFxF
                q_rot = (q.unsqueeze(-1).mT @ R).squeeze(-2)  # (BxHxTx1xF) -> (BxHxTxF)
                k_rot = (k.unsqueeze(-1).mT @ R).squeeze(-2)
            else:
                # Element-wise rotation without creating a rotation matrix
                cos, sin = self.cos[:T], self.sin[:T]  # (T,head_dim)
                q_rot = q * cos + rotate_half(q) * sin
                k_rot = k * cos + rotate_half(k) * sin

            # Merge the results and calculate the score
            scores = q_rot @ k_rot.mT
            scores = scores / math.sqrt(self.head_dim)
        else:
            scores = q @ k.mT  # (Bx4xTx64) x (Bx4x64xT) -> (Bx4xTxT) --- (Bx4xqxk)
            scores = scores / math.sqrt(self.head_dim)  ## keeps softmax out of saturation

        ## Pad must never attend to keys
        ## Ensure broadcasting works, so: (B,T) -> (B,1,1,T)
        mask = mask.unsqueeze(dim=-1).unsqueeze(dim=-1).permute(0, 2, 3, 1).to(torch.bool)
        mask = ~mask.bool()  # masked_fill expectes 1 for padded values
        ## do not use -inf, under mixed precision row that is filled with -inf softmaxes to NaN
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        # Softmax: T exists for both q and k, but we do softmax over k
        weights = torch.softmax(scores, dim=3)  # softmax over the keys

        # Weighted sum
        w_sum = weights @ v  # (Bx4xTxT) x (Bx4xTx64) -> (Bx4xTx64)

        # Heads. Permute first, otherwise reshape merges along the wrong dimension
        w_sum = w_sum.permute(0, 2, 1, 3).reshape(w_sum.shape[0], w_sum.shape[2], self.dim)  # (B,T,256)
        out = self.W_o(w_sum)  # (B,T,256) x (256,256)

        return out


from configs.cfg_b1 import cfg

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

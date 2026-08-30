# %%
import math
from types import SimpleNamespace
from typing import Any, Literal

import pandas as pd
import torch
from jurigged import watch
from torch import Tensor, concat, nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, DebertaV2Model

from configs.cfg_b1 import cfg
from data.ds_b1 import CustomDataset

# Equivalent of %autoreload
watch(".")

mode = "train"
ds = pd.read_parquet("datamount/train_folds5.parquet")
dsw = CustomDataset(ds, cfg, mode=mode)
dsl: DataLoader = DataLoader(dsw)


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

    def forward(self, batch: Tensor, mask: Any):
        """
        Args:
            batch (BxTx3): the event-derived input features
            mask (BxT): mask extracted from the tokenizer

        Note:
            The 1s in the mask represent real values and 0s padded values."""
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
        return result  # BxTxout_feats


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
        """
        Args:
            dim: the feature dimension (e.g., 256)
            num_heads: the number of parallel streams to process (e.g., 4)
            cfg: the experiment config
        """
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

    def forward(self, batch, attn_mask):
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
            assert self.head_dim % 2 == 0

            if self.cfg.slow_RoPE:
                # Straight-forward implementation but slow, costs B*H*T*head_dim**2
                Rs = []
                for r in range(T):
                    Rs.append(rotation_slow(r, self.head_dim))

                ## Get rotation matrix
                R = torch.stack(Rs)  # TxFxF
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

        ## Pad must never attend to keys: (B,T) -> (B,1,1,T)
        attn_mask = attn_mask.unsqueeze(dim=-1).unsqueeze(dim=-1).permute(0, 2, 3, 1).to(torch.bool)
        attn_mask = ~attn_mask.bool()  # masked_fill: 1 for padded values
        ## do not use -inf, under mixed precision row that is filled with -inf softmaxes to NaN
        scores = scores.masked_fill(attn_mask, torch.finfo(scores.dtype).min)

        # Softmax: T exists for both q and k, but we do softmax over k
        weights = torch.softmax(scores, dim=3)  # softmax over the keys

        # Weighted sum
        w_sum = weights @ v  # (Bx4xTxT) x (Bx4xTx64) -> (Bx4xTx64)

        # Heads. Permute first, otherwise reshape merges along the wrong dimension
        w_sum = w_sum.permute(0, 2, 1, 3).reshape(w_sum.shape[0], w_sum.shape[2], self.dim)  # (B,T,256)
        out = self.W_o(w_sum)  # (B,T,256) x (256,256)

        return out


# %%


class AttentionBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, ksize: int):
        super(AttentionBlock, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats

        self.attn: Attention = Attention(dim=cfg.feat_dim, num_heads=cfg.num_heads, cfg=cfg)
        self.fc1 = nn.Linear(in_feats, out_feats)
        self.fc2 = nn.Linear(in_feats, out_feats)
        self.conv = nn.Conv1d(in_channels=in_feats, out_channels=out_feats, kernel_size=ksize, padding=(ksize - 1) // 2)

    def forward(self, feats: Tensor, attn_mask: Tensor) -> Tensor:
        """
        Args:
            feats (BxTx256): feature extractor output
            attn_mask (BxT): obtained from the tokenizer

        Return:
            out (BxTx256): padded values are zeroed out
        """
        B, T, _ = feats.shape

        out = self.attn(feats, attn_mask=attn_mask)
        assert out.shape == (B, T, self.in_feats)

        out = self.fc1(out)

        attn_mask = attn_mask.unsqueeze(-1)
        out = out * attn_mask

        out = out.permute(0, 2, 1)
        assert out.shape == (B, self.out_feats, T)

        out = self.conv(out)
        out = out.permute(0, 2, 1)
        assert out.shape == (B, T, self.out_feats)

        out = self.fc2(out)
        assert out.shape == (B, T, self.out_feats)

        # Zero-out positions under the mask
        flat_out = out.flatten(0, 1)
        flat_mask = attn_mask.flatten(0, 2).bool()
        real_rows = flat_out[flat_mask]

        result = torch.zeros_like(flat_out)
        result[flat_mask] = real_rows
        result = result.view(out.shape)
        return result


# %%


class SqueezeFormer(nn.Module):
    def __init__(self, cfg: SimpleNamespace):
        super(SqueezeFormer, self).__init__()
        self.attn = Attention(dim=cfg.feat_dim, num_heads=cfg.num_heads, cfg=cfg)
        self.feats_extractor = FeatureExtractor(in_feats=cfg.in_feats, out_feats=cfg.out_feats, ksize=cfg.ksize)

    def forward(self, feats: Tensor, attention_mask: Tensor) -> dict[str, Any]:

        # Stem: (BxTx3) -> (BxTx256)
        out = self.feats_extractor(feats, attention_mask)

        # (BxTx256) -> (BxTx256)
        output = self.attn(out, attention_mask)

        return output


# %%


class Net(nn.Module):
    def __init__(self, dataset: CustomDataset, cfg: SimpleNamespace, mode: Literal["train", "val"]):
        super(Net, self).__init__()

        self.dataset = dataset
        self.cfg = cfg
        self.mode = mode

        config = AutoConfig.from_pretrained(cfg.backbone, **cfg.backbone_cfg)
        self.deberta: DebertaV2Model = AutoModel.from_pretrained(cfg.backbone, config=config)
        self.squeezeformer = SqueezeFormer(cfg)
        self.fc = nn.Linear(1024, 1)
        self.criterion = nn.MSELoss()

        if self.cfg.gradient_checkpointing:
            self.deberta.gradient_checkpointing_enable()

    def len(self) -> int:
        return len(self.dataset)

    def forward(self, batch: Any) -> dict[str, Any]:
        """
        Args:
            batch: a tensor with the data
            attn_mask: decides what to attend to. 1s for real values, 0s for padding
        """
        # Deberta: (B,T) -> (B,T,768)
        out_deb = self.deberta(input_ids=batch["input_deb"], attention_mask=batch["attention_mask"])

        # Squeeze former: (B,T,3) -> (B,T,256)
        out_sq = self.squeezeformer(feats=batch["input_sf"], attention_mask=batch["attention_mask"])

        # Concatenate
        composed = concat((out_deb.last_hidden_state, out_sq), dim=2)
        assert composed.shape[2] == 1024  # 768+256=1024

        # Pads carry non-zero hidden states, so set them to 0 before summing
        mask = batch["attention_mask"].unsqueeze(-1)  # (B,T) -> (B,T,1)
        pooled = (composed * mask).sum(dim=1) / mask.sum(dim=1)  # (B,T,1024) -> (B,1024)
        logits = self.fc(pooled).squeeze(-1)  # (B,)

        loss = torch.sqrt(self.criterion(logits, batch["target"]))

        return {"loss": loss, "preds": logits}

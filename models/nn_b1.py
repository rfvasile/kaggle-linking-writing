from types import SimpleNamespace

from torch import nn
from transformers import AutoConfig, AutoModel

from data.ds_b1 import CustomDataset


class Net(nn.Module):
    def __init__(self, dataset: CustomDataset, cfg: SimpleNamespace, mode: str):
        super(Net, self).__init__()

        self.dataset = dataset
        self.cfg = cfg
        self.mode = mode

        config = AutoConfig.from_pretrained(cfg.backbone, **cfg.backbone_cfg)
        self.backbone = AutoModel.from_pretrained(cfg.backbone, config=config)

        if self.cfg.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

    def len(self) -> int:
        return len(self.dataset)

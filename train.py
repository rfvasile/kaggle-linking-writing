# %%
import importlib
from argparse import ArgumentParser
from copy import copy

import numpy as np
import pandas as pd
from jurigged import watch

from utils import set_seed

# Equivalent of %autoreload
watch(".")

# %%
parser = ArgumentParser()
parser.add_argument("-C", "--config", help="config filename", default="baseline_default")
args = parser.parse_args()

cfg = copy(importlib.import_module(f"configs.{args.config}").cfg)

# %%
# Setup

if cfg.seed < 0:
    cfg.seed = np.random.randint(1_000_000)
print("seed", cfg.seed)
set_seed(cfg.seed)

Net = importlib.import_module(cfg.model).Net
CustomDataset = importlib.import_module(cfg.dataset).CustomDataset
tr_collate_fn = importlib.import_module(cfg.dataset).tr_collate_fn
val_collate_fn = importlib.import_module(cfg.dataset).val_collate_fn
batch_to_device = importlib.import_module(cfg.dataset).batch_to_device


# %%
# Load data
data = pd.read_csv("datamount/persuade_corpus_2.0_train.csv")

data.head(5)
data.to_csv("datamount/persuade_2.0_train.csv.gz")

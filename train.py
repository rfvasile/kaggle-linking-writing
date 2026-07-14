# %%
import glob
import importlib
import os
from argparse import ArgumentParser
from copy import copy

import numpy as np
import pandas as pd
from jurigged import watch

import wandb
from utils import set_seed

# Equivalent of %autoreload
watch(".")

# %%
parser = ArgumentParser()
parser.add_argument("-C", "--config", help="config filename", default="cfg_b1")
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
# File snapshots that should be stored in wandb
fns = [args.config] + [getattr(cfg, s) for s in "dataset model".split()]
fns = sum([glob.glob(f"./*/{fn.split('.')[-1]}.py") for fn in fns], [])

run = wandb.init(
    project=cfg.wandb_project,
    tags=["demo"],
    mode="online",
    config=vars(cfg),
    settings=wandb.Settings(console="off"),
    reinit="finish_previous",  # to stop run, close kernel
)

# Log code files
snapshot = {os.path.abspath(fn) for fn in fns}
run.log_code(root=".", include_fn=lambda path: path in snapshot)

# Define metrics
run.define_metric("curr_step")
run.define_metric("*", step_metric="curr_step")

print(f"wandb run id : {run.id}")
print(f"wandb URL    : {run.url}")


# %%
# Load data
data = pd.read_csv("datamount/persuade_corpus_2.0_train.csv")

data.head(5)
data.to_csv("datamount/persuade_2.0_train.csv.gz")

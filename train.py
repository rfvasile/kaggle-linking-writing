# %%
import gc
import glob
import importlib
import os
from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from jurigged import watch
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

import wandb
from utils import set_seed

# Equivalent of %autoreload
watch(".")


# %%
# 1. Input
parser = ArgumentParser()
parser.add_argument("-C", "--config", default="cfg_b1", help="config name")

args = parser.parse_args()

print(f"Config used: {args.config}")

# 2. Config. Deepcopy is used because importlib caches the modules as a singleton
cfg = deepcopy(importlib.import_module(f"configs.{args.config}").cfg)
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"


# %%
# Setup

# 3. Seed
if cfg.seed < 0:
    cfg.seed = np.random.randint(1_000_000)

set_seed(cfg.seed)
print("seed", cfg.seed)

# 4. Imports
Net = importlib.import_module(cfg.model).Net
CustomDataset = importlib.import_module(cfg.dataset).CustomDataset
tr_collate_fn = importlib.import_module(cfg.dataset).tr_collate_fn
val_collate_fn = importlib.import_module(cfg.dataset).val_collate_fn
batch_to_device = importlib.import_module(cfg.dataset).batch_to_device
post_process_pipeline = importlib.import_module(cfg.post_process_pipeline).post_process_pipeline
calc_metric = importlib.import_module(cfg.metric).calc_metric

# %%
# 5. Wandb: init (a)
run = wandb.init(
    project=cfg.wandb_project,
    tags=["baseline"],
    mode="online",
    config=vars(cfg),
    settings=wandb.Settings(console="auto"),
    reinit="return_previous",  # return previous existing run (needed because of REPL)
)

# Wandb logs (b)
modules = [args.config] + [getattr(cfg, s) for s in "dataset model post_process_pipeline metric".split()]
rel_paths = sum([glob.glob(f"./*/{file.split('.')[-1]}.py") for file in modules], [])
abs_paths = [os.path.abspath(path) for path in rel_paths]
run.log_code(root=".", include_fn=lambda file, _: file in abs_paths)

# Wandb metrics (c)
run.define_metric("curr_step")
run.define_metric(name="*", step_metric="curr_step")

print("wandb run id", run.id)
print("wandb URL   ", run.url)


# %%
# 6. Data (read, fold management, dataset wrapper, dataset loader)
df = pd.read_parquet(cfg.train_df)

train_df = df[df["fold"] != cfg.fold].copy()
if cfg.fold == -1:
    # even when all ds is used, keep fold 0 as validation
    val_df = df[df["fold"] == 0].copy()
else:
    val_df = df[df["fold"] == cfg.fold].copy()

train_dataset = CustomDataset(train_df, cfg=cfg, mode="train")
val_dataset = CustomDataset(val_df, cfg=cfg, mode="val")

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=cfg.batch_size,
    num_workers=cfg.num_workers,
    pin_memory=cfg.pin_memory,
    collate_fn=tr_collate_fn,
)
val_dataloader = DataLoader(
    dataset=val_dataset,
    shuffle=False,
    batch_size=cfg.batch_size,
    num_workers=cfg.num_workers,
    pin_memory=cfg.pin_memory,
    collate_fn=val_collate_fn,
)


# %%
# 7. Model, optimizer, scheduler, scaler
model = Net(dataset=train_dataset, cfg=cfg, mode="train")

optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=cfg.warmup * (len(train_dataset) // cfg.batch_size),
    num_training_steps=cfg.epochs * (len(train_dataset) // cfg.batch_size),
)
scaler = GradScaler(device=cfg.device, enabled=cfg.mixed_precision)


# %%
# Training
# Training
# for epoch in epochs:
#
#     # ---- TRAIN ----
#     model.train()
#     for batch in train_dataloader:
#         loss = model(batch)                      # forward, under autocast
#         backward(loss / grad_accumulation)       # scaled by AMP scaler
#
#         every grad_accumulation steps:
#             unscale grads                        # only if clipping or tracking norm
#             [measure grad norm] [clip]           # optional diagnostics
#             optimizer.step()                     # via scaler
#             optimizer.zero_grad()
#
#         scheduler.step()                         # per-batch, not per-epoch
#         log(train losses, lr, grad norms)
#
#     # ---- VALIDATE ----  (every eval_epochs, and always on last epoch)
#     model.eval(), no grads
#     val_data = collect model outputs over val_dataloader
#     val_data = concat per-batch outputs into flat tensors
#     preds    = post_process(val_data)
#     score    = metric(preds, val_df)
#     log(score)
#
#     # ---- CHECKPOINT ----
#     save model            # every epoch, unless save_only_last_ckpt


cfg.curr_step = 0
i = 0
optimizer.zero_grad()
ckpt_path = f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_last_seed{cfg.seed}.pth"
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

for epoch in range(cfg.epochs):
    # cfg, tqdm, iter, losses
    cfg.curr_epoch = epoch
    progress_bar = tqdm(range(len(train_dataloader)), desc=f"Train epoch {epoch}")
    tr_it = iter(train_dataloader)
    losses = []
    gc.collect()

    # ---- TRAIN ----
    model.train()
    torch.set_grad_enabled(True)
    for _ in progress_bar:
        i += 1
        cfg.curr_step += cfg.batch_size

        ...

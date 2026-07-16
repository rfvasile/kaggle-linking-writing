# %%
import gc
import glob
import importlib
import os
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from jurigged import watch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

import wandb
from utils import calc_grad_norm, set_seed

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
cfg.curr_step = 0
i = 0
optimizer.zero_grad()
ckpt_path = f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_last_seed{cfg.seed}.pth"
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

for epoch in range(cfg.epochs):
    cfg.curr_epoch = epoch
    progress_bar = tqdm(range(len(train_dataloader)), desc=f"Train epoch {epoch}")
    tr_it = iter(train_dataloader)
    losses = []
    gc.collect()

    model.train()
    torch.set_grad_enabled(True)
    for _ in progress_bar:
        i += 1
        cfg.curr_step += cfg.batch_size
        total_grad_norm = None
        total_grad_norm_after_clip = None

        batch = batch_to_device(next(tr_it), cfg.device)
        with autocast(cfg.device, enabled=cfg.mixed_precision):
            output_dict = model(batch)
        loss = output_dict["loss"]
        losses.append(loss.item())
        progress_bar.set_postfix(loss=np.mean(losses[-10:]))

        # Divide out-of-place so the logged train/loss keeps its unscaled value
        scaler.scale(loss / cfg.grad_accumulation).backward()

        if i % cfg.grad_accumulation == 0:
            if cfg.track_grad_norm or cfg.clip_grad > 0:
                scaler.unscale_(optimizer)
            if cfg.track_grad_norm:
                total_grad_norm = calc_grad_norm(model.parameters())
            if cfg.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                if cfg.track_grad_norm:
                    total_grad_norm_after_clip = calc_grad_norm(model.parameters())
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        log_dict = {f"train/{key}": output_dict[key].item() for key in output_dict if "loss" in key}
        log_dict["lr"] = optimizer.param_groups[0]["lr"]
        if total_grad_norm is not None:
            log_dict["total_grad_norm"] = total_grad_norm.item()
        if total_grad_norm_after_clip is not None:
            log_dict["total_grad_norm_after_clip"] = total_grad_norm_after_clip.item()
        run.log({**log_dict, "curr_step": cfg.curr_step})

    if (epoch + 1) % cfg.eval_epochs == 0 or (epoch + 1) == cfg.epochs:
        model.eval()
        torch.set_grad_enabled(False)

        val_data = defaultdict(list)
        for data in tqdm(val_dataloader, desc=f"Val epoch {epoch}"):
            batch = batch_to_device(data, cfg.device)
            with autocast(cfg.device, enabled=cfg.mixed_precision):
                output = model(batch)
            for key, val in output.items():
                val_data[key].append(val)

        # Collapse the per-batch outputs into single tensors/lists
        for key, value in val_data.items():
            if isinstance(value[0], list):
                val_data[key] = [item for sublist in value for item in sublist]
            elif value[0].dim() == 0:
                val_data[key] = torch.stack(value)
            else:
                val_data[key] = torch.cat(value, dim=0)

        if cfg.save_val_data:
            torch.save(val_data, f"{cfg.output_dir}/fold{cfg.fold}/val_data_seed{cfg.seed}.pth")

        pp_out = post_process_pipeline(cfg, val_data, val_df)
        val_score = calc_metric(cfg, pp_out, val_df, "val")
        if not isinstance(val_score, dict):
            val_score = {"score": val_score}

        for k, v in val_score.items():
            print(f"val_{k}: {v:.3f}")
        run.log({**{f"val/{k}": v for k, v in val_score.items()}, "curr_step": cfg.curr_step})

    if not cfg.save_only_last_ckpt:
        torch.save({"model": model.state_dict()}, ckpt_path)

torch.save({"model": model.state_dict()}, ckpt_path)
print(f"Checkpoint saved: {ckpt_path}")

run.log_model(path=ckpt_path, name=f"fold{cfg.fold}-seed{cfg.seed}")
run.finish()

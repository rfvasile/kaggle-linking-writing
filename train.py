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
from torch.amp import GradScaler
from torch.amp.autocast_mode import autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

import wandb
from train_train import total_grad_norm_after_clip
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
if cfg.seed == -1:
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

cfg.curr_step = 0
i = 0
optimizer.zero_grad()
ckpt_path = f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_last_seed{cfg.seed}.pth"
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

# Training with: mixed precision, track/clip gradients.
for epoch in range(cfg.epochs):
    # Epoch setup: curr epoch, tqdm, iter, losses, memory, training mode.
    cfg.curr_epoch = epoch
    progress_bar = tqdm(range(len(train_dataloader)), desc=f"Train epoch {epoch}")
    tr_it = iter(train_dataloader)
    losses = []
    gc.collect()

    # Train
    model.train()
    torch.set_grad_enabled(True)
    for _ in progress_bar:
        ## Iteration setup: iter, seen so far at curr step, init vars.
        i += 1
        cfg.curr_step += cfg.batch_size
        total_grad_norm = None
        total_grad_norm_after_clip = None

        ## Training
        batch = batch_to_device(next(tr_it), cfg.device)
        with autocast(cfg.device, enabled=cfg.mixed_precision, dtype=torch.float16):
            output = model(batch)
        loss = output["loss"]
        losses.append(loss.item())
        progress_bar.set_postfix(loss=np.mean(losses[-10:]))

        ## Accumulate scaled gradients (the logs use the unscaled value)
        scaler.scale(output["loss"] / cfg.grad_accumulation).backward()

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

        ## Log output parameters, lr, grad norms
        log_dict = {f"train/{key}": output[key].item() for key in output if "loss" in key}
        log_dict["lr"] = optimizer.param_groups[0]["lr"]
        if total_grad_norm is not None:
            log_dict["total_grad_norm"] = total_grad_norm.item()
        if total_grad_norm_after_clip is not None:
            log_dict["total_grad_norm_after_clip"] = total_grad_norm_after_clip.item()
        run.log({**log_dict, "curr_step": cfg.curr_step})

    # Validation
    if (epoch + 1) % cfg.eval_epochs == 0 or (epoch + 1) == cfg.epochs:
        ## Init data
        model.eval()
        torch.set_grad_enabled(False)

        val_data = defaultdict(list)
        ## Evaluate
        for data in tqdm(val_dataloader, desc=f"Val epoch {epoch}"):
            batch = batch_to_device(data, cfg.device)
            with autocast(cfg.device, enabled=cfg.mixed_precision, dtype=torch.float16):
                output = model(batch)

            ### Save the results
            for key, val in output.items():
                val_data[key].append(val)

        ## make uniform data
        for key, val in val_data.items():
            if isinstance(val[0], list):
                val_data[key] = [item for sublist in val for item in sublist]
            elif val[0].dim() == 0:  # Free tensor
                val_data[key] = torch.stack(val_data[key], dim=0)
            else:
                val_data[key] = torch.cat(val_data[key], dim=0)

        ## Save checkpoints
        if cfg.save_val_data:
            torch.save(val_data, f"{cfg.output_dir}/fold{cfg.fold}/val_data_seed{cfg.seed}.pth")

        ## Obtain validation score
        pp_out = post_process_pipeline(cfg, val_data, val_df)
        val_score = calc_metric(cfg, pp_out, val_df, "val")
        if not isinstance(val_score, dict):
            val_score = {"score": val_score}

        ## Log data
        for k, v in val_score.items():
            print(f"val_{k}: {v:.3f}")
        log_dict = {f"val/{k}": v for k, v in val_score.items()}
        run.log({**log_dict, "curr_step": cfg.curr_step})

    ## Saving
    if not cfg.save_only_last_ckpt:
        torch.save({"model": model.state_dict()}, ckpt_path)

# Save results
torch.save({"model": model.state_dict()}, ckpt_path)
print(f"Checkpoint saved: {ckpt_path}")

run.log_model(path=ckpt_path, name=f"fold{cfg.fold}-seed{cfg.seed}")
run.finish()

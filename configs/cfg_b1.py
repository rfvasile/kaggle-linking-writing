import os
from types import SimpleNamespace

cfg = SimpleNamespace(**{})


# init
cfg.name = os.path.basename(__file__).split(".")[0]
cfg.seed = 3
cfg.model = "models.nn_b1"
cfg.dataset = "data.ds_b1"
cfg.post_process_pipeline = "postprocess.pp_b1"
cfg.metric = "metrics.metric_b1"
cfg.output_dir = f"output/{os.path.basename(__file__).split('.')[0]}"

# data
cfg.pin_memory = False  # ds is small
cfg.num_workers = 4
cfg.batch_size = 4
cfg.train_df = "train_folds.parquet"

# model
cfg.backbone = "microsoft/deberta-v3-base"
cfg.backbone_cfg = {"attention_probs_dropout_prob": 0.0, "hidden_dropout_prob": 0.0}


# stages
cfg.train = True
cfg.val = True
cfg.test = True
cfg.train_val = True


# OPTIMIZATIONS & SCHEDULE
cfg.fold = 0
cfg.epochs = 7

cfg.lr = 2e-5
cfg.weight_decay = 1e-3
cfg.gradient_checkpointing = True
cfg.warmup = 1.0

cfg.mixed_precision = True
cfg.grad_accumulation = 1
cfg.clip_grad = 0.0
cfg.track_grad_norm = True

# eval & checkpoints
cfg.eval_epochs = 1
cfg.save_val_data = True
cfg.save_only_last_ckpt = False

# logs
cfg.wandb_project = "kaggle-linking-writing"

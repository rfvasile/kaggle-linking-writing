# Experiment Log

| # | Date | Hypothesis | Change | CV RMSE | Public LB | Decision / learning | Run |
|---|---|---|---|---:|---:|---|---|
| 1 | 2026-09-10 | Complete the first baseline to expose pipeline failures and establish a starting CV result. | `cfg_b1`, fold 0, DeBERTa + SqueezeFormer | 0.6169 (one fold) | — | ✅ Baseline reference. Training, validation, checkpointing, and W&B work. Logged `val/loss=0.5620` is biased because it averages batch RMSEs. | [nxzgslrt](https://wandb.ai/rfv/kaggle-linking-writing/runs/nxzgslrt) |


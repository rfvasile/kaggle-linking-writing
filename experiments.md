# Experiment Log

| # | Date | Hypothesis | Change | CV RMSE | Public LB | Decision / learning | Run |
|---|---|---|---|---:|---:|---|---|
| 1 | 2026-09-10 | Complete the first baseline to expose pipeline failures and establish a starting CV result. | `cfg_b1`, fold 0, DeBERTa + SqueezeFormer | 0.6169 (one fold) | — | ✅ Baseline reference. Training, validation, checkpointing, and W&B work. Logged `val/loss=0.5620` is biased because it averages batch RMSEs. | [nxzgslrt](https://wandb.ai/rfv/kaggle-linking-writing/runs/nxzgslrt) |
| 2 | 2026-09-10 | Correct metric aggregation and log full-epoch training results for a fair comparison. | Same `cfg_b1`, fold 0; metrics/logging only | Pending | — | Metric verified against saved run: 0.6169. | — |


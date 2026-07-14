# %%
import json
import subprocess

import pandas as pd
from autogluon.common.dataset import TabularDataset
from autogluon.core.metrics import root_mean_squared_error
from autogluon.tabular.predictor.predictor import TabularPredictor
from jurigged import watch

# Equivalent of %autoreload
watch(".")

# %%
# Data pipeline
train_feats = TabularDataset("datamount/train_features.parquet")
train_folds = TabularDataset("datamount/train_folds.parquet")

"""
# Explanator data analysis
# - Cheatsheet: https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf
# For better view run: `make open marimo scripts/train_ensemble_gluon.py`

from utils import eda_summary, plot_hist

eda_summary(df_train_labels)
plot_hist(df_train_labels)
plot_hist(df_train_feats)
"""

# This yields the original data (`train_features.parquet`) but
# with additional training columns (`fold`, `score`).
scores = train_folds[["id", "score"]].drop_duplicates()
folds = train_folds[["id", "fold"]].drop_duplicates()
df_train = train_feats.merge(scores, on="id", how="left")
df_train = df_train.merge(folds, on="id", how="left")

df_train
# %%
# Training

# We must use groups='fold' to train with pre-assigned fold grouping
# This is mandatory to ensure coherent predictions during blending
# i.e., when we do 0.6 * model_a + 0.4 * model_b
predictor = TabularPredictor(
    label="score",
    learner_kwargs={"ignored_columns": ["id", "idx"]},
    path="./models/GBM/",
    problem_type="regression",
    eval_metric="root_mean_squared_error",
    groups="fold",
).fit(df_train, num_gpus=1, presets="medium")

print("--- Training complete ---")
print("Model load path:")
print(f"  TabularPredictor.load({predictor.path})")

"""
# Reload model
predictor = TabularPredictor.load("models/GBM")
"""

# %%
# Check model performance
oof = predictor.predict_oof()
oof_df = pd.DataFrame({"id": df_train["id"], "pred": oof})

print(f"OOF RMSE: {root_mean_squared_error.error(df_train['score'], oof)}")

# Record this run (reproduce with: git checkout <hash> + rerun)
run_tag = subprocess.check_output(["git", "describe", "--always", "--dirty"], text=True).strip()
oof_df.to_csv(f"output/{run_tag}_oof.csv", index=False)
predictor.leaderboard().to_csv(f"output/{run_tag}_leaderboard.csv", index=False)

with open(f"output/{run_tag}_fit_info.json", "w") as f:
    json.dump(predictor.info(), f, indent=2, default=str)

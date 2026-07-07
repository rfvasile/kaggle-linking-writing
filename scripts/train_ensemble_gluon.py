# %%
from autogluon.tabular import TabularDataset, TabularPredictor
from jurigged import watch

# Equivalent of %autoreload
watch(".")

# %%
# Data pipeline
df_train_labels = TabularDataset("data/train_scores.csv")

# Generated via silver_bullet_feats_v1.py
df_train_feats = TabularDataset("datamount/train_features.csv.gz")

"""
# Explanator data analysis
# - Cheatsheet: https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf

from utils import eda_summary, plot_hist

eda_summary(df_train_labels)
plot_hist(df_train_labels)
plot_hist(df_train_feats)
"""

df_train = df_train_labels.merge(df_train_feats, on="id")

# %%
# Training
predictor = TabularPredictor(
    label="score", learner_kwargs={"ignored_columns": ["Id"]}, path="./models/GBM/"
).fit(df_train)

print("--- Training complete ---")
print("Model load path:")
print(f"  TabularPredictor.load({predictor.path})")

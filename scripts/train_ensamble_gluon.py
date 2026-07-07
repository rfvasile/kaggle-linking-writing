# %%
from jurigged import watch
from autogluon.tabular import TabularDataset, TabularPredictor

# Equivalent of %autoreload
watch(".")

# %%
# Data pipeline
df_train_labels = TabularDataset("data/train_scores.csv")

# Generated via silver_bullet_feats_v1.py
df_train_feats = TabularDataset("datamount/train_features.csv.gz")

"""
Cheatsheet: https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf

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


# %%
# Submission
df_test = TabularDataset("datamount/test_features.csv.gz")

submission = predictor.predict(df_test)

submission.to_csv("output/submission.csv")

# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

# %%
# Submission
df_test = TabularDataset("datamount/test_features.csv.gz")

predictor = TabularPredictor.load("models/GBM")

submission = predictor.predict(df_test)
assert isinstance(submission, pd.Series)

submission.to_csv("output/submission.csv")

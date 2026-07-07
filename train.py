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

import pathlib

import pandas as pd
from open_spiel.python.utils.gfile import IsDirectory

# %%
cwd = pathlib.Path().resolve()

IsDirectory(cwd / "datamount")

# %%
# Load data
data = pd.read_csv("datamount/persuade_corpus_2.0_train.csv")

data.head(5)
data.to_csv("datamount/persuade_2.0_train.csv.gz")


import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from pandas import DataFrame


def eda_summary(df: DataFrame, id_col: str | None = None, max_unique: int = 30) -> None:
    """EDA (explanatory data analysis) summary for a dataframe"""

    def h(label: str) -> None:
        print(f"\n--- {label} ---")

    h("shape")
    print(df.shape)

    h("head")
    print(df.head())

    h("dtypes")
    print(df.dtypes)

    h("info")
    print(df.info())

    h("isnull")
    print(df.isnull().sum())

    h("duplicates")
    print(df.duplicated().sum())

    if id_col:
        h("id unique")
        print(df[id_col].nunique() == len(df))

    h("describe")
    print(df.describe())

    # Spot data inconsistencies (i.e. a string appears as "ACTIVE", "active ", "pending")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if df[col].nunique() <= max_unique:
            h(f"values: {col}")
            print(df[col].value_counts())

    # Correlation using a single numeric column is meaningless
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] > 1:
        h("corr")
        print(num_df.corr())


def plot_hist(dataframe: DataFrame, columns: list[str] | None = None) -> None:
    """Plot the data distribution of specific columns, or all numeric columns."""
    if columns is None:
        columns = dataframe.select_dtypes(include=np.number).columns.tolist()

    n = len(columns)
    if n == 0:
        print("No numeric columns to plot.")
        return

    layout_cols = min(n, 4)
    layout_rows = math.ceil(n / layout_cols)

    axes = dataframe[columns].hist(
        layout=(layout_rows, layout_cols),
        figsize=(layout_cols * 4.5, layout_rows * 3.2),
        bins=50,
        xlabelsize=8,
        ylabelsize=8,
        alpha=0.4,
    )

    for ax in np.asarray(axes).flatten():
        ax.title.set_size(9)

    np.asarray(axes).flatten()[0].get_figure().subplots_adjust(hspace=0.6, wspace=0.35)
    plt.show()


def calc_grad_norm(parameters, norm_ord=2) -> torch.Tensor | None:
    """Total gradient norm across parameters; None if there are no grads or the norm is nan/inf."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return None
    total_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(g.detach(), norm_ord=norm_ord) for g in grads]), norm_ord=norm_ord
    )
    if total_norm.isnan() or total_norm.isinf():
        return None
    return total_norm


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

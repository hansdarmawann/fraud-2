"""Exploratory data analysis: univariate + multivariate.

Outputs:
    outputs/figures/univariate/   -- per-feature distributions
    outputs/figures/multivariate/ -- correlation, pair relationships, target overlays
    outputs/reports/data_summary.csv
    outputs/reports/class_balance.csv
    outputs/reports/categorical_target_rates.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.logger import get_logger
from utils.paths import (
    FIG_MULTI,
    FIG_UNI,
    ID_COLS,
    RAW_CSV,
    REPORTS,
    TARGET_COL,
)

sns.set_theme(style="whitegrid")
LOGGER = get_logger("eda")


def load_data() -> pd.DataFrame:
    LOGGER.info("Loading raw dataset from %s", RAW_CSV)
    df = pd.read_csv(RAW_CSV)
    LOGGER.info("Loaded data shape=%s", df.shape)
    return df


def summarise(df: pd.DataFrame) -> None:
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "missing_pct": df.isna().mean().mul(100).round(3),
            "n_unique": df.nunique(),
        }
    )
    desc = df.describe(include="all").T
    summary = summary.join(desc, how="left")
    out = REPORTS / "data_summary.csv"
    summary.to_csv(out)
    LOGGER.info("Wrote dataset summary -> %s", out)

    balance = df[TARGET_COL].value_counts(dropna=False).rename("count").to_frame()
    balance["pct"] = (balance["count"] / balance["count"].sum() * 100).round(3)
    balance.to_csv(REPORTS / "class_balance.csv")
    LOGGER.info("Class balance:\n%s", balance.to_string())


def _feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    drop = set(ID_COLS + [TARGET_COL])
    numeric = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in drop
    ]
    categorical = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c not in drop
    ]
    return numeric, categorical


def univariate(df: pd.DataFrame) -> None:
    numeric, categorical = _feature_columns(df)
    LOGGER.info(
        "Univariate plots: %d numeric, %d categorical features",
        len(numeric),
        len(categorical),
    )

    for col in numeric:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        sns.histplot(df[col], bins=40, kde=True, ax=axes[0], color="steelblue")
        axes[0].set_title(f"Histogram of {col}")
        sns.boxplot(x=df[col], ax=axes[1], color="lightcoral")
        axes[1].set_title(f"Boxplot of {col}")
        fig.tight_layout()
        path = FIG_UNI / f"num_{col}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        LOGGER.debug("saved %s", path)

    cat_rates_rows = []
    for col in categorical:
        order = df[col].value_counts().index.tolist()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(x=col, data=df, order=order, ax=ax, color="steelblue")
        ax.set_title(f"Distribution of {col}")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(FIG_UNI / f"cat_{col}.png", dpi=110)
        plt.close(fig)

        rates = df.groupby(col)[TARGET_COL].agg(["mean", "count"]).reset_index()
        rates["feature"] = col
        rates = rates.rename(columns={col: "value", "mean": "fraud_rate"})
        cat_rates_rows.append(rates[["feature", "value", "count", "fraud_rate"]])

    if cat_rates_rows:
        cat_rates = pd.concat(cat_rates_rows, ignore_index=True)
        cat_rates.to_csv(REPORTS / "categorical_target_rates.csv", index=False)
        LOGGER.info("Saved per-category fraud rates")

    # Target distribution.
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=TARGET_COL, data=df, ax=ax, palette=["#4c72b0", "#c44e52"])
    ax.set_title(f"Target distribution: {TARGET_COL}")
    fig.tight_layout()
    fig.savefig(FIG_UNI / f"target_{TARGET_COL}.png", dpi=110)
    plt.close(fig)


def multivariate(df: pd.DataFrame) -> None:
    numeric, categorical = _feature_columns(df)

    # Correlation heatmap on numeric + target.
    corr_cols = numeric + [TARGET_COL]
    corr = df[corr_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Pearson correlation (numeric features + target)")
    fig.tight_layout()
    fig.savefig(FIG_MULTI / "correlation_heatmap.png", dpi=110)
    plt.close(fig)
    LOGGER.info("Saved correlation heatmap")

    # Warn if features have suspiciously high correlation with target
    target_corr = corr[TARGET_COL].drop(TARGET_COL).abs().sort_values(ascending=False)
    very_high_corr = target_corr[target_corr > 0.95]
    if len(very_high_corr) > 0:
        LOGGER.warning(
            f"⚠️  Features with very high correlation (>0.95) with target detected: {dict(very_high_corr)}. "
            f"This indicates the dataset may have perfect or near-perfect feature-target separation. "
            f"Model performance may be unrealistically high and may not generalize to real-world data."
        )

    # Numeric feature distributions split by target.
    for col in numeric:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.kdeplot(
            data=df, x=col, hue=TARGET_COL, common_norm=False,
            fill=True, alpha=0.4, ax=ax, palette=["#4c72b0", "#c44e52"],
        )
        ax.set_title(f"{col} density by {TARGET_COL}")
        fig.tight_layout()
        fig.savefig(FIG_MULTI / f"kde_{col}_by_target.png", dpi=110)
        plt.close(fig)

    # Categorical x target fraud-rate bar plots.
    for col in categorical:
        rates = (
            df.groupby(col)[TARGET_COL]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=rates, x=col, y=TARGET_COL, ax=ax, color="steelblue")
        ax.set_title(f"Fraud rate by {col}")
        ax.set_ylabel("fraud rate")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(FIG_MULTI / f"fraud_rate_by_{col}.png", dpi=110)
        plt.close(fig)

    # Pairplot for the highest-signal numerics (cap to 4 to keep it cheap).
    top_numeric = (
        corr[TARGET_COL].drop(TARGET_COL).abs().sort_values(ascending=False).head(4).index.tolist()
    )
    if top_numeric:
        LOGGER.info("Top correlated numeric features with target: %s", top_numeric)
        sample = df[top_numeric + [TARGET_COL]].sample(
            n=min(len(df), 3000), random_state=42
        )
        g = sns.pairplot(
            sample,
            hue=TARGET_COL,
            palette=["#4c72b0", "#c44e52"],
            diag_kind="kde",
            plot_kws={"alpha": 0.5, "s": 12},
        )
        g.fig.suptitle("Pairwise plots of top numeric features", y=1.02)
        g.savefig(FIG_MULTI / "pairplot_top_numeric.png", dpi=110)
        plt.close("all")


def main() -> None:
    LOGGER.info("=== EDA stage ===")
    df = load_data()
    summarise(df)
    univariate(df)
    multivariate(df)
    LOGGER.info("EDA complete")


if __name__ == "__main__":
    main()

"""
GA4 UX Funnel Audit

This script analyzes exported BigQuery CSV files from Google's public GA4
ecommerce sample dataset. It performs:

- session-level data cleaning
- feature engineering
- funnel and segment diagnostics
- hypothesis tests
- train/test purchase propensity modeling
- UX behavior segmentation

Required input:
    data/raw/session_features.csv

Optional inputs:
    data/raw/landing_page_performance.csv
    data/raw/product_discovery.csv
    data/raw/traffic_source_quality.csv
    data/raw/funnel_by_device.csv
    data/raw/checkout_friction.csv
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"

TARGET = "converted_purchase"
RANDOM_STATE = 42

FUNNEL_STEPS = [
    ("Sessions", "session_key"),
    ("Product discovery", "reached_product_discovery"),
    ("Product detail", "reached_product_detail"),
    ("Add to cart", "reached_cart"),
    ("Begin checkout", "reached_checkout"),
    ("Purchase", "converted_purchase"),
]


def ensure_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
    return df


def load_csv(path: Path, required: bool = False) -> pd.DataFrame | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Missing required file: {path}\n"
                "Run sql/01_session_features.sql in BigQuery and export the result "
                "to data/raw/session_features.csv."
            )
        return None
    return normalize_columns(pd.read_csv(path))


def to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def winsorize_iqr(series: pd.Series, lower_floor: float | None = 0) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return numeric.fillna(0)

    q1, q3 = numeric.quantile([0.25, 0.75])
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return numeric.fillna(numeric.median())

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    if lower_floor is not None:
        lower = max(lower, lower_floor)
    return numeric.clip(lower=lower, upper=upper).fillna(numeric.median())


def clean_category(series: pd.Series, missing_value: str = "unknown") -> pd.Series:
    cleaned = series.astype("string").fillna(missing_value).str.strip()
    cleaned = cleaned.replace(
        {
            "": missing_value,
            "<Other>": "other",
            "(not set)": "not set",
            "nan": missing_value,
            "None": missing_value,
        }
    )
    return cleaned.fillna(missing_value)


def extract_path(value: Any) -> str:
    if pd.isna(value):
        return "(missing)"
    text = str(value).strip()
    if not text:
        return "(missing)"
    parsed = urlparse(text)
    path = parsed.path if parsed.scheme or parsed.netloc else text
    if not path:
        path = "/"
    return path[:140]


def limit_cardinality(series: pd.Series, top_n: int = 25) -> pd.Series:
    cleaned = clean_category(series)
    keepers = set(cleaned.value_counts(dropna=False).head(top_n).index)
    return cleaned.where(cleaned.isin(keepers), "other")


def clean_sessions(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    if "session_key" not in df.columns:
        raise ValueError("session_features.csv must include a session_key column.")

    df = df.drop_duplicates(subset=["session_key"]).copy()

    numeric_cols = [
        "session_duration_sec",
        "event_count",
        "unique_event_count",
        "page_view_count",
        "view_item_list_count",
        "view_item_count",
        "add_to_cart_count",
        "begin_checkout_count",
        "purchase_count",
        "scroll_count",
        "user_engagement_count",
        "engagement_time_msec",
        "purchase_revenue_usd",
        "total_item_quantity",
        "item_touch_count",
        "unique_products",
        "avg_item_price_usd",
        "reached_session_start",
        "reached_product_discovery",
        "reached_product_detail",
        "reached_cart",
        "reached_checkout",
        "converted_purchase",
    ]
    df = to_numeric(df, numeric_cols)

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).clip(lower=0)

    bool_cols = [
        "reached_session_start",
        "reached_product_discovery",
        "reached_product_detail",
        "reached_cart",
        "reached_checkout",
        "converted_purchase",
    ]
    for col in bool_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = (df[col].fillna(0) > 0).astype(int)

    category_cols = [
        "device_category",
        "operating_system",
        "country",
        "region",
        "traffic_source",
        "traffic_medium",
        "traffic_campaign",
        "landing_page",
        "landing_page_title",
    ]
    for col in category_cols:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = clean_category(df[col])

    if "session_date" in df.columns:
        df["session_date"] = pd.to_datetime(df["session_date"], errors="coerce")

    for col in [
        "session_duration_sec",
        "event_count",
        "page_view_count",
        "engagement_time_msec",
        "purchase_revenue_usd",
    ]:
        if col in df.columns:
            df[f"{col}_winsorized"] = winsorize_iqr(df[col])

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["engagement_seconds"] = df.get("engagement_time_msec", 0) / 1000
    df["engagement_seconds_winsorized"] = df.get(
        "engagement_time_msec_winsorized", df.get("engagement_time_msec", 0)
    ) / 1000
    df["landing_path"] = df["landing_page"].map(extract_path)
    df["landing_category_grouped"] = limit_cardinality(df["landing_page_title"], top_n=20)
    df["source_medium"] = (
        clean_category(df["traffic_source"]) + " / " + clean_category(df["traffic_medium"])
    )

    if "session_date" in df.columns:
        df["session_weekday"] = df["session_date"].dt.day_name().fillna("unknown")
        df["session_month"] = df["session_date"].dt.to_period("M").astype("string").fillna("unknown")
        df["is_weekend"] = df["session_date"].dt.dayofweek.isin([5, 6]).astype(int).fillna(0)
    else:
        df["session_weekday"] = "unknown"
        df["session_month"] = "unknown"
        df["is_weekend"] = 0

    df["journey_depth"] = (
        df["reached_product_discovery"]
        + df["reached_product_detail"]
        + df["reached_cart"]
        + df["reached_checkout"]
        + df["converted_purchase"]
    )
    df["journey_depth_pre_purchase"] = (
        df["reached_product_discovery"]
        + df["reached_product_detail"]
        + df["reached_cart"]
        + df["reached_checkout"]
    )
    df["bounce_like_session"] = (
        (df.get("event_count", 0) <= 2)
        & (df.get("page_view_count", 0) <= 1)
        & (df["reached_product_detail"] == 0)
    ).astype(int)
    df["high_engagement_no_purchase"] = (
        (df["converted_purchase"] == 0)
        & (df["journey_depth_pre_purchase"] >= 3)
        & (df["engagement_seconds"] >= df["engagement_seconds"].median())
    ).astype(int)
    df["cart_without_checkout"] = (
        (df["reached_cart"] == 1) & (df["reached_checkout"] == 0)
    ).astype(int)
    df["checkout_without_purchase"] = (
        (df["reached_checkout"] == 1) & (df["converted_purchase"] == 0)
    ).astype(int)

    df["ux_intent_segment"] = df.apply(assign_intent_segment, axis=1)
    return df


def assign_intent_segment(row: pd.Series) -> str:
    if row.get("converted_purchase", 0) == 1:
        return "Purchasers"
    if row.get("reached_checkout", 0) == 1:
        return "Checkout abandoners"
    if row.get("reached_cart", 0) == 1:
        return "Cart abandoners"
    if row.get("reached_product_detail", 0) == 1:
        return "Product detail browsers"
    if row.get("reached_product_discovery", 0) == 1:
        return "Discovery-only browsers"
    if row.get("bounce_like_session", 0) == 1:
        return "Shallow low-intent sessions"
    return "General visitors"


def safe_divide(num: float, den: float) -> float:
    return float(num / den) if den else np.nan


def overall_funnel(df: pd.DataFrame) -> pd.DataFrame:
    total_sessions = len(df)
    rows = []
    previous = total_sessions
    for step_name, col in FUNNEL_STEPS:
        count = total_sessions if col == "session_key" else int(df[col].sum())
        rows.append(
            {
                "step": step_name,
                "sessions": count,
                "share_of_all_sessions": safe_divide(count, total_sessions),
                "step_conversion_rate": safe_divide(count, previous),
                "step_dropoff_rate": 1 - safe_divide(count, previous)
                if previous
                else np.nan,
            }
        )
        previous = count
    return pd.DataFrame(rows)


def aggregate_segments(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["ux_intent_segment", "device_category", "source_medium"], dropna=False)
        .agg(
            sessions=("session_key", "count"),
            users=("user_pseudo_id", "nunique") if "user_pseudo_id" in df.columns else ("session_key", "count"),
            avg_events=("event_count", "mean"),
            avg_engagement_seconds=("engagement_seconds_winsorized", "mean"),
            median_engagement_seconds=("engagement_seconds", "median"),
            product_detail_rate=("reached_product_detail", "mean"),
            cart_rate=("reached_cart", "mean"),
            checkout_rate=("reached_checkout", "mean"),
            purchase_rate=("converted_purchase", "mean"),
            revenue_usd=("purchase_revenue_usd", "sum"),
            checkout_abandonment_rate=("checkout_without_purchase", "mean"),
            cart_without_checkout_rate=("cart_without_checkout", "mean"),
        )
        .reset_index()
    )
    grouped["revenue_per_session"] = grouped["revenue_usd"] / grouped["sessions"].replace(0, np.nan)
    grouped["friction_score"] = (
        grouped["cart_without_checkout_rate"].fillna(0) * 0.45
        + grouped["checkout_abandonment_rate"].fillna(0) * 0.45
        + (1 - grouped["purchase_rate"].fillna(0)) * 0.10
    )
    return grouped.sort_values(["friction_score", "sessions"], ascending=[False, False])


def cramers_v(table: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    if n == 0:
        return np.nan
    r, k = table.shape
    return math.sqrt((chi2 / n) / max(min(k - 1, r - 1), 1))


def chi_square_test(df: pd.DataFrame, category: str, outcome: str, label: str) -> dict[str, Any]:
    if category not in df.columns or outcome not in df.columns:
        return {
            "test": "Chi-square",
            "hypothesis": label,
            "status": "skipped",
            "reason": f"Missing required columns: {category}, {outcome}.",
        }
    subset = df[[category, outcome]].dropna()
    table = pd.crosstab(subset[category], subset[outcome])
    table = table.loc[table.sum(axis=1) > 0, table.sum(axis=0) > 0]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {
            "test": "Chi-square",
            "hypothesis": label,
            "status": "skipped",
            "reason": "Not enough category/outcome variation.",
        }

    chi2, p_value, dof, _ = stats.chi2_contingency(table)
    return {
        "test": "Chi-square",
        "hypothesis": label,
        "status": "completed",
        "statistic": chi2,
        "p_value": p_value,
        "degrees_of_freedom": dof,
        "effect_size": cramers_v(table),
        "effect_size_name": "Cramer's V",
        "n": int(table.to_numpy().sum()),
    }


def mann_whitney_test(df: pd.DataFrame, metric: str, outcome: str, label: str) -> dict[str, Any]:
    if metric not in df.columns or outcome not in df.columns:
        return {
            "test": "Mann-Whitney U",
            "hypothesis": label,
            "status": "skipped",
            "reason": f"Missing required columns: {metric}, {outcome}.",
        }
    subset = df[[metric, outcome]].dropna()
    groups = []
    for value in [0, 1]:
        group = subset.loc[subset[outcome] == value, metric].astype(float)
        if len(group) > 0:
            groups.append(group)

    if len(groups) < 2 or min(len(g) for g in groups) < 5:
        return {
            "test": "Mann-Whitney U",
            "hypothesis": label,
            "status": "skipped",
            "reason": "Not enough observations in both groups.",
        }

    u_stat, p_value = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
    n0, n1 = len(groups[0]), len(groups[1])
    rank_biserial = (2 * u_stat) / (n0 * n1) - 1
    return {
        "test": "Mann-Whitney U",
        "hypothesis": label,
        "status": "completed",
        "statistic": u_stat,
        "p_value": p_value,
        "effect_size": rank_biserial,
        "effect_size_name": "Rank-biserial correlation",
        "n": int(n0 + n1),
        "group_0_median": float(groups[0].median()),
        "group_1_median": float(groups[1].median()),
    }


def kruskal_test(df: pd.DataFrame, category: str, metric: str, label: str) -> dict[str, Any]:
    if category not in df.columns or metric not in df.columns:
        return {
            "test": "Kruskal-Wallis",
            "hypothesis": label,
            "status": "skipped",
            "reason": f"Missing required columns: {category}, {metric}.",
        }
    subset = df[[category, metric]].dropna()
    grouped = [
        group[metric].astype(float).values
        for _, group in subset.groupby(category)
        if len(group) >= 5
    ]
    if len(grouped) < 2:
        return {
            "test": "Kruskal-Wallis",
            "hypothesis": label,
            "status": "skipped",
            "reason": "Not enough groups with sufficient observations.",
        }

    h_stat, p_value = stats.kruskal(*grouped)
    n = sum(len(g) for g in grouped)
    k = len(grouped)
    epsilon_sq = (h_stat - k + 1) / (n - k) if n > k else np.nan
    return {
        "test": "Kruskal-Wallis",
        "hypothesis": label,
        "status": "completed",
        "statistic": h_stat,
        "p_value": p_value,
        "effect_size": epsilon_sq,
        "effect_size_name": "Epsilon squared",
        "n": int(n),
        "groups": int(k),
    }


def spearman_test(df: pd.DataFrame, x_col: str, y_col: str, label: str) -> dict[str, Any]:
    if x_col not in df.columns or y_col not in df.columns:
        return {
            "test": "Spearman correlation",
            "hypothesis": label,
            "status": "skipped",
            "reason": f"Missing required columns: {x_col}, {y_col}.",
        }
    subset = df[[x_col, y_col]].dropna()
    if len(subset) < 10:
        return {
            "test": "Spearman correlation",
            "hypothesis": label,
            "status": "skipped",
            "reason": "Not enough observations.",
        }
    rho, p_value = stats.spearmanr(subset[x_col], subset[y_col])
    return {
        "test": "Spearman correlation",
        "hypothesis": label,
        "status": "completed",
        "statistic": rho,
        "p_value": p_value,
        "effect_size": rho,
        "effect_size_name": "Spearman rho",
        "n": int(len(subset)),
    }


def shapiro_normality_check(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    label: str,
    max_sample: int = 5000,
) -> pd.DataFrame:
    rows = []
    if metric not in df.columns or group_col not in df.columns:
        return pd.DataFrame()

    for group_name, group in df[[metric, group_col]].dropna().groupby(group_col):
        values = pd.to_numeric(group[metric], errors="coerce").dropna()
        if len(values) < 3:
            rows.append(
                {
                    "metric": metric,
                    "group": group_name,
                    "status": "skipped",
                    "reason": "Fewer than 3 observations.",
                    "n": len(values),
                    "sample_n": len(values),
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "interpretation": "Insufficient data.",
                    "label": label,
                }
            )
            continue

        sample = values.sample(n=min(len(values), max_sample), random_state=RANDOM_STATE)
        statistic, p_value = stats.shapiro(sample)
        rows.append(
            {
                "metric": metric,
                "group": group_name,
                "status": "completed",
                "reason": "",
                "n": len(values),
                "sample_n": len(sample),
                "statistic": statistic,
                "p_value": p_value,
                "interpretation": "non-normal" if p_value < 0.05 else "normal-like",
                "label": label,
            }
        )
    return pd.DataFrame(rows)


def run_normality_checks(df: pd.DataFrame) -> pd.DataFrame:
    checks = [
        shapiro_normality_check(
            df,
            "engagement_seconds_winsorized",
            "converted_purchase",
            "Engagement time normality by purchase outcome.",
        ),
        shapiro_normality_check(
            df,
            "session_duration_sec_winsorized",
            "converted_purchase",
            "Session duration normality by purchase outcome.",
        ),
        shapiro_normality_check(
            df,
            "event_count_winsorized",
            "reached_cart",
            "Event count normality by cart-intent outcome.",
        ),
    ]
    checks = [check for check in checks if not check.empty]
    return pd.concat(checks, ignore_index=True) if checks else pd.DataFrame()


def run_hypothesis_tests(df: pd.DataFrame) -> pd.DataFrame:
    test_rows = [
        chi_square_test(
            df,
            "device_category",
            "converted_purchase",
            "Purchase conversion differs by device category.",
        ),
        chi_square_test(
            df,
            "traffic_medium",
            "converted_purchase",
            "Purchase conversion differs by traffic medium.",
        ),
        chi_square_test(
            df,
            "landing_category_grouped",
            "converted_purchase",
            "Purchase conversion differs by product/category entry point.",
        ),
        chi_square_test(
            df[df["reached_product_detail"] == 1],
            "landing_category_grouped",
            "reached_cart",
            "Product-detail-to-cart behavior differs by category.",
        ),
        chi_square_test(
            df[df["reached_product_detail"] == 1],
            "device_category",
            "reached_cart",
            "Product-detail-to-cart behavior differs by device.",
        ),
        chi_square_test(
            df[df["reached_checkout"] == 1],
            "device_category",
            "converted_purchase",
            "Checkout completion differs by device.",
        ),
        mann_whitney_test(
            df,
            "engagement_seconds_winsorized",
            "converted_purchase",
            "Converted sessions have different engagement time than non-converted sessions.",
        ),
        mann_whitney_test(
            df,
            "session_duration_sec_winsorized",
            "converted_purchase",
            "Converted sessions have different duration than non-converted sessions.",
        ),
        mann_whitney_test(
            df,
            "event_count_winsorized",
            "reached_cart",
            "Cart-intent sessions have different event volume than non-cart sessions.",
        ),
        mann_whitney_test(
            df,
            "avg_item_price_usd",
            "converted_purchase",
            "Converted sessions involve different average item prices than non-converted sessions.",
        ),
        kruskal_test(
            df,
            "device_category",
            "journey_depth",
            "Journey depth differs by device category.",
        ),
        kruskal_test(
            df,
            "traffic_medium",
            "journey_depth",
            "Journey depth differs by traffic medium.",
        ),
        kruskal_test(
            df,
            "landing_category_grouped",
            "journey_depth",
            "Journey depth differs by product/category entry point.",
        ),
        kruskal_test(
            df,
            "landing_category_grouped",
            "engagement_seconds_winsorized",
            "Engagement time differs by product/category entry point.",
        ),
        spearman_test(
            df,
            "event_count_winsorized",
            "journey_depth",
            "Event volume is associated with deeper journey progression.",
        ),
        spearman_test(
            df,
            "unique_products",
            "journey_depth",
            "Unique product breadth is associated with deeper journey progression.",
        ),
    ]
    return pd.DataFrame(test_rows)


def prepare_landing_page_data(df: pd.DataFrame, landing_df: pd.DataFrame | None) -> pd.DataFrame:
    if landing_df is not None:
        landing = landing_df.copy()
        numeric_cols = [
            "sessions",
            "avg_events_per_session",
            "avg_page_views_per_session",
            "avg_engagement_seconds",
            "product_detail_sessions",
            "cart_sessions",
            "checkout_sessions",
            "purchase_sessions",
            "revenue_usd",
            "product_detail_rate",
            "cart_rate",
            "purchase_rate",
            "revenue_per_session",
        ]
        landing = to_numeric(landing, numeric_cols)
        if "landing_page" in landing.columns:
            landing["landing_path"] = landing["landing_page"].map(extract_path)
        return landing

    landing = (
        df.groupby("landing_path", dropna=False)
        .agg(
            sessions=("session_key", "count"),
            avg_events_per_session=("event_count", "mean"),
            avg_page_views_per_session=("page_view_count", "mean"),
            avg_engagement_seconds=("engagement_seconds", "mean"),
            product_detail_sessions=("reached_product_detail", "sum"),
            cart_sessions=("reached_cart", "sum"),
            checkout_sessions=("reached_checkout", "sum"),
            purchase_sessions=("converted_purchase", "sum"),
            revenue_usd=("purchase_revenue_usd", "sum"),
        )
        .reset_index()
    )
    landing["product_detail_rate"] = landing["product_detail_sessions"] / landing["sessions"]
    landing["cart_rate"] = landing["cart_sessions"] / landing["sessions"]
    landing["purchase_rate"] = landing["purchase_sessions"] / landing["sessions"]
    landing["revenue_per_session"] = landing["revenue_usd"] / landing["sessions"]
    return landing.sort_values("sessions", ascending=False)


def prepare_product_data(product_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if product_df is None:
        return None
    product = product_df.copy()
    numeric_cols = [
        "list_impressions",
        "product_views",
        "add_to_cart_events",
        "begin_checkout_events",
        "purchase_events",
        "viewing_sessions",
        "cart_sessions",
        "purchase_sessions",
        "avg_item_price_usd",
        "units_purchased",
        "revenue_usd",
        "view_to_cart_rate",
        "view_to_purchase_rate",
    ]
    product = to_numeric(product, numeric_cols)
    for col in ["item_name", "item_category", "item_brand"]:
        if col in product.columns:
            product[col] = clean_category(product[col])
    product["friction_gap"] = product["view_to_cart_rate"].fillna(0).max() - product["view_to_cart_rate"].fillna(0)
    return product


def train_purchase_model(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    model_df = df.copy()
    if TARGET not in model_df.columns or model_df[TARGET].nunique() < 2:
        skipped = pd.DataFrame(
            [{"status": "skipped", "reason": "Target has fewer than two classes."}]
        )
        return skipped, pd.DataFrame(), pd.DataFrame()

    class_counts = model_df[TARGET].value_counts()
    if class_counts.min() < 10:
        skipped = pd.DataFrame(
            [{"status": "skipped", "reason": "At least one target class has fewer than 10 rows."}]
        )
        return skipped, pd.DataFrame(), pd.DataFrame()

    numeric_features = [
        "session_duration_sec_winsorized",
        "event_count_winsorized",
        "page_view_count",
        "view_item_list_count",
        "view_item_count",
        "add_to_cart_count",
        "begin_checkout_count",
        "scroll_count",
        "user_engagement_count",
        "engagement_seconds_winsorized",
        "journey_depth_pre_purchase",
        "bounce_like_session",
        "cart_without_checkout",
    ]
    categorical_features = [
        "device_category",
        "operating_system",
        "country",
        "traffic_source",
        "traffic_medium",
        "landing_path",
        "session_weekday",
    ]
    numeric_features = [col for col in numeric_features if col in model_df.columns]
    categorical_features = [col for col in categorical_features if col in model_df.columns]

    for col in categorical_features:
        model_df[col] = limit_cardinality(model_df[col], top_n=25)

    X = model_df[numeric_features + categorical_features]
    y = model_df[TARGET].astype(int)

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "status": "completed",
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "average_precision": average_precision_score(y_test, y_prob),
    }

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metrics["classification_report"] = json.dumps(report)
    metrics["confusion_matrix"] = json.dumps(confusion_matrix(y_test, y_pred).tolist())
    metrics_df = pd.DataFrame([metrics])

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[0]
    feature_importance = (
        pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
        .assign(abs_coefficient=lambda frame: frame["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
    )

    all_prob = model.predict_proba(X)[:, 1]
    scored = model_df[
        [
            "session_key",
            "ux_intent_segment",
            "device_category",
            "source_medium",
            "landing_path",
            TARGET,
            "reached_cart",
            "reached_checkout",
            "engagement_seconds_winsorized",
            "journey_depth_pre_purchase",
        ]
    ].copy()
    scored["predicted_purchase_probability"] = all_prob
    threshold = max(0.5, float(np.nanquantile(all_prob, 0.90)))
    high_intent_no_purchase = scored[
        (scored[TARGET] == 0) & (scored["predicted_purchase_probability"] >= threshold)
    ]
    high_intent_segments = (
        high_intent_no_purchase.groupby(
            ["ux_intent_segment", "device_category", "source_medium", "landing_path"], dropna=False
        )
        .agg(
            sessions=("session_key", "count"),
            avg_predicted_purchase_probability=("predicted_purchase_probability", "mean"),
            cart_rate=("reached_cart", "mean"),
            checkout_rate=("reached_checkout", "mean"),
            avg_engagement_seconds=("engagement_seconds_winsorized", "mean"),
            avg_journey_depth=("journey_depth_pre_purchase", "mean"),
        )
        .reset_index()
        .sort_values(["sessions", "avg_predicted_purchase_probability"], ascending=[False, False])
    )

    return metrics_df, feature_importance, high_intent_segments


def plot_overall_funnel(funnel: pd.DataFrame) -> None:
    plt.figure(figsize=(11, 6))
    sns.barplot(data=funnel, x="step", y="sessions", color="#2E86AB")
    plt.title("GA4 Ecommerce Funnel: Session Progression")
    plt.xlabel("")
    plt.ylabel("Sessions")
    plt.xticks(rotation=20, ha="right")
    total = funnel.loc[0, "sessions"] if not funnel.empty else 0
    for idx, row in funnel.iterrows():
        label = f"{row['sessions']:,.0f}\n{row['share_of_all_sessions']:.1%}"
        plt.text(idx, row["sessions"], label, ha="center", va="bottom", fontsize=9)
    plt.ylim(0, total * 1.18 if total else 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_overall_funnel.png", dpi=180)
    plt.close()


def plot_device_funnel(df: pd.DataFrame) -> None:
    device = (
        df.groupby("device_category", dropna=False)
        .agg(
            sessions=("session_key", "count"),
            product_discovery=("reached_product_discovery", "mean"),
            product_detail=("reached_product_detail", "mean"),
            add_to_cart=("reached_cart", "mean"),
            checkout=("reached_checkout", "mean"),
            purchase=("converted_purchase", "mean"),
        )
        .query("sessions >= 10")
        .sort_values("sessions", ascending=False)
    )
    if device.empty:
        return
    heatmap = device.drop(columns=["sessions"])
    plt.figure(figsize=(10, max(3.5, 0.55 * len(heatmap))))
    sns.heatmap(heatmap, annot=True, fmt=".1%", cmap="YlGnBu", cbar_kws={"label": "Rate"})
    plt.title("Funnel Rates by Device Category")
    plt.xlabel("Funnel step")
    plt.ylabel("Device")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_device_funnel_heatmap.png", dpi=180)
    plt.close()


def plot_source_quality(df: pd.DataFrame) -> None:
    source = (
        df.groupby("source_medium", dropna=False)
        .agg(
            sessions=("session_key", "count"),
            purchase_rate=("converted_purchase", "mean"),
            cart_rate=("reached_cart", "mean"),
            revenue_usd=("purchase_revenue_usd", "sum"),
        )
        .query("sessions >= 25")
        .sort_values("sessions", ascending=False)
        .head(15)
        .sort_values("purchase_rate", ascending=True)
    )
    if source.empty:
        return
    plt.figure(figsize=(10, 7))
    sns.barplot(data=source, y=source.index, x="purchase_rate", color="#4C956C")
    plt.title("Purchase Rate by Traffic Source / Medium")
    plt.xlabel("Purchase rate")
    plt.ylabel("")
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_traffic_source_purchase_rate.png", dpi=180)
    plt.close()


def plot_landing_pages(landing: pd.DataFrame) -> None:
    if landing is None or landing.empty or "sessions" not in landing.columns:
        return
    landing = landing.copy()
    landing = landing[landing["sessions"].fillna(0) >= 25].sort_values("sessions", ascending=False).head(30)
    if landing.empty:
        return
    plt.figure(figsize=(11, 7))
    size = landing.get("revenue_usd", pd.Series(1, index=landing.index)).fillna(0).clip(lower=0)
    size = 60 + (size / max(size.max(), 1)) * 500
    sns.scatterplot(
        data=landing,
        x="product_detail_rate",
        y="purchase_rate",
        size=size,
        sizes=(60, 560),
        hue="sessions",
        palette="viridis",
        legend=False,
    )
    plt.title("Landing Page UX Performance: Discovery vs Purchase")
    plt.xlabel("Product detail rate")
    plt.ylabel("Purchase rate")
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_landing_page_performance_matrix.png", dpi=180)
    plt.close()


def plot_product_friction(product: pd.DataFrame | None) -> None:
    if product is None or product.empty:
        return
    required = {"product_views", "view_to_cart_rate", "revenue_usd"}
    if not required.issubset(product.columns):
        return
    product = product[product["product_views"].fillna(0) >= 10].sort_values("product_views", ascending=False).head(50)
    if product.empty:
        return
    plt.figure(figsize=(11, 7))
    size = product["revenue_usd"].fillna(0).clip(lower=0)
    size = 60 + (size / max(size.max(), 1)) * 520
    sns.scatterplot(
        data=product,
        x="product_views",
        y="view_to_cart_rate",
        size=size,
        sizes=(60, 560),
        hue="revenue_usd",
        palette="mako",
        legend=False,
    )
    plt.title("Product Discovery Friction: Views vs View-to-Cart Rate")
    plt.xlabel("Product views")
    plt.ylabel("View-to-cart rate")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_product_friction_matrix.png", dpi=180)
    plt.close()


def plot_segment_distribution(segment_df: pd.DataFrame) -> None:
    if segment_df.empty:
        return
    intent = (
        segment_df.groupby("ux_intent_segment", dropna=False)
        .agg(sessions=("sessions", "sum"), friction_score=("friction_score", "mean"))
        .sort_values("sessions", ascending=True)
        .reset_index()
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=intent, y="ux_intent_segment", x="sessions", color="#D95D39")
    plt.title("Behavioral UX Intent Segments")
    plt.xlabel("Sessions")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_ux_intent_segments.png", dpi=180)
    plt.close()


def plot_model_importance(feature_importance: pd.DataFrame) -> None:
    if feature_importance is None or feature_importance.empty:
        return
    top = feature_importance.head(20).sort_values("coefficient", ascending=True)
    plt.figure(figsize=(10, 7))
    colors = np.where(top["coefficient"] >= 0, "#4C956C", "#C44536")
    plt.barh(top["feature"], top["coefficient"], color=colors)
    plt.title("Purchase Propensity Model: Top Logistic Coefficients")
    plt.xlabel("Coefficient")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_model_feature_importance.png", dpi=180)
    plt.close()


def plot_statistical_tests(test_df: pd.DataFrame) -> None:
    completed = test_df[test_df["status"] == "completed"].copy()
    if completed.empty:
        return
    completed["p_value_display"] = completed["p_value"].map(lambda x: f"{x:.2e}")
    completed["effect_size_display"] = completed["effect_size"].map(lambda x: f"{x:.3f}")
    table_df = completed[
        ["test", "hypothesis", "p_value_display", "effect_size_name", "effect_size_display"]
    ].head(8)
    wrapped = table_df.copy()
    wrapped["hypothesis"] = wrapped["hypothesis"].map(lambda text: "\n".join(textwrap.wrap(str(text), 42)))

    fig, ax = plt.subplots(figsize=(13, 0.8 + 0.7 * len(wrapped)))
    ax.axis("off")
    table = ax.table(
        cellText=wrapped.values,
        colLabels=["Test", "Hypothesis", "p-value", "Effect", "Size"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    plt.title("Statistical Test Summary", pad=18)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_statistical_test_summary.png", dpi=180)
    plt.close()


def run_visualizations(
    df: pd.DataFrame,
    funnel: pd.DataFrame,
    landing: pd.DataFrame,
    product: pd.DataFrame | None,
    segments: pd.DataFrame,
    feature_importance: pd.DataFrame,
    tests: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plot_overall_funnel(funnel)
    plot_device_funnel(df)
    plot_source_quality(df)
    plot_landing_pages(landing)
    plot_product_friction(product)
    plot_segment_distribution(segments)
    plot_model_importance(feature_importance)
    plot_statistical_tests(tests)


def write_markdown_summary(
    df: pd.DataFrame,
    funnel: pd.DataFrame,
    tests: pd.DataFrame,
    model_metrics: pd.DataFrame,
) -> None:
    sessions = len(df)
    users = df["user_pseudo_id"].nunique() if "user_pseudo_id" in df.columns else np.nan
    purchase_rate = df["converted_purchase"].mean()
    revenue = df["purchase_revenue_usd"].sum()

    dropoffs = funnel.iloc[1:].copy()
    dropoffs = dropoffs.replace([np.inf, -np.inf], np.nan).dropna(subset=["step_dropoff_rate"])
    worst_step = (
        dropoffs.sort_values("step_dropoff_rate", ascending=False).iloc[0].to_dict()
        if not dropoffs.empty
        else None
    )

    significant = tests[
        (tests["status"] == "completed") & (pd.to_numeric(tests["p_value"], errors="coerce") < 0.05)
    ]
    model_status = model_metrics.iloc[0].get("status", "unknown") if not model_metrics.empty else "skipped"
    is_kaggle_clickstream = (
        "traffic_source" in df.columns and df["traffic_source"].astype(str).eq("kaggle_electronics").any()
    )
    title = (
        "# Ecommerce UX Funnel Audit - Analysis Summary"
        if is_kaggle_clickstream
        else "# GA4 UX Funnel Audit - Analysis Summary"
    )
    source_line = (
        "- Data source: **Kaggle eCommerce events history in electronics store**"
        if is_kaggle_clickstream
        else "- Data source: **Google GA4 BigQuery public ecommerce sample**"
    )
    limitation = (
        "The dataset tracks `view`, `cart`, and `purchase` events. Cart behavior is treated as the "
        "measurable purchase-intent stage, so later-stage findings should be read as cart-to-purchase friction."
        if is_kaggle_clickstream
        else "Google's GA4 ecommerce sample data is obfuscated. Use this project to demonstrate "
        "analytics workflow and UX reasoning, not to claim exact Google Merchandise Store performance."
    )

    lines = [
        title,
        "",
        "## Dataset Snapshot",
        source_line,
        f"- Sessions analyzed: **{sessions:,.0f}**",
        f"- Users analyzed: **{users:,.0f}**" if not pd.isna(users) else "- Users analyzed: unavailable",
        f"- Session purchase rate: **{purchase_rate:.2%}**",
        f"- Observed purchase revenue: **${revenue:,.2f}**",
        "",
        "## Funnel Diagnostic",
    ]
    if worst_step:
        lines.append(
            f"- Largest step drop-off: **{worst_step['step']}** "
            f"({worst_step['step_dropoff_rate']:.2%} drop-off from prior step)."
        )
    lines.extend(
        [
            "- Full funnel table: `outputs/overall_funnel.csv`",
            "- Funnel chart: `outputs/01_overall_funnel.png`",
            "",
            "## Statistical Testing",
            f"- Completed significant tests at alpha=0.05: **{len(significant)}**",
            "- Test details: `outputs/statistical_tests.csv`",
            "",
            "## Predictive Modeling",
            f"- Purchase propensity model status: **{model_status}**",
            "- Model metrics: `outputs/model_performance.csv`",
            "- Feature importance: `outputs/model_feature_importance.csv`",
            "- High-intent no-purchase segments: `outputs/high_intent_no_purchase_segments.csv`",
            "",
            "## UX Segmentation",
            "- Segment table: `outputs/ux_behavior_segments.csv`",
            "- High-intent no-purchase segments: `outputs/high_intent_no_purchase_segments.csv`",
            "",
            "## Event Scope",
            limitation,
            "",
        ]
    )
    (REPORT_DIR / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_output_index() -> None:
    outputs = sorted(OUTPUT_DIR.glob("*"))
    reports = sorted(REPORT_DIR.glob("*"))
    lines = ["# Output Index", "", "## Outputs"]
    lines.extend(f"- `{path.relative_to(ROOT)}`" for path in outputs)
    lines.extend(["", "## Reports"])
    lines.extend(f"- `{path.relative_to(ROOT)}`" for path in reports)
    (REPORT_DIR / "output_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(session_file: Path) -> None:
    ensure_dirs()
    raw_sessions = load_csv(session_file, required=True)
    assert raw_sessions is not None

    sessions = engineer_features(clean_sessions(raw_sessions))
    sessions.to_csv(PROCESSED_DIR / "session_features_clean.csv", index=False)

    landing_raw = load_csv(RAW_DIR / "landing_page_performance.csv")
    product_raw = load_csv(RAW_DIR / "product_discovery.csv")
    landing = prepare_landing_page_data(sessions, landing_raw)
    product = prepare_product_data(product_raw)

    funnel = overall_funnel(sessions)
    segments = aggregate_segments(sessions)
    tests = run_hypothesis_tests(sessions)
    normality = run_normality_checks(sessions)
    model_metrics, feature_importance, high_intent = train_purchase_model(sessions)

    funnel.to_csv(OUTPUT_DIR / "overall_funnel.csv", index=False)
    landing.to_csv(OUTPUT_DIR / "landing_page_scorecard.csv", index=False)
    segments.to_csv(OUTPUT_DIR / "ux_behavior_segments.csv", index=False)
    tests.to_csv(OUTPUT_DIR / "statistical_tests.csv", index=False)
    normality.to_csv(OUTPUT_DIR / "normality_checks.csv", index=False)
    model_metrics.to_csv(OUTPUT_DIR / "model_performance.csv", index=False)
    feature_importance.to_csv(OUTPUT_DIR / "model_feature_importance.csv", index=False)
    high_intent.to_csv(OUTPUT_DIR / "high_intent_no_purchase_segments.csv", index=False)
    if product is not None:
        product.to_csv(OUTPUT_DIR / "product_discovery_scorecard.csv", index=False)

    run_visualizations(sessions, funnel, landing, product, segments, feature_importance, tests)
    write_markdown_summary(sessions, funnel, tests, model_metrics)
    write_output_index()


def print_checklist() -> None:
    ensure_dirs()
    expected = [
        RAW_DIR / "session_features.csv",
        RAW_DIR / "landing_page_performance.csv",
        RAW_DIR / "product_discovery.csv",
        RAW_DIR / "traffic_source_quality.csv",
        RAW_DIR / "funnel_by_device.csv",
        RAW_DIR / "checkout_friction.csv",
    ]
    print("GA4 UX Funnel Audit input checklist")
    print("-----------------------------------")
    for path in expected:
        status = "FOUND" if path.exists() else "missing"
        requirement = "required" if path.name == "session_features.csv" else "optional"
        print(f"{status:7} {requirement:8} {path.relative_to(ROOT)}")

    print("\nRequired first step:")
    print("1. Open BigQuery.")
    print("2. Run sql/01_session_features.sql.")
    print("3. Export the results to data/raw/session_features.csv.")
    print("4. Run: python analysis.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GA4 UX Funnel Audit analysis.")
    parser.add_argument(
        "--session-file",
        type=Path,
        default=RAW_DIR / "session_features.csv",
        help="Path to the exported session_features CSV.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Print required/optional input file checklist and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_only:
        print_checklist()
        return

    session_file = args.session_file
    if not session_file.is_absolute():
        session_file = ROOT / session_file

    run_analysis(session_file)
    print("Analysis complete. See reports/analysis_summary.md and reports/output_index.md.")


if __name__ == "__main__":
    main()

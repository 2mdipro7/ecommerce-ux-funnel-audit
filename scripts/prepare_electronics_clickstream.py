"""
Prepare the Kaggle electronics ecommerce clickstream dataset for the GA4-style
UX funnel pipeline.

Input:
    data/raw/electronics_events/events.csv

Outputs:
    data/raw/session_features.csv
    data/raw/product_discovery.csv
    data/raw/event_overview.csv
    reports/data_source_note.md

Dataset:
    https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-electronics-store
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"


def first_non_null(series: pd.Series) -> str:
    values = series.dropna().astype(str).str.strip()
    values = values[values.ne("")]
    return values.iloc[0] if not values.empty else "unknown"


def load_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [col.strip().lower() for col in df.columns]

    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["event_time", "event_type", "user_id", "user_session", "product_id"])
    df = df[df["event_type"].isin(["view", "cart", "purchase"])]
    df = df[df["price"].fillna(0) >= 0]

    df["category_code"] = df["category_code"].fillna("unknown").astype(str)
    df["brand"] = df["brand"].fillna("unknown").astype(str)
    df["product_id"] = df["product_id"].astype(str)
    df["user_id"] = df["user_id"].astype(str)
    df["user_session"] = df["user_session"].astype(str)
    df["event_date"] = df["event_time"].dt.date
    return df.sort_values(["user_session", "event_time"])


def build_session_features(df: pd.DataFrame) -> pd.DataFrame:
    event_counts = (
        pd.crosstab(df["user_session"], df["event_type"])
        .rename(columns={"view": "view_item_count", "cart": "add_to_cart_count", "purchase": "purchase_count"})
        .reset_index()
    )
    for col in ["view_item_count", "add_to_cart_count", "purchase_count"]:
        if col not in event_counts.columns:
            event_counts[col] = 0

    revenue = (
        df[df["event_type"].eq("purchase")]
        .groupby("user_session", dropna=False)
        .agg(
            purchase_revenue_usd=("price", "sum"),
            total_item_quantity=("product_id", "count"),
        )
        .reset_index()
    )

    base = (
        df.groupby("user_session", dropna=False)
        .agg(
            user_pseudo_id=("user_id", "first"),
            session_date=("event_time", "min"),
            session_start_ts=("event_time", "min"),
            session_end_ts=("event_time", "max"),
            event_count=("event_type", "count"),
            unique_event_count=("event_type", "nunique"),
            item_touch_count=("product_id", "count"),
            unique_products=("product_id", "nunique"),
            avg_item_price_usd=("price", "mean"),
            landing_page_title=("category_code", "first"),
            first_brand=("brand", "first"),
        )
        .reset_index()
        .rename(columns={"user_session": "session_key"})
    )
    base["session_date"] = base["session_date"].dt.date

    features = base.merge(event_counts, left_on="session_key", right_on="user_session", how="left")
    features = features.drop(columns=["user_session"], errors="ignore")
    features = features.merge(revenue, left_on="session_key", right_on="user_session", how="left")
    features = features.drop(columns=["user_session"], errors="ignore")

    count_cols = [
        "view_item_count",
        "add_to_cart_count",
        "purchase_count",
        "purchase_revenue_usd",
        "total_item_quantity",
    ]
    for col in count_cols:
        features[col] = features[col].fillna(0)

    features["session_duration_sec"] = (
        features["session_end_ts"] - features["session_start_ts"]
    ).dt.total_seconds().clip(lower=0)
    features["engagement_time_msec"] = features["session_duration_sec"] * 1000

    features["page_view_count"] = features["view_item_count"]
    features["view_item_list_count"] = 0
    features["begin_checkout_count"] = features["add_to_cart_count"]
    features["scroll_count"] = 0
    features["user_engagement_count"] = np.where(features["event_count"] > 1, 1, 0)

    features["reached_session_start"] = 1
    features["reached_product_discovery"] = (features["event_count"] > 0).astype(int)
    features["reached_product_detail"] = (features["view_item_count"] > 0).astype(int)
    features["reached_cart"] = (features["add_to_cart_count"] > 0).astype(int)
    # Dataset limitation: no explicit checkout event exists. Cart is the closest
    # observable checkout-intent proxy for the generic analysis pipeline.
    features["reached_checkout"] = features["reached_cart"]
    features["converted_purchase"] = (features["purchase_count"] > 0).astype(int)

    features["device_category"] = "unknown"
    features["operating_system"] = "unknown"
    features["country"] = "unknown"
    features["region"] = "unknown"
    features["traffic_source"] = "kaggle_electronics"
    features["traffic_medium"] = "clickstream"
    features["traffic_campaign"] = "electronics_events_history"
    features["landing_page"] = (
        "/category/" + features["landing_page_title"].fillna("unknown").astype(str).str.replace(" ", "-", regex=False)
    )

    ordered = [
        "session_key",
        "user_pseudo_id",
        "session_date",
        "session_start_ts",
        "session_end_ts",
        "session_duration_sec",
        "device_category",
        "operating_system",
        "country",
        "region",
        "traffic_source",
        "traffic_medium",
        "traffic_campaign",
        "landing_page",
        "landing_page_title",
        "event_count",
        "unique_event_count",
        "unique_products",
        "avg_item_price_usd",
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
        "reached_session_start",
        "reached_product_discovery",
        "reached_product_detail",
        "reached_cart",
        "reached_checkout",
        "converted_purchase",
    ]
    return features[ordered]


def build_product_discovery(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["product_id", "brand", "category_code"]
    event_counts = df.groupby(keys + ["event_type"], dropna=False).size().unstack(fill_value=0)
    session_counts = (
        df.groupby(keys + ["event_type"], dropna=False)["user_session"].nunique().unstack(fill_value=0)
    )
    avg_price = df.groupby(keys, dropna=False)["price"].mean()
    purchase_revenue = df[df["event_type"].eq("purchase")].groupby(keys, dropna=False)["price"].sum()

    product = event_counts.reset_index()
    for col in ["view", "cart", "purchase"]:
        if col not in product.columns:
            product[col] = 0

    sessions = session_counts.reset_index()
    for col in ["view", "cart", "purchase"]:
        if col not in sessions.columns:
            sessions[col] = 0

    product = product.merge(
        sessions[keys + ["view", "cart", "purchase"]],
        on=keys,
        how="left",
        suffixes=("", "_sessions"),
    )
    product = product.merge(avg_price.rename("avg_item_price_usd").reset_index(), on=keys, how="left")
    product = product.merge(purchase_revenue.rename("revenue_usd").reset_index(), on=keys, how="left")
    product["revenue_usd"] = product["revenue_usd"].fillna(0)

    product = product.rename(
        columns={
            "product_id": "item_id",
            "brand": "item_brand",
            "category_code": "item_category",
            "view": "product_views",
            "cart": "add_to_cart_events",
            "purchase": "purchase_events",
            "view_sessions": "viewing_sessions",
            "cart_sessions": "cart_sessions",
            "purchase_sessions": "purchase_sessions",
        }
    )
    product["list_impressions"] = 0
    product["begin_checkout_events"] = product["add_to_cart_events"]
    product["units_purchased"] = product["purchase_events"]
    product["item_name"] = "product_" + product["item_id"].astype(str)
    product["item_category2"] = product["item_category"].str.split(".").str[0].fillna("unknown")
    product["item_category3"] = product["item_category"].str.split(".").str[1].fillna("unknown")
    product["view_to_cart_rate"] = product["add_to_cart_events"] / product["product_views"].replace(0, np.nan)
    product["view_to_purchase_rate"] = product["purchase_events"] / product["product_views"].replace(0, np.nan)
    cols = [
        "item_id",
        "item_name",
        "item_brand",
        "item_category",
        "item_category2",
        "item_category3",
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
    return product[cols].sort_values("product_views", ascending=False)


def build_event_overview(df: pd.DataFrame) -> pd.DataFrame:
    event_counts = df.groupby(["event_date", "event_type"], dropna=False).size().unstack(fill_value=0)
    for col in ["view", "cart", "purchase"]:
        if col not in event_counts.columns:
            event_counts[col] = 0

    base = df.groupby("event_date", dropna=False).agg(
        total_events=("event_type", "count"),
        users=("user_id", "nunique"),
        sessions=("user_session", "nunique"),
    )
    revenue = df[df["event_type"].eq("purchase")].groupby("event_date", dropna=False)["price"].sum()
    overview = base.join(event_counts[["view", "cart", "purchase"]]).join(
        revenue.rename("purchase_revenue_usd")
    )
    overview["purchase_revenue_usd"] = overview["purchase_revenue_usd"].fillna(0)
    overview = overview.reset_index().rename(columns={"event_date": "event_dt"})
    overview["page_views"] = overview["view"]
    overview["product_views"] = overview["view"]
    overview["add_to_cart_events"] = overview["cart"]
    overview["begin_checkout_events"] = overview["cart"]
    overview["purchase_events"] = overview["purchase"]
    return overview[
        [
            "event_dt",
            "total_events",
            "users",
            "sessions",
            "page_views",
            "product_views",
            "add_to_cart_events",
            "begin_checkout_events",
            "purchase_events",
            "purchase_revenue_usd",
        ]
    ].sort_values("event_dt")


def write_note(events: pd.DataFrame, sessions: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    event_counts = "\n".join(
        f"- {event_type}: {count:,}"
        for event_type, count in events["event_type"].value_counts().items()
    )
    note = f"""# Data Source Note

This project uses a real ecommerce clickstream dataset from an electronics store:

https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-electronics-store

Rows after cleaning: {len(events):,}

Sessions after aggregation: {len(sessions):,}

Observed event types:

{event_counts}

Event taxonomy: this dataset has `view`, `cart`, and `purchase` events. The analysis treats cart behavior as the measurable purchase-intent stage and focuses later-stage findings on cart-to-purchase friction.
"""
    (REPORT_DIR / "data_source_note.md").write_text(note, encoding="utf-8")


def main() -> None:
    input_path = RAW_DIR / "electronics_events" / "events.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    events = load_events(input_path)
    sessions = build_session_features(events)
    product = build_product_discovery(events)
    overview = build_event_overview(events)

    sessions.to_csv(RAW_DIR / "session_features.csv", index=False)
    product.to_csv(RAW_DIR / "product_discovery.csv", index=False)
    overview.to_csv(RAW_DIR / "event_overview.csv", index=False)
    write_note(events, sessions)

    print(f"Prepared {len(events):,} events into {len(sessions):,} sessions.")
    print(f"Wrote {RAW_DIR / 'session_features.csv'}")
    print(f"Wrote {RAW_DIR / 'product_discovery.csv'}")
    print(f"Wrote {RAW_DIR / 'event_overview.csv'}")


if __name__ == "__main__":
    main()

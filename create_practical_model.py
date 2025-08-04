#!/usr/bin/env python3
"""
Train two models (Random-Forest & LinearRegression) to forecast RSVP counts,
engineer features automatically, and write:

• rf_model.pkl / lr_model.pkl
• model_metadata.json  (single source of truth for feature_cols + target + version)
"""
from __future__ import annotations

import json
import logging
import pickle
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

CSV_PATH = Path("historical_rsvp_data.csv")
RF_MODEL_PATH = Path("rf_model.pkl")
LR_MODEL_PATH = Path("lr_model.pkl")
METADATA_PATH = Path("model_metadata.json")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data prep                                                                   #
# --------------------------------------------------------------------------- #
def load_and_prepare(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Load CSV, validate required columns, create engineered features."""
    required = {"ds", "y", "RegisteredCount", "WeatherTemperature", "SunsetTime"}
    df = pd.read_csv(csv_path)

    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # --- basic cleaning ---------------------------------------------------- #
    df = df.dropna(subset=list(required))
    df["SunsetHour"] = df["SunsetTime"].str.split(":").str[0].astype(int)

    ts = pd.to_datetime(df["ds"], format="%Y-%m-%d", errors="coerce")
    df["EventMonth"] = ts.dt.month.astype(int)
    df["EventWeekday"] = ts.dt.weekday.astype(int)

    # --- optional categorical features ------------------------------------- #
    if "WeatherType" in df.columns:
        df = pd.get_dummies(df, columns=["WeatherType"], drop_first=True)

    if "SpecialEvent" in df.columns:
        df["is_special"] = df["SpecialEvent"].notna().astype(int)

    return df


# --------------------------------------------------------------------------- #
# Training + metadata                                                         #
# --------------------------------------------------------------------------- #
def create_practical_model(csv_path: Path = CSV_PATH) -> None:
    df = load_and_prepare(csv_path)

    # Use *every* numeric column except the target y
    features: list[str] = [
        c for c in df.select_dtypes(include=["int64", "float64"]).columns if c != "y"
    ]
    X, y = df[features], df["y"]

    logger.info("Training Random-Forest on %d samples / %d features", len(df), len(features))
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X, y)

    logger.info("Training LinearRegression")
    lr = LinearRegression()
    lr.fit(X, y)

    # persist
    RF_MODEL_PATH.write_bytes(pickle.dumps(rf))
    LR_MODEL_PATH.write_bytes(pickle.dumps(lr))

    metadata = {
        "feature_cols": features,        #  <- single canonical key
        "target": "y",
        "model_version": date.today().isoformat(),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    logger.info("Saved models + metadata (version %s)", metadata["model_version"])


if __name__ == "__main__":
    create_practical_model()

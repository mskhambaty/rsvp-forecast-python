#!/usr/bin/env python3
"""
Train Random-Forest & Linear Regression models to forecast RSVP counts,
then write:

• rf_model.pkl
• lr_model.pkl
• model_metadata.json   (feature_cols, target, model_version)

The script auto-discovers all numeric predictors (plus engineered
categoricals) so the API and model always stay in sync.
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

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
CSV_PATH = Path("historical_rsvp_data.csv")
RF_MODEL_PATH = Path("rf_model.pkl")
LR_MODEL_PATH = Path("lr_model.pkl")
METADATA_PATH = Path("model_metadata.json")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Data preparation                                                            #
# --------------------------------------------------------------------------- #
def load_and_prepare(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Load CSV, validate essentials, and create engineered features."""
    required = {
        "ds",                # event date (MM/DD/YY)
        "y",                 # actual attendance
        "RegisteredCount",
        "WeatherTemperature",
        "SunsetTime",
    }

    df = pd.read_csv(csv_path)
    if missing := required - set(df.columns):
        raise KeyError(f"Missing required columns: {missing}")

    # Drop any row still missing critical data
    df = df.dropna(subset=list(required))

    # 1) SunsetHour ---------------------------------------------------------- #
    df["SunsetHour"] = (
        pd.to_numeric(df["SunsetTime"].str.split(":").str[0], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # 2) EventMonth / EventWeekday ------------------------------------------ #
    # NEW DATE FORMAT: MM/DD/YY  ➜  %m/%d/%y
    ts = pd.to_datetime(df["ds"], format="%m/%d/%y", errors="coerce")
    df["EventMonth"]   = ts.dt.month.fillna(0).astype(int)
    df["EventWeekday"] = ts.dt.weekday.fillna(0).astype(int)

    # 3) Optional categorical features -------------------------------------- #
    if "WeatherType" in df.columns:
        df = pd.get_dummies(df, columns=["WeatherType"], drop_first=True)
    if "SpecialEvent" in df.columns:
        df["is_special"] = df["SpecialEvent"].notna().astype(int)

    return df


# --------------------------------------------------------------------------- #
# Model training + metadata                                                   #
# --------------------------------------------------------------------------- #
def create_practical_model(csv_path: Path = CSV_PATH) -> None:
    df = load_and_prepare(csv_path)

    # Use every numeric predictor except the target y
    features: list[str] = [
        c for c in df.select_dtypes(include=["int64", "float64"]).columns if c != "y"
    ]
    X, y = df[features], df["y"]

    logger.info(
        "Training models on %d rows with %d features", len(df), len(features)
    )

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X, y)

    lr = LinearRegression()
    lr.fit(X, y)

    # Persist models
    RF_MODEL_PATH.write_bytes(pickle.dumps(rf))
    LR_MODEL_PATH.write_bytes(pickle.dumps(lr))

    # Metadata keeps API & model aligned
    metadata = {
        "feature_cols": features,           # canonical key
        "target": "y",
        "model_version": date.today().isoformat(),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    logger.info("Saved models and metadata (version %s)", metadata["model_version"])


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    create_practical_model()

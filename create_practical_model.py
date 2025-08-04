#!/usr/bin/env python3
"""
Train models and write rf_model.pkl / lr_model.pkl + model_metadata.json
"""
from __future__ import annotations

import json, logging, pickle
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

CSV_PATH        = Path("historical_rsvp_data.csv")
RF_MODEL_PATH   = Path("rf_model.pkl")
LR_MODEL_PATH   = Path("lr_model.pkl")
METADATA_PATH   = Path("model_metadata.json")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
def load_and_prepare(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    required = {"ds", "y", "RegisteredCount", "WeatherTemperature", "SunsetTime"}
    df = pd.read_csv(csv_path)
    if missing := required - set(df.columns):
        raise KeyError(f"Missing required columns: {missing}")

    df = df.dropna(subset=list(required))
    # SunsetHour                                                         🟢 FIX
    df["SunsetHour"] = (
        pd.to_numeric(df["SunsetTime"].str.split(":").str[0], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    ts = pd.to_datetime(df["ds"], errors="coerce")
    df["EventMonth"]   = ts.dt.month.fillna(0).astype(int)      # 🟢 FIX
    df["EventWeekday"] = ts.dt.weekday.fillna(0).astype(int)    # 🟢 FIX

    # Optional categoricals
    if "WeatherType" in df.columns:
        df = pd.get_dummies(df, columns=["WeatherType"], drop_first=True)
    if "SpecialEvent" in df.columns:
        df["is_special"] = df["SpecialEvent"].notna().astype(int)

    return df

# --------------------------------------------------------------------------- #
def create_practical_model(csv_path: Path = CSV_PATH) -> None:
    df = load_and_prepare(csv_path)
    features = [c for c in df.select_dtypes(include=["int64", "float64"]).columns if c != "y"]
    X, y = df[features], df["y"]

    logger.info("Training RF & LR on %d samples / %d features", len(df), len(features))
    rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X, y)
    lr = LinearRegression().fit(X, y)

    RF_MODEL_PATH.write_bytes(pickle.dumps(rf))
    LR_MODEL_PATH.write_bytes(pickle.dumps(lr))
    METADATA_PATH.write_text(json.dumps(
        {"feature_cols": features, "target": "y", "model_version": date.today().isoformat()},
        indent=2
    ))

    logger.info("Saved models + metadata (version %s)", date.today().isoformat())

if __name__ == "__main__":
    create_practical_model()

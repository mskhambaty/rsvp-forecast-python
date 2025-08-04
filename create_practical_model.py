#!/usr/bin/env python3
"""
Train RandomForest + LinearRegression on historic RSVP-event data.
Relies on 'historical_rsvp_data.csv' in project root (no `data/` folder).
"""

import os, json, pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_X_y

# === CONFIG ===
CSV_PATH = os.getenv("RSVP_CSV_PATH", "historical_rsvp_data.csv")
FEATURES = [
    "registered_count",
    "days_to_event",
    "venue_good",
    "venue_bad",
    "special_event",
    # add any other features here, matching the REST API name in main.py
]
Y_COL = os.getenv("APPROXACTUAL_COL", "ActualCount")

def load_and_prepare():
    # Load CSV, but if it's missing, error with clear guidance
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"\nERROR: CSV_PATH '{CSV_PATH}' not found. "
                                f"Expected your data file to exist in project root.\n")

    print(f"Loading events from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    for col in (Y_COL, "RegisteredCount"):
        if col not in df.columns:
            raise KeyError(f"ERROR: Required column '{col}' not found in CSV.")

    df = df.dropna(subset=[Y_COL])
    print(f"Training on {len(df)} rows after dropping missing {Y_COL}")

    df = df.rename(columns={
        "RegisteredCount": "registered_count",
        Y_COL: "y",
        # map other columns if necessary:
        "VenueIsGood": "venue_good",
        "VenueIsBad": "venue_bad",
        "SpecialEvent": "special_event",
    })

    return df

def create_practical_model():
    df = load_and_prepare()
    X = df[FEATURES]
    y = df["y"]

    X_checked, y_checked = check_X_y(X.values, y.values)
    print("✅ Training features and target validated — no NaN")

    rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                               random_state=42, n_jobs=-1)
    rf.fit(X_checked, y_checked)
    print("  RF CV score:", cross_val_score(rf, X_checked, y_checked, cv=5).mean())

    lr = LinearRegression()
    lr.fit(X_checked, y_checked)
    print("  LR  CV score:", cross_val_score(lr, X_checked, y_checked, cv=5).mean())

    metadata = {
        "feature_order": FEATURES,
        "y_column": Y_COL,
        "training_samples": len(df),
    }

    with open("rf_model.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open("lr_model.pkl", "wb") as f:
        pickle.dump(lr, f)
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("✅ Models & metadata saved successfully.")

if __name__ == "__main__":
    print("\n=== CREATING PRACTICAL FORECASTING MODELS ===")
    create_practical_model()

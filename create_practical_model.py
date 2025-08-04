#!/usr/bin/env python3
"""
Train RandomForest + LinearRegression on historic RSVP‑event data.
Requires CSV with columns:
- RegisteredCount : num RSVPs (this is your predictor)
- ActualCount ‑ or maybe y, final attendance (this is your target)
- Date, Venue, EventType, etc (for other features)

Writes out files: model_rf.pkl, model_linreg.pkl, features.json
"""

import os
import json
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_X_y

# === Configuration ===
CSV_PATH = os.getenv("RSVP_CSV_PATH", "data/rsvp_data.csv")
FEATURES = [
    # IMPORTANT: RegisteredCount (RSVPs), must add this first
    'registered_count',
    'days_to_event',
    'venue_good',
    'venue_bad',
    'special_event',
    # …add any other engineered features you already use
]
Y_COL = os.getenv("APPROXACTUAL_COL", "ActualCount")  # rename if yours differs

def load_and_prepare():
    df = pd.read_csv(CSV_PATH, parse_dates=['Date'])
    if Y_COL not in df.columns or 'RegisteredCount' not in df.columns:
        msg = f"ERROR: `{Y_COL}` or `RegisteredCount` column missing in CSV!"
        print(msg)
        raise KeyError(msg)

    # drop rows where the target (actual attendance) is missing
    df = df.dropna(subset=[Y_COL])
    print(f"Training on {len(df)} events after dropping missing {Y_COL}")

    # rename columns to match your feature names
    df = df.rename(columns={'RegisteredCount': 'registered_count',
                             Y_COL: 'y',
                             'VenueIsGood': 'venue_good',
                             'VenueIsBad': 'venue_bad',
                             'Special': 'special_event'})
    return df

def create_practical_model():
    df = load_and_prepare()

    # Build feature matrix X and target y
    X = df[FEATURES]
    y = df['y']

    # This call will check for NaN in y/X and fail early if still present
    X_checked, y_checked = check_X_y(X.values, y.values)
    print("  Input shape ✓, no NaNs in y")

    # Train Random Forest
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_checked, y_checked)
    print("  RF trained — cv‑score:", cross_val_score(rf, X_checked, y_checked, cv=5).mean())

    # Train Linear Regression
    lr = LinearRegression()
    lr.fit(X_checked, y_checked)
    print("  LinearReg trained — cv‑score:", cross_val_score(lr, X_checked, y_checked, cv=5).mean())

    metadata = {
        'feature_order': FEATURES,
        'y_column': Y_COL,
        'n_samples': len(df),
        'columns_available': list(df.columns),
    }

    with open('model_rf.pkl', 'wb') as f:
        pickle.dump(rf, f)
    with open('model_linreg.pkl', 'wb') as f:
        pickle.dump(lr, f)
    with open('features.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("Models & metadata saved")

if __name__ == '__main__':
    print("=== CREATING PRACTICAL FORECASTING MODELS ===")
    create_practical_model()

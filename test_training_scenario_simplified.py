#!/usr/bin/env python3
"""
Test simplified model with exact training scenario
"""
import pandas as pd
from prophet.serialize import model_from_json
import json

def test_training_scenario():
    """Test with a scenario that should match training data"""
    print("=== TESTING TRAINING SCENARIO WITH SIMPLIFIED MODEL ===")
    
    # Load simplified model
    with open("serialized_model_simplified.json", "r") as fin:
        model = model_from_json(fin.read())
    
    with open("model_columns_simplified.json", "r") as fin:
        regressor_columns = json.load(fin)
    
    # Load original training data to understand what we should match
    data = pd.read_csv('historical_rsvp_data.csv')
    print(f"Training data sample:")
    print(data[['ds', 'y', 'RegisteredCount', 'WeatherTemperature', 'SunsetTime', 'EventName']].head())
    
    # Try to match March 1 data: 880 RSVP, 744 registered, 28°F, 18:42 sunset, Sherullah event
    test_date = pd.to_datetime('2025-03-01')  # March 1, 2025 (Saturday)
    pred_data = {'ds': test_date}
    
    # Initialize all regressor columns to 0
    for col in regressor_columns:
        pred_data[col] = 0
    
    # Set features to match March 1 training data
    pred_data['RegisteredCount_reg'] = 744  # Exact from training
    pred_data['WeatherType_reg'] = 0  # Clear (no rain in training)
    pred_data['SpecialEvent_reg'] = 0  # Not special (NaN in training)
    pred_data['SunsetMinutes_reg'] = 18 * 60 + 42  # 18:42 = 1122 minutes
    
    # Categorical features based on training data
    pred_data['TempCategory_Cold'] = 1  # 28°F is Cold
    pred_data['EventType_Sherullah'] = 1  # "Sherullah Raat - 3/1"
    pred_data['DayOfWeek_Saturday'] = 1  # March 1, 2025 is Saturday
    pred_data['SunsetCategory_Early'] = 1  # 18:42 is Early
    
    print(f"\nActive features (matching training data):")
    for k, v in pred_data.items():
        if v != 0 and k != 'ds':
            print(f"  {k}: {v}")
    
    # Make prediction
    prediction_df = pd.DataFrame([pred_data])
    forecast = model.predict(prediction_df)
    predicted = max(int(round(forecast['yhat'].values[0])), 0)
    
    print(f"\nResults:")
    print(f"Training actual: 880 RSVP")
    print(f"Prediction: {predicted} RSVP")
    print(f"Raw yhat: {forecast['yhat'].values[0]}")
    print(f"Difference: {abs(predicted - 880)}")
    
    if abs(predicted - 880) < 100:
        print("✓ Excellent match with training data!")
        return True
    elif predicted > 0:
        print("✓ Positive prediction, but not exact match")
        return True
    else:
        print("✗ Still getting 0/negative prediction")
        return False

def check_model_baseline():
    """Check what the model predicts with all features set to 0"""
    print(f"\n=== CHECKING MODEL BASELINE ===")
    
    # Load simplified model
    with open("serialized_model_simplified.json", "r") as fin:
        model = model_from_json(fin.read())
    
    with open("model_columns_simplified.json", "r") as fin:
        regressor_columns = json.load(fin)
    
    # Create prediction with all features = 0 (baseline)
    test_date = pd.to_datetime('2025-03-01')
    pred_data = {'ds': test_date}
    
    # Initialize all regressor columns to 0
    for col in regressor_columns:
        pred_data[col] = 0
    
    # Make prediction
    prediction_df = pd.DataFrame([pred_data])
    forecast = model.predict(prediction_df)
    baseline = forecast['yhat'].values[0]
    
    print(f"Baseline prediction (all features = 0): {baseline}")
    
    # Now test with just RegisteredCount
    pred_data['RegisteredCount_reg'] = 500
    prediction_df = pd.DataFrame([pred_data])
    forecast = model.predict(prediction_df)
    with_registered = forecast['yhat'].values[0]
    
    print(f"With 500 registered: {with_registered}")
    print(f"RegisteredCount effect: {with_registered - baseline}")
    
    if baseline < -10000:
        print("⚠ Very negative baseline suggests model training issue")
    elif baseline < 0:
        print("⚠ Negative baseline - model expects positive features to work")
    else:
        print("✓ Positive baseline")

if __name__ == "__main__":
    success = test_training_scenario()
    check_model_baseline()

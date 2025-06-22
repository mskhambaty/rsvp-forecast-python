#!/usr/bin/env python3
"""
Quick test to verify the sunset model works correctly
"""
import pandas as pd
from prophet.serialize import model_from_json
import json

print("=== TESTING SUNSET MODEL ===")

# Load the model
try:
    with open("serialized_model.json", "r") as fin:
        model = model_from_json(fin.read())
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

# Load regressor columns
try:
    with open("model_columns.json", "r") as fin:
        regressor_columns = json.load(fin)
    print(f"✓ Loaded {len(regressor_columns)} regressor columns")
except Exception as e:
    print(f"✗ Error loading columns: {e}")
    exit(1)

# Check for sunset features
sunset_features = [col for col in regressor_columns if 'sunset' in col.lower()]
print(f"✓ Found {len(sunset_features)} sunset features:")
for feature in sunset_features:
    print(f"  - {feature}")

# Create test prediction data
print("\n=== CREATING TEST PREDICTION ===")

# Test data for March 15, 2024 with sunset at 19:30
pred_data = {'ds': pd.to_datetime('2024-03-15')}

# Initialize all regressor columns to 0
for col in regressor_columns:
    pred_data[col] = 0

# Set basic features
pred_data['RegisteredCount_reg'] = 500
pred_data['WeatherType_reg'] = 0  # Clear weather
pred_data['SpecialEvent_reg'] = 1  # Special event

# Set sunset features
sunset_minutes = 19 * 60 + 30  # 19:30 = 1170 minutes
pred_data['SunsetMinutes_reg'] = sunset_minutes
pred_data['SunsetCategory_Normal'] = 1  # 19:30 is Normal category
pred_data['SunsetHour_19'] = 1  # Hour 19

# Set day of week (Thursday for 2024-03-15)
pred_data['DayOfWeek_Friday'] = 1

# Set temperature (use 75 which exists in training)
pred_data['WeatherTemperature_75'] = 1

print(f"Sunset minutes: {sunset_minutes}")
print(f"Active sunset features:")
for col in regressor_columns:
    if 'sunset' in col.lower() and pred_data[col] == 1:
        print(f"  - {col}: {pred_data[col]}")

# Create DataFrame and make prediction
prediction_df = pd.DataFrame([pred_data])
print(f"\nPrediction DataFrame shape: {prediction_df.shape}")

try:
    forecast = model.predict(prediction_df)
    predicted_rsvp = max(int(round(forecast['yhat'].values[0])), 0)
    lower_bound = max(int(round(forecast['yhat_lower'].values[0])), 0)
    upper_bound = max(int(round(forecast['yhat_upper'].values[0])), 0)
    
    print(f"\n✓ PREDICTION SUCCESSFUL!")
    print(f"  Predicted RSVP: {predicted_rsvp}")
    print(f"  Lower bound: {lower_bound}")
    print(f"  Upper bound: {upper_bound}")
    
    if predicted_rsvp > 0:
        print(f"✓ Model is working - non-zero prediction!")
    else:
        print(f"⚠ Warning: Prediction is 0")
        
except Exception as e:
    print(f"✗ Error making prediction: {e}")

print("\n=== TEST COMPLETE ===")

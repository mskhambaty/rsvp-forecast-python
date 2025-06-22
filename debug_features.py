#!/usr/bin/env python3
"""
Debug which features are being set and why prediction might be 0
"""
import pandas as pd
import json
from datetime import datetime

# Load regressor columns
with open("model_columns.json", "r") as fin:
    regressor_columns = json.load(fin)

print("=== DEBUGGING FEATURE SETTING ===")

# Check what day March 15, 2024 is
test_date = pd.to_datetime('2024-03-15')
day_name = test_date.day_name()
print(f"March 15, 2024 is a {day_name}")

# Check available day columns
day_cols = [col for col in regressor_columns if col.startswith("DayOfWeek_")]
print(f"Available day columns: {day_cols}")

# Check if our day exists
day_col = f"DayOfWeek_{day_name}"
if day_col in regressor_columns:
    print(f"✓ {day_col} exists in model")
else:
    print(f"✗ {day_col} NOT found in model")

# Check temperature columns
temp_cols = [col for col in regressor_columns if col.startswith("WeatherTemperature_")]
print(f"Available temperatures: {[col.split('_')[1] for col in temp_cols]}")

# Check event columns
event_cols = [col for col in regressor_columns if col.startswith("EventName_")]
print(f"Available events: {len(event_cols)} total")

# Let's try to replicate a prediction similar to training data
print(f"\n=== TRYING TRAINING-LIKE DATA ===")

# Use data similar to training (March 1st was in training)
pred_data = {'ds': pd.to_datetime('2024-03-01')}

# Initialize all regressor columns to 0
for col in regressor_columns:
    pred_data[col] = 0

# Set features similar to training data
pred_data['RegisteredCount_reg'] = 744  # Similar to March 1 training data
pred_data['WeatherType_reg'] = 0  # Clear weather
pred_data['SpecialEvent_reg'] = 0  # Not special

# Set day of week for March 1, 2024
march1_day = pd.to_datetime('2024-03-01').day_name()
march1_day_col = f"DayOfWeek_{march1_day}"
print(f"March 1, 2024 is a {march1_day}")
if march1_day_col in pred_data:
    pred_data[march1_day_col] = 1
    print(f"✓ Set {march1_day_col}")

# Set temperature from training (28 was used for March 1)
temp_col = "WeatherTemperature_28"
if temp_col in pred_data:
    pred_data[temp_col] = 1
    print(f"✓ Set {temp_col}")

# Set sunset features (18:42 for March 1 in training)
sunset_minutes = 18 * 60 + 42  # 18:42 = 1122 minutes
pred_data['SunsetMinutes_reg'] = sunset_minutes
pred_data['SunsetCategory_Early'] = 1  # 18:42 is Early
pred_data['SunsetHour_18'] = 1
print(f"✓ Set sunset features for 18:42 ({sunset_minutes} minutes)")

# Try to use a known event name
known_event_col = "EventName_Sherullah Raat - 3/1"
if known_event_col in pred_data:
    pred_data[known_event_col] = 1
    print(f"✓ Set {known_event_col}")

# Count active features
active_features = [k for k, v in pred_data.items() if v != 0 and k != 'ds']
print(f"\nActive features ({len(active_features)}):")
for feature in active_features:
    print(f"  {feature}: {pred_data[feature]}")

# Test prediction
from prophet.serialize import model_from_json

with open("serialized_model.json", "r") as fin:
    model = model_from_json(fin.read())

prediction_df = pd.DataFrame([pred_data])
forecast = model.predict(prediction_df)
predicted_rsvp = max(int(round(forecast['yhat'].values[0])), 0)

print(f"\nPrediction result: {predicted_rsvp}")
print(f"Raw yhat: {forecast['yhat'].values[0]}")

if predicted_rsvp > 0:
    print("✓ SUCCESS: Non-zero prediction!")
else:
    print("✗ Still getting 0 prediction")
    print("This suggests the model might need more investigation")

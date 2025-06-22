#!/usr/bin/env python3
"""
Test prediction with 2023 date (within training range)
"""
import pandas as pd
from prophet.serialize import model_from_json
import json

# Load model and columns
with open("serialized_model.json", "r") as fin:
    model = model_from_json(fin.read())

with open("model_columns.json", "r") as fin:
    regressor_columns = json.load(fin)

print("=== TESTING WITH 2023 DATE ===")

# Use a date within the training range (2023)
test_date = pd.to_datetime('2023-04-15')  # April 15, 2023
print(f"Testing with date: {test_date}")

pred_data = {'ds': test_date}

# Initialize all regressor columns to 0
for col in regressor_columns:
    pred_data[col] = 0

# Set reasonable features
pred_data['RegisteredCount_reg'] = 500  # Reasonable registered count
pred_data['WeatherType_reg'] = 0  # Clear weather
pred_data['SpecialEvent_reg'] = 0  # Not special

# Set day of week
day_name = test_date.day_name()
day_col = f"DayOfWeek_{day_name}"
if day_col in pred_data:
    pred_data[day_col] = 1
    print(f"✓ Set {day_col}")

# Set temperature (use a common one from training)
temp_col = "WeatherTemperature_45"
if temp_col in pred_data:
    pred_data[temp_col] = 1
    print(f"✓ Set {temp_col}")

# Set sunset features for April (around 19:30)
sunset_minutes = 19 * 60 + 30  # 19:30 = 1170 minutes
pred_data['SunsetMinutes_reg'] = sunset_minutes
pred_data['SunsetCategory_Normal'] = 1  # 19:30 is Normal
pred_data['SunsetHour_19'] = 1
print(f"✓ Set sunset features for 19:30 ({sunset_minutes} minutes)")

# Count active features
active_features = [k for k, v in pred_data.items() if v != 0 and k != 'ds']
print(f"\nActive features ({len(active_features)}):")
for feature in active_features:
    print(f"  {feature}: {pred_data[feature]}")

# Make prediction
prediction_df = pd.DataFrame([pred_data])
forecast = model.predict(prediction_df)
predicted_rsvp = max(int(round(forecast['yhat'].values[0])), 0)

print(f"\nPrediction results:")
print(f"  Raw yhat: {forecast['yhat'].values[0]:.2f}")
print(f"  Clipped prediction: {predicted_rsvp}")
print(f"  Lower bound: {forecast['yhat_lower'].values[0]:.2f}")
print(f"  Upper bound: {forecast['yhat_upper'].values[0]:.2f}")

if predicted_rsvp > 0:
    print("✓ SUCCESS: Positive prediction!")
else:
    print("✗ Still negative prediction")

# Try with higher registered count
print(f"\n=== TRYING WITH HIGHER REGISTERED COUNT ===")
pred_data['RegisteredCount_reg'] = 800  # Higher registered count
prediction_df = pd.DataFrame([pred_data])
forecast = model.predict(prediction_df)
predicted_rsvp = max(int(round(forecast['yhat'].values[0])), 0)

print(f"With 800 registered:")
print(f"  Raw yhat: {forecast['yhat'].values[0]:.2f}")
print(f"  Clipped prediction: {predicted_rsvp}")

if predicted_rsvp > 0:
    print("✓ SUCCESS with higher registered count!")
else:
    print("✗ Still negative even with higher registered count")

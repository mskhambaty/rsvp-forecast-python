#!/usr/bin/env python3
"""
Test with exact training data to see if model can reproduce training results
"""
import pandas as pd
from prophet.serialize import model_from_json
import json

# Load model and columns
with open("serialized_model.json", "r") as fin:
    model = model_from_json(fin.read())

with open("model_columns.json", "r") as fin:
    regressor_columns = json.load(fin)

print("=== TESTING WITH EXACT TRAINING DATA ===")

# Use exact data from training (March 1, 2023)
# From training: 880 RSVP, 744 registered, 18:42 sunset, not special
test_date = pd.to_datetime('2023-03-01')
print(f"Testing with exact training date: {test_date}")

pred_data = {'ds': test_date}

# Initialize all regressor columns to 0
for col in regressor_columns:
    pred_data[col] = 0

# Set exact features from training data
pred_data['RegisteredCount_reg'] = 744  # Exact from training
pred_data['WeatherType_reg'] = 0  # Clear weather (no rain in training)
pred_data['SpecialEvent_reg'] = 0  # Not special (NaN in training = 0)

# Set day of week (March 1, 2023 was a Wednesday)
day_name = test_date.day_name()
day_col = f"DayOfWeek_{day_name}"
if day_col in pred_data:
    pred_data[day_col] = 1
    print(f"✓ Set {day_col}")

# Set exact temperature from training (28)
temp_col = "WeatherTemperature_28"
if temp_col in pred_data:
    pred_data[temp_col] = 1
    print(f"✓ Set {temp_col}")

# Set exact sunset features from training (18:42)
sunset_minutes = 18 * 60 + 42  # 18:42 = 1122 minutes (exact from training)
pred_data['SunsetMinutes_reg'] = sunset_minutes
pred_data['SunsetCategory_Early'] = 1  # 18:42 is Early
pred_data['SunsetHour_18'] = 1
print(f"✓ Set sunset features for 18:42 ({sunset_minutes} minutes)")

# Try to use exact event name from training
event_col = "EventName_Sherullah Raat - 3/1"
if event_col in pred_data:
    pred_data[event_col] = 1
    print(f"✓ Set {event_col}")

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
print(f"  Training actual: 880")
print(f"  Difference: {predicted_rsvp - 880}")

if predicted_rsvp > 0:
    print("✓ SUCCESS: Positive prediction!")
    if abs(predicted_rsvp - 880) < 200:
        print("✓ Prediction is reasonably close to training data!")
    else:
        print("⚠ Prediction differs significantly from training data")
else:
    print("✗ Still negative prediction even with exact training data")
    print("This suggests a fundamental issue with the model")

# Let's also try without sunset features to isolate the issue
print(f"\n=== TESTING WITHOUT SUNSET FEATURES ===")
pred_data_no_sunset = pred_data.copy()
pred_data_no_sunset['SunsetMinutes_reg'] = 0
pred_data_no_sunset['SunsetCategory_Early'] = 0
pred_data_no_sunset['SunsetHour_18'] = 0

prediction_df_no_sunset = pd.DataFrame([pred_data_no_sunset])
forecast_no_sunset = model.predict(prediction_df_no_sunset)
predicted_no_sunset = max(int(round(forecast_no_sunset['yhat'].values[0])), 0)

print(f"Without sunset features:")
print(f"  Raw yhat: {forecast_no_sunset['yhat'].values[0]:.2f}")
print(f"  Clipped prediction: {predicted_no_sunset}")

if predicted_no_sunset > predicted_rsvp:
    print("⚠ Removing sunset features improves prediction - sunset features may be problematic")

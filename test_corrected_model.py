#!/usr/bin/env python3
"""
Test the corrected model with 2025 dates and realistic ratio expectations
"""
import pandas as pd
from prophet.serialize import model_from_json
import json

# Load model and columns
with open("serialized_model.json", "r") as fin:
    model = model_from_json(fin.read())

with open("model_columns.json", "r") as fin:
    regressor_columns = json.load(fin)

print("=== TESTING CORRECTED MODEL (2025 DATES) ===")

# Test 1: Exact training data reproduction (March 1, 2025)
print("\n--- Test 1: Exact Training Data (March 1, 2025) ---")
test_date = pd.to_datetime('2025-03-01')
pred_data = {'ds': test_date}

# Initialize all regressor columns to 0
for col in regressor_columns:
    pred_data[col] = 0

# Set exact features from training data (March 1: 880 RSVP, 744 registered)
pred_data['RegisteredCount_reg'] = 744
pred_data['WeatherType_reg'] = 0  # Clear
pred_data['SpecialEvent_reg'] = 0  # Not special

# Set day of week
day_name = test_date.day_name()
day_col = f"DayOfWeek_{day_name}"
if day_col in pred_data:
    pred_data[day_col] = 1

# Set temperature (28 from training)
if "WeatherTemperature_28" in pred_data:
    pred_data["WeatherTemperature_28"] = 1

# Set sunset features (18:42 from training)
sunset_minutes = 18 * 60 + 42  # 1122 minutes
pred_data['SunsetMinutes_reg'] = sunset_minutes
pred_data['SunsetCategory_Early'] = 1
pred_data['SunsetHour_18'] = 1

# Set event name
if "EventName_Sherullah Raat - 3/1" in pred_data:
    pred_data["EventName_Sherullah Raat - 3/1"] = 1

# Make prediction
prediction_df = pd.DataFrame([pred_data])
forecast = model.predict(prediction_df)
predicted_rsvp = max(int(round(forecast['yhat'].values[0])), 0)

print(f"Training data: 880 RSVP, 744 Registered (ratio: {880/744:.3f})")
print(f"Prediction: {predicted_rsvp} RSVP (ratio: {predicted_rsvp/744:.3f})")
print(f"Difference: {predicted_rsvp - 880}")

if abs(predicted_rsvp - 880) < 100:
    print("✓ Excellent reproduction of training data!")
elif abs(predicted_rsvp - 880) < 200:
    print("✓ Good reproduction of training data")
else:
    print("⚠ Prediction differs significantly from training data")

# Test 2: Future prediction with realistic expectations
print("\n--- Test 2: Future Prediction (July 2025) ---")
future_date = pd.to_datetime('2025-07-15')
pred_data_future = {'ds': future_date}

# Initialize all regressor columns to 0
for col in regressor_columns:
    pred_data_future[col] = 0

# Set realistic features for July event
pred_data_future['RegisteredCount_reg'] = 600  # 600 registered
pred_data_future['WeatherType_reg'] = 0  # Clear weather
pred_data_future['SpecialEvent_reg'] = 1  # Special event

# Set day of week
future_day = future_date.day_name()
future_day_col = f"DayOfWeek_{future_day}"
if future_day_col in pred_data_future:
    pred_data_future[future_day_col] = 1

# Set temperature (use 75 which exists in training)
if "WeatherTemperature_75" in pred_data_future:
    pred_data_future["WeatherTemperature_75"] = 1

# Set sunset features for July (around 20:00)
july_sunset_minutes = 20 * 60 + 0  # 20:00 = 1200 minutes
pred_data_future['SunsetMinutes_reg'] = july_sunset_minutes
pred_data_future['SunsetCategory_Late'] = 1
pred_data_future['SunsetHour_20'] = 1

# Make prediction
prediction_df_future = pd.DataFrame([pred_data_future])
forecast_future = model.predict(prediction_df_future)
predicted_future = max(int(round(forecast_future['yhat'].values[0])), 0)

ratio_future = predicted_future / 600
print(f"Future prediction: {predicted_future} RSVP for 600 Registered (ratio: {ratio_future:.3f})")

if 0.7 <= ratio_future <= 1.3:
    print("✓ Realistic ratio - prediction looks good!")
elif 0.5 <= ratio_future <= 1.5:
    print("✓ Acceptable ratio - within reasonable range")
else:
    print("⚠ Ratio seems unrealistic - may need model adjustment")

# Test 3: Different scenarios
print("\n--- Test 3: Scenario Comparison ---")
scenarios = [
    ("Small Event", 200, "Clear", False, "18:45"),
    ("Medium Event", 500, "Clear", False, "19:30"), 
    ("Large Event", 800, "Clear", True, "19:30"),
    ("Rainy Event", 500, "Rain", False, "19:30")
]

for scenario_name, registered, weather, special, sunset in scenarios:
    pred_scenario = {'ds': pd.to_datetime('2025-08-15')}
    
    # Initialize
    for col in regressor_columns:
        pred_scenario[col] = 0
    
    # Set features
    pred_scenario['RegisteredCount_reg'] = registered
    pred_scenario['WeatherType_reg'] = 1 if weather == "Rain" else 0
    pred_scenario['SpecialEvent_reg'] = 1 if special else 0
    
    # Day of week
    if "DayOfWeek_Friday" in pred_scenario:
        pred_scenario["DayOfWeek_Friday"] = 1
    
    # Temperature
    if "WeatherTemperature_75" in pred_scenario:
        pred_scenario["WeatherTemperature_75"] = 1
    
    # Sunset
    sunset_hour = int(sunset.split(':')[0])
    sunset_minute = int(sunset.split(':')[1])
    sunset_mins = sunset_hour * 60 + sunset_minute
    pred_scenario['SunsetMinutes_reg'] = sunset_mins
    
    if sunset_hour < 19:
        pred_scenario['SunsetCategory_Early'] = 1
    elif sunset_hour < 20:
        pred_scenario['SunsetCategory_Normal'] = 1
    else:
        pred_scenario['SunsetCategory_Late'] = 1
        
    sunset_hour_col = f"SunsetHour_{sunset_hour}"
    if sunset_hour_col in pred_scenario:
        pred_scenario[sunset_hour_col] = 1
    
    # Predict
    df_scenario = pd.DataFrame([pred_scenario])
    forecast_scenario = model.predict(df_scenario)
    predicted_scenario = max(int(round(forecast_scenario['yhat'].values[0])), 0)
    ratio_scenario = predicted_scenario / registered
    
    print(f"  {scenario_name}: {predicted_scenario} RSVP / {registered} Reg = {ratio_scenario:.3f}")

print(f"\n=== SUMMARY ===")
print(f"Model now uses 2025 dates and should produce realistic ~1:1 ratios")
print(f"Expected RSVP/Registered ratio should be around 0.95 based on training data")

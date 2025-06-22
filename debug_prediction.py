#!/usr/bin/env python3
"""
Debug script to test prediction logic and identify issues
"""
import pandas as pd
import json
from datetime import datetime

# Load the model columns
with open("model_columns.json", "r") as fin:
    regressor_columns = json.load(fin)

print("=== MODEL COLUMNS DEBUG ===")
print(f"Total regressor columns: {len(regressor_columns)}")
print("\nTemperature columns available:")
temp_cols = [col for col in regressor_columns if col.startswith("WeatherTemperature_")]
print(temp_cols)

print("\nEvent name columns available:")
event_cols = [col for col in regressor_columns if col.startswith("EventName_")]
for col in event_cols:
    print(f"  {col}")

print("\nDay of week columns available:")
day_cols = [col for col in regressor_columns if col.startswith("DayOfWeek_")]
print(day_cols)

# Simulate the prediction logic from main.py
def simulate_prediction_data(event_date_str, registered_count, weather_temperature, weather_type, special_event, event_name):
    """Simulate the prediction data creation logic"""
    
    event_date = pd.to_datetime(event_date_str)
    
    # Create prediction data like in main.py
    pred_data = {'ds': event_date}
    
    # Initialize all known regressor columns to 0
    for col in regressor_columns:
        pred_data[col] = 0
    
    # Fill in the simple numeric regressors
    pred_data['RegisteredCount_reg'] = registered_count
    pred_data['WeatherType_reg'] = 1 if weather_type == "Rain" else 0
    pred_data['SpecialEvent_reg'] = 1 if special_event else 0
    
    # Set the specific one-hot encoded columns to 1
    # DayOfWeek
    day_of_week_col = f"DayOfWeek_{event_date.day_name()}"
    if day_of_week_col in pred_data:
        pred_data[day_of_week_col] = 1
        print(f"✓ Found day column: {day_of_week_col}")
    else:
        print(f"✗ Day column not found: {day_of_week_col}")
        
    # WeatherTemperature
    temp_col = f"WeatherTemperature_{weather_temperature}"
    if temp_col in pred_data:
        pred_data[temp_col] = 1
        print(f"✓ Found temperature column: {temp_col}")
    else:
        print(f"✗ Temperature column not found: {temp_col}")
        print(f"  Available temperatures: {[col.split('_')[1] for col in temp_cols]}")

    # EventName
    event_name_col = f"EventName_{event_name}"
    if event_name_col in pred_data:
        pred_data[event_name_col] = 1
        print(f"✓ Found event column: {event_name_col}")
    else:
        print(f"✗ Event column not found: {event_name_col}")
    
    # Show which columns are set to 1
    active_cols = [k for k, v in pred_data.items() if v == 1 and k != 'ds']
    print(f"\nActive features (set to 1): {active_cols}")
    
    return pred_data

# Test with some example inputs
print("\n=== TESTING PREDICTION SCENARIOS ===")

print("\n--- Test 1: Temperature 75.5 (decimal) ---")
simulate_prediction_data(
    "2024-01-15", 
    500, 
    75.5,  # This will likely fail - no decimal temps in training
    "Clear", 
    False, 
    "Test Event"
)

print("\n--- Test 2: Temperature 75 (integer) ---")
simulate_prediction_data(
    "2024-01-15", 
    500, 
    75,  # This should work
    "Clear", 
    False, 
    "Test Event"
)

print("\n--- Test 3: Known event name ---")
simulate_prediction_data(
    "2024-01-15", 
    500, 
    75, 
    "Clear", 
    False, 
    "Eid-e-Gadheer-e-Khum"  # This exists in training data
)

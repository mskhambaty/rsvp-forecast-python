#!/usr/bin/env python3
"""
Test with corrected feature mapping
"""
import pandas as pd
from prophet.serialize import model_from_json
import json

def test_corrected_prediction():
    """Test with proper feature mapping"""
    print("=== TESTING CORRECTED PREDICTION ===")
    
    # Load simplified model
    with open("serialized_model_simplified.json", "r") as fin:
        model = model_from_json(fin.read())
    
    with open("model_columns_simplified.json", "r") as fin:
        regressor_columns = json.load(fin)
    
    print(f"Available event types: {[col for col in regressor_columns if col.startswith('EventType_')]}")
    
    # Test with proper event type mapping
    test_date = pd.to_datetime('2025-06-26')
    pred_data = {'ds': test_date}
    
    # Initialize all regressor columns to 0
    for col in regressor_columns:
        pred_data[col] = 0
    
    # Set basic features
    pred_data['RegisteredCount_reg'] = 500
    pred_data['WeatherType_reg'] = 0  # Clear
    pred_data['SpecialEvent_reg'] = 0  # Not special
    pred_data['SunsetMinutes_reg'] = 20 * 60 + 25  # 20:25
    
    # Set categorical features correctly
    pred_data['TempCategory_Warm'] = 1  # 75°F is Warm
    pred_data['DayOfWeek_Thursday'] = 1  # June 26, 2025 is Thursday
    pred_data['SunsetCategory_Late'] = 1  # 20:25 is Late
    
    # For event type, since "Other" doesn't exist, use "Educational" as default
    pred_data['EventType_Educational'] = 1
    
    print(f"Active features:")
    for k, v in pred_data.items():
        if v != 0 and k != 'ds':
            print(f"  {k}: {v}")
    
    # Make prediction
    prediction_df = pd.DataFrame([pred_data])
    forecast = model.predict(prediction_df)
    predicted = max(int(round(forecast['yhat'].values[0])), 0)
    
    print(f"Prediction: {predicted}")
    print(f"Raw yhat: {forecast['yhat'].values[0]}")
    print(f"Ratio: {predicted/500:.3f}")
    
    if predicted > 0:
        print("✓ Fixed prediction works!")
        return True
    else:
        print("✗ Still getting 0 prediction")
        return False

def test_all_event_types():
    """Test with different event types to see which work"""
    print(f"\n=== TESTING ALL EVENT TYPES ===")
    
    # Load model
    with open("serialized_model_simplified.json", "r") as fin:
        model = model_from_json(fin.read())
    
    with open("model_columns_simplified.json", "r") as fin:
        regressor_columns = json.load(fin)
    
    event_types = [col.replace('EventType_', '') for col in regressor_columns if col.startswith('EventType_')]
    
    for event_type in event_types:
        test_date = pd.to_datetime('2025-06-26')
        pred_data = {'ds': test_date}
        
        # Initialize all regressor columns to 0
        for col in regressor_columns:
            pred_data[col] = 0
        
        # Set basic features
        pred_data['RegisteredCount_reg'] = 500
        pred_data['WeatherType_reg'] = 0
        pred_data['SpecialEvent_reg'] = 0
        pred_data['SunsetMinutes_reg'] = 20 * 60 + 25
        pred_data['TempCategory_Warm'] = 1
        pred_data['DayOfWeek_Thursday'] = 1
        pred_data['SunsetCategory_Late'] = 1
        
        # Set this event type
        pred_data[f'EventType_{event_type}'] = 1
        
        # Make prediction
        prediction_df = pd.DataFrame([pred_data])
        forecast = model.predict(prediction_df)
        predicted = max(int(round(forecast['yhat'].values[0])), 0)
        ratio = predicted / 500
        
        print(f"EventType_{event_type}: {predicted} RSVP (ratio: {ratio:.3f})")

if __name__ == "__main__":
    success = test_corrected_prediction()
    if not success:
        test_all_event_types()

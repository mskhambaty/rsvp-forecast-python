#!/usr/bin/env python3
"""
Debug the simplified model
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def debug_simplified_model():
    """Debug what's happening with the simplified model"""
    print("=== DEBUGGING SIMPLIFIED MODEL ===")
    
    # Test 1: Check model info
    try:
        response = requests.get(f"{BASE_URL}/model_info")
        if response.status_code == 200:
            data = response.json()
            print("Model info:")
            print(f"  Available temperatures: {data.get('available_temperatures', [])}")
            print(f"  Available events: {data.get('available_events', [])}")
            print(f"  Input schema keys: {list(data.get('input_schema', {}).keys())}")
        else:
            print(f"Model info failed: {response.status_code}")
    except Exception as e:
        print(f"Error getting model info: {e}")
    
    # Test 2: Simple prediction
    print(f"\n--- Simple Prediction Test ---")
    payload = {
        "event_date": "2025-06-26",
        "registered_count": 500,
        "weather_temperature": 75,
        "weather_type": "Clear",
        "special_event": False,
        "event_name": "Community Event",
        "sunset_time": "20:25"
    }
    
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(f"{BASE_URL}/predict_event_rsvp", json=payload)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"Error response: {response.text}")
    except Exception as e:
        print(f"Request error: {e}")
    
    # Test 3: Check if we can access the model directly
    print(f"\n--- Direct Model Test ---")
    try:
        import pandas as pd
        from prophet.serialize import model_from_json
        
        # Load simplified model directly
        with open("serialized_model_simplified.json", "r") as fin:
            model = model_from_json(fin.read())
        
        with open("model_columns_simplified.json", "r") as fin:
            regressor_columns = json.load(fin)
        
        print(f"Direct model loaded successfully")
        print(f"Regressor columns: {regressor_columns}")
        
        # Create test data
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
        
        # Set simplified categorical features
        pred_data['TempCategory_Warm'] = 1  # 75°F is Warm
        pred_data['EventType_Other'] = 1   # Generic event
        pred_data['DayOfWeek_Thursday'] = 1  # June 26, 2025 is Thursday
        pred_data['SunsetCategory_Late'] = 1  # 20:25 is Late
        
        print(f"Active features:")
        for k, v in pred_data.items():
            if v != 0 and k != 'ds':
                print(f"  {k}: {v}")
        
        # Make prediction
        prediction_df = pd.DataFrame([pred_data])
        forecast = model.predict(prediction_df)
        predicted = max(int(round(forecast['yhat'].values[0])), 0)
        
        print(f"Direct prediction: {predicted}")
        print(f"Raw yhat: {forecast['yhat'].values[0]}")
        
        if predicted > 0:
            print("✓ Direct model works!")
        else:
            print("✗ Direct model also returns 0")
            
    except Exception as e:
        print(f"Direct model test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_simplified_model()

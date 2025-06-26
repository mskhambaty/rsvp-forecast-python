#!/usr/bin/env python3
"""
Debug why the model is returning 0 predictions for the exact payloads
"""
import requests
import json
import pandas as pd
import numpy as np
import pickle

# Test payloads from the user
test_payloads = [
    {
        "event_date": "2025-06-26",
        "registered_count": 358,
        "weather_temperature": 93.6,
        "weather_type": "Clear",
        "special_event": True,
        "event_name": "Ashara 1447H - Pehli Tarikh",
        "sunset_time": "20:30"
    },
    {
        "event_date": "2025-06-27",
        "registered_count": 305,
        "weather_temperature": 87.3,
        "weather_type": "Clear",
        "special_event": True,
        "event_name": "Ashara 1447H - 2nd Muharram",
        "sunset_time": "20:30"
    },
    {
        "event_date": "2025-06-28",
        "registered_count": 302,
        "weather_temperature": 88.7,
        "weather_type": "Clear",
        "special_event": True,
        "event_name": "Ashara 1447H - 3rd Muharram",
        "sunset_time": "20:30"
    }
]

def debug_feature_creation():
    """Debug the feature creation process"""
    print("=== DEBUGGING FEATURE CREATION ===")
    
    # Load models and metadata
    try:
        with open("rf_model.pkl", "rb") as f:
            rf_model = pickle.load(f)
        print("✓ Random Forest model loaded")
        
        with open("lr_model.pkl", "rb") as f:
            lr_model = pickle.load(f)
        print("✓ Linear Regression model loaded")
        
        with open("model_metadata.json", "r") as f:
            metadata = json.load(f)
        print("✓ Metadata loaded")
        
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return
    
    print(f"\nModel expects {len(metadata['feature_cols'])} features:")
    for i, col in enumerate(metadata['feature_cols']):
        print(f"  {i+1:2d}. {col}")
    
    # Test each payload
    for i, payload in enumerate(test_payloads, 1):
        print(f"\n=== TESTING PAYLOAD {i} ===")
        print(f"Event: {payload['event_name']}")
        print(f"Date: {payload['event_date']}")
        print(f"Registered: {payload['registered_count']}")
        
        try:
            # Recreate the feature creation logic from main.py
            event_date = pd.to_datetime(payload['event_date'])
            
            # Initialize feature vector
            features = {}
            
            # Basic features
            features['RegisteredCount'] = payload['registered_count']
            features['is_rain'] = 1 if payload['weather_type'].lower() in ['rain', 'rainy'] else 0
            features['is_special'] = 1 if payload['special_event'] else 0
            
            print(f"Basic features:")
            print(f"  RegisteredCount: {features['RegisteredCount']}")
            print(f"  is_rain: {features['is_rain']}")
            print(f"  is_special: {features['is_special']}")
            
            # Temperature features
            temp_normalized = (payload['weather_temperature'] - metadata['temp_stats']['mean']) / metadata['temp_stats']['std']
            features['temp_normalized'] = temp_normalized
            features['temp_cold'] = 1 if payload['weather_temperature'] < 40 else 0
            features['temp_hot'] = 1 if payload['weather_temperature'] > 75 else 0
            
            print(f"Temperature features:")
            print(f"  temp_normalized: {features['temp_normalized']:.3f}")
            print(f"  temp_cold: {features['temp_cold']}")
            print(f"  temp_hot: {features['temp_hot']}")
            
            # Sunset features
            sunset_parts = payload['sunset_time'].split(":")
            sunset_minutes = int(sunset_parts[0]) * 60 + int(sunset_parts[1])
            sunset_normalized = (sunset_minutes - metadata['sunset_stats']['mean']) / metadata['sunset_stats']['std']
            features['sunset_normalized'] = sunset_normalized
            features['sunset_early'] = 1 if sunset_minutes < 1140 else 0  # Before 19:00
            features['sunset_late'] = 1 if sunset_minutes > 1200 else 0   # After 20:00
            
            print(f"Sunset features:")
            print(f"  sunset_minutes: {sunset_minutes}")
            print(f"  sunset_normalized: {features['sunset_normalized']:.3f}")
            print(f"  sunset_early: {features['sunset_early']}")
            print(f"  sunset_late: {features['sunset_late']}")
            
            # Day of week features
            day_name = event_date.day_name()
            for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
                features[f'is_{day.lower()}'] = 1 if day_name == day else 0
            
            print(f"Day of week: {day_name}")
            active_days = [day for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'] 
                          if features[f'is_{day.lower()}'] == 1]
            print(f"  Active day features: {active_days}")
            
            # Event type features
            event_name_lower = payload['event_name'].lower()
            features['is_sherullah'] = 1 if 'sherullah' in event_name_lower else 0
            features['is_eid'] = 1 if 'eid' in event_name_lower else 0
            features['is_urs'] = 1 if 'urs' in event_name_lower else 0
            features['is_milad'] = 1 if 'milad' in event_name_lower else 0
            
            print(f"Event type features:")
            print(f"  is_sherullah: {features['is_sherullah']}")
            print(f"  is_eid: {features['is_eid']}")
            print(f"  is_urs: {features['is_urs']}")
            print(f"  is_milad: {features['is_milad']}")
            
            # Convert to array in correct order
            feature_array = [features.get(col, 0) for col in metadata['feature_cols']]
            
            print(f"\nFeature array (length {len(feature_array)}):")
            for j, (col, val) in enumerate(zip(metadata['feature_cols'], feature_array)):
                if val != 0:  # Only show non-zero features
                    print(f"  {j+1:2d}. {col}: {val}")
            
            # Test predictions
            print(f"\nPredictions:")
            
            # Random Forest
            rf_pred = rf_model.predict([feature_array])[0]
            print(f"  Random Forest: {rf_pred:.1f}")
            
            # Linear Regression
            lr_pred = lr_model.predict([feature_array])[0]
            print(f"  Linear Regression: {lr_pred:.1f}")
            
            # Final prediction (with max(0, round()))
            final_pred = max(int(round(rf_pred)), 0)
            print(f"  Final (max(0, round(RF))): {final_pred}")
            
            # Check for issues
            if final_pred == 0:
                print(f"  ⚠️ ISSUE: Final prediction is 0!")
                if rf_pred < 0:
                    print(f"    - Random Forest predicted negative: {rf_pred:.1f}")
                if abs(rf_pred) < 0.5:
                    print(f"    - Random Forest prediction very small: {rf_pred:.1f}")
            else:
                print(f"  ✓ Prediction looks good: {final_pred}")
                
        except Exception as e:
            print(f"✗ Error processing payload {i}: {e}")
            import traceback
            traceback.print_exc()

def test_api_directly():
    """Test the API directly with the payloads"""
    print(f"\n=== TESTING API DIRECTLY ===")
    
    BASE_URL = "http://localhost:8000"
    
    for i, payload in enumerate(test_payloads, 1):
        print(f"\nTesting payload {i}: {payload['event_name']}")
        
        try:
            response = requests.post(f"{BASE_URL}/predict_event_rsvp", json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                predicted = result.get("predicted_rsvp_count", "N/A")
                lower = result.get("lower_bound", "N/A")
                upper = result.get("upper_bound", "N/A")
                warnings = result.get("warnings", [])
                
                print(f"  ✓ Success: {predicted} RSVP (range: {lower}-{upper})")
                if warnings:
                    print(f"  Warnings: {warnings}")
                
                if predicted == 0:
                    print(f"  ⚠️ ISSUE: API returned 0 prediction!")
                    
            else:
                print(f"  ✗ API Error {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print(f"  ✗ Cannot connect to API at {BASE_URL}")
            print(f"    Make sure the API is running: python main.py")
            break
        except Exception as e:
            print(f"  ✗ Request error: {e}")

if __name__ == "__main__":
    print("🔍 DEBUGGING ZERO PREDICTIONS")
    print("=" * 50)
    
    # First test feature creation locally
    debug_feature_creation()
    
    # Then test API if available
    test_api_directly()
    
    print(f"\n=== SUMMARY ===")
    print("If predictions are 0:")
    print("1. Check if Random Forest is predicting negative values")
    print("2. Check if feature array has correct values")
    print("3. Check if model was trained properly")
    print("4. Check for any data type mismatches")

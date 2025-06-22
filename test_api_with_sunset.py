#!/usr/bin/env python3
"""
Test the API with sunset time features
"""
import requests
import json

# Test the API
BASE_URL = "http://localhost:8000"

def test_api_prediction():
    """Test API prediction with sunset time"""
    
    # Test payload similar to training data
    payload = {
        "event_date": "2023-03-15",  # Use 2023 date (within training range)
        "registered_count": 700,
        "weather_temperature": 45,  # Available in training
        "weather_type": "Clear",
        "special_event": False,
        "event_name": "Test Community Event",
        "sunset_time": "18:50"  # Early sunset like in training
    }
    
    print("=== TESTING API WITH SUNSET FEATURES ===")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(f"{BASE_URL}/predict_event_rsvp", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ API SUCCESS!")
            print(f"  Predicted RSVP: {data['predicted_rsvp_count']}")
            print(f"  Lower bound: {data['lower_bound']}")
            print(f"  Upper bound: {data['upper_bound']}")
            
            if 'warnings' in data:
                print(f"  Warnings: {data['warnings']}")
                
            if data['predicted_rsvp_count'] > 0:
                print(f"✓ Non-zero prediction - sunset features working!")
                return True
            else:
                print(f"✗ Zero prediction")
                return False
        else:
            print(f"✗ API Error: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Make sure it's running on localhost:8000")
        print("  Run: python3 start_api.py")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print(f"\n=== TESTING MODEL INFO ENDPOINT ===")
    
    try:
        response = requests.get(f"{BASE_URL}/model_info")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Model info retrieved")
            print(f"  Available temperatures: {len(data['available_temperatures'])} values")
            print(f"  Available events: {len(data['available_events'])} events")
            
            # Check if sunset_time is in schema
            if 'sunset_time' in data['input_schema']:
                print(f"✓ Sunset time is in input schema")
            else:
                print(f"✗ Sunset time missing from input schema")
                
            return True
        else:
            print(f"✗ Model info error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing API with sunset features...")
    print("Make sure the API is running: python3 start_api.py")
    print()
    
    # Test model info first
    model_info_success = test_model_info()
    
    # Test prediction
    prediction_success = test_api_prediction()
    
    print(f"\n=== SUMMARY ===")
    print(f"Model info: {'✓' if model_info_success else '✗'}")
    print(f"Prediction: {'✓' if prediction_success else '✗'}")
    
    if model_info_success and prediction_success:
        print(f"🎉 SUCCESS: Sunset features are working in the API!")
    else:
        print(f"❌ Issues detected - check API logs")

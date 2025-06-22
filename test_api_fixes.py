#!/usr/bin/env python3
"""
Test script to verify the API fixes work correctly
"""
import requests
import json
from datetime import datetime, timedelta

# API base URL - adjust as needed
BASE_URL = "http://localhost:8000"

def test_model_info():
    """Test the model info endpoint"""
    print("=== Testing /model_info endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}/model_info")
        if response.status_code == 200:
            data = response.json()
            print("✓ Model info retrieved successfully")
            print(f"  Available temperatures: {len(data['available_temperatures'])} values")
            print(f"  Temperature range: {data['temperature_range']}")
            print(f"  Available events: {len(data['available_events'])} events")
            return data
        else:
            print(f"✗ Failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def test_prediction(test_name, payload, expected_success=True):
    """Test a prediction with given payload"""
    print(f"\n=== Testing: {test_name} ===")
    try:
        response = requests.post(f"{BASE_URL}/predict_event_rsvp", json=payload)
        
        if expected_success:
            if response.status_code == 200:
                data = response.json()
                print("✓ Prediction successful")
                print(f"  Predicted RSVP: {data['predicted_rsvp_count']}")
                print(f"  Range: {data['lower_bound']} - {data['upper_bound']}")
                if 'warnings' in data:
                    print(f"  Warnings: {data['warnings']}")
                return data
            else:
                print(f"✗ Failed: {response.status_code} - {response.text}")
                return None
        else:
            if response.status_code != 200:
                print(f"✓ Expected failure: {response.status_code} - {response.text}")
                return response
            else:
                print(f"✗ Expected failure but got success: {response.json()}")
                return None
                
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def run_tests():
    """Run all tests"""
    print("Starting API tests...\n")
    
    # Test 1: Model info
    model_info = test_model_info()
    if not model_info:
        print("Cannot continue without model info")
        return
    
    # Test 2: Valid prediction with decimal temperature
    test_prediction("Decimal temperature handling", {
        "event_date": "2024-03-15",
        "registered_count": 500,
        "weather_temperature": 75.5,  # Should round to 75
        "weather_type": "Clear",
        "special_event": True,
        "event_name": "Test Event",
        "sunset_time": "19:30"
    })
    
    # Test 3: Rain weather type variations
    test_prediction("Rain weather type", {
        "event_date": "2024-03-16",
        "registered_count": 400,
        "weather_temperature": 65,
        "weather_type": "rain",  # Lowercase
        "special_event": False,
        "event_name": "Rainy Day Event",
        "sunset_time": "18:45"
    })

    # Test 4: Known event name
    if model_info and model_info['available_events']:
        known_event = model_info['available_events'][0]
        test_prediction("Known event name", {
            "event_date": "2024-03-17",
            "registered_count": 600,
            "weather_temperature": 70,
            "weather_type": "Clear",
            "special_event": True,
            "event_name": known_event,
            "sunset_time": "19:00"
        })

    # Test 5: Temperature at edge of range
    if model_info:
        min_temp = model_info['temperature_range']['min']
        test_prediction("Minimum temperature", {
            "event_date": "2024-03-18",
            "registered_count": 300,
            "weather_temperature": min_temp - 5,  # Should round to min
            "weather_type": "Clear",
            "special_event": False,
            "event_name": "Cold Weather Event",
            "sunset_time": "20:15"
        })
    
    # Test 6: Invalid date format (should fail)
    test_prediction("Invalid date format", {
        "event_date": "15-03-2024",  # Wrong format
        "registered_count": 500,
        "weather_temperature": 75,
        "weather_type": "Clear",
        "special_event": False,
        "event_name": "Invalid Date Test",
        "sunset_time": "19:30"
    }, expected_success=False)

    # Test 7: Negative registered count (should fail)
    test_prediction("Negative registered count", {
        "event_date": "2024-03-19",
        "registered_count": -100,  # Invalid
        "weather_temperature": 75,
        "weather_type": "Clear",
        "special_event": False,
        "event_name": "Negative Test",
        "sunset_time": "19:30"
    }, expected_success=False)

    # Test 8: Extreme temperature (should fail)
    test_prediction("Extreme temperature", {
        "event_date": "2024-03-20",
        "registered_count": 500,
        "weather_temperature": 200,  # Too high
        "weather_type": "Clear",
        "special_event": False,
        "event_name": "Hot Test",
        "sunset_time": "19:30"
    }, expected_success=False)
    
    print("\n=== Test Summary ===")
    print("Tests completed. Check results above for any failures.")

if __name__ == "__main__":
    run_tests()

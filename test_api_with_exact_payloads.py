#!/usr/bin/env python3
"""
Test API with the exact payloads that are failing
"""
import requests
import json

BASE_URL = "http://localhost:8000"

# Exact payloads from the user
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

def test_api():
    print("=== TESTING API WITH EXACT PAYLOADS ===")
    
    # Test API connection
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API running: {data['message']}")
            print(f"✓ Model loaded: {data['model_loaded']}")
        else:
            print(f"✗ API connection failed: {response.status_code}")
            return
    except:
        print("✗ Cannot connect to API. Make sure it's running on port 8000")
        return
    
    # Test each payload
    for i, payload in enumerate(test_payloads, 1):
        print(f"\n--- Test {i}: {payload['event_name']} ---")
        print(f"Date: {payload['event_date']}")
        print(f"Registered: {payload['registered_count']}")
        print(f"Temperature: {payload['weather_temperature']}°F")
        print(f"Special Event: {payload['special_event']}")
        
        try:
            response = requests.post(f"{BASE_URL}/predict_event_rsvp", json=payload, timeout=10)
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                predicted = result.get("predicted_rsvp_count", "N/A")
                lower = result.get("lower_bound", "N/A")
                upper = result.get("upper_bound", "N/A")
                warnings = result.get("warnings", [])
                
                print(f"✓ Response received")
                print(f"  Predicted RSVP: {predicted}")
                print(f"  Range: {lower} - {upper}")
                print(f"  Ratio: {predicted/payload['registered_count']:.3f}" if predicted != "N/A" and predicted != 0 else "  Ratio: N/A")
                
                if warnings:
                    print(f"  Warnings: {warnings}")
                
                if predicted == 0:
                    print(f"  ⚠️ ISSUE: API returned 0 prediction!")
                    print(f"  Full response: {json.dumps(result, indent=2)}")
                else:
                    print(f"  ✓ Prediction looks good: {predicted}")
                    
            else:
                print(f"✗ API Error {response.status_code}")
                print(f"  Response: {response.text}")
                
        except Exception as e:
            print(f"✗ Request error: {e}")

if __name__ == "__main__":
    test_api()

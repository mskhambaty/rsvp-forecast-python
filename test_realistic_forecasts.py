#!/usr/bin/env python3
"""
Test the API with realistic fake events between June 26 - July 6
"""
import requests
import json
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000"

# Realistic test events based on historical patterns
test_events = [
    {
        "date": "2025-06-26",
        "name": "Community Iftar Gathering",
        "registered": 520,
        "temp": 78,
        "weather": "Clear",
        "special": False,
        "sunset": "20:25",
        "expected_ratio": "~1.0"
    },
    {
        "date": "2025-06-28", 
        "name": "Urs Mubarak Celebration",
        "registered": 680,
        "temp": 82,
        "weather": "Clear", 
        "special": True,
        "sunset": "20:26",
        "expected_ratio": "~1.1 (special event)"
    },
    {
        "date": "2025-06-30",
        "name": "Weekly Community Dinner",
        "registered": 450,
        "temp": 75,
        "weather": "Rain",
        "special": False,
        "sunset": "20:27",
        "expected_ratio": "~0.85 (rain impact)"
    },
    {
        "date": "2025-07-02",
        "name": "Youth Program Night",
        "registered": 320,
        "temp": 79,
        "weather": "Clear",
        "special": False,
        "sunset": "20:27",
        "expected_ratio": "~0.95"
    },
    {
        "date": "2025-07-04",
        "name": "Independence Day Special Event",
        "registered": 750,
        "temp": 85,
        "weather": "Clear",
        "special": True,
        "sunset": "20:27",
        "expected_ratio": "~1.1 (special + holiday)"
    },
    {
        "date": "2025-07-06",
        "name": "Sunday Family Gathering",
        "registered": 600,
        "temp": 77,
        "weather": "Clear",
        "special": False,
        "sunset": "20:26",
        "expected_ratio": "~0.95"
    }
]

def test_api_connection():
    """Test if API is running"""
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✓ API is running")
            return True
        else:
            print(f"✗ API returned {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API")
        print("  Please run: python3 start_api.py")
        return False

def get_model_info():
    """Get model information"""
    try:
        response = requests.get(f"{BASE_URL}/model_info")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"✗ Model info failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"✗ Error getting model info: {e}")
        return None

def predict_event(event):
    """Make prediction for a single event"""
    payload = {
        "event_date": event["date"],
        "registered_count": event["registered"],
        "weather_temperature": event["temp"],
        "weather_type": event["weather"],
        "special_event": event["special"],
        "event_name": event["name"],
        "sunset_time": event["sunset"]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict_event_rsvp", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"✗ Prediction failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return None
    except Exception as e:
        print(f"✗ Error making prediction: {e}")
        return None

def analyze_prediction(event, result):
    """Analyze if prediction is realistic"""
    if not result:
        return False, "No prediction result"
    
    predicted = result["predicted_rsvp_count"]
    registered = event["registered"]
    ratio = predicted / registered if registered > 0 else 0
    
    # Expected ratio ranges based on event type
    if event["special"]:
        expected_min, expected_max = 0.9, 1.3  # Special events can have higher turnout
    elif event["weather"] == "Rain":
        expected_min, expected_max = 0.7, 1.0   # Rain reduces turnout
    else:
        expected_min, expected_max = 0.8, 1.1   # Normal events
    
    is_realistic = expected_min <= ratio <= expected_max
    
    status = "✓" if is_realistic else "⚠"
    analysis = f"{status} {predicted} RSVP / {registered} Reg = {ratio:.3f}"
    
    if not is_realistic:
        if ratio < expected_min:
            analysis += f" (LOW - expected >{expected_min:.2f})"
        else:
            analysis += f" (HIGH - expected <{expected_max:.2f})"
    
    return is_realistic, analysis

def main():
    print("=== TESTING RSVP FORECAST API ===")
    print("Testing realistic events between June 26 - July 6, 2025")
    print()
    
    # Test API connection
    if not test_api_connection():
        return
    
    # Get model info
    print("\n--- Model Information ---")
    model_info = get_model_info()
    if model_info:
        print(f"✓ Available temperatures: {len(model_info['available_temperatures'])} values")
        print(f"✓ Available events: {len(model_info['available_events'])} events")
        print(f"✓ Temperature range: {model_info['temperature_range']['min']}°F - {model_info['temperature_range']['max']}°F")
    
    # Test predictions
    print(f"\n--- Event Predictions ---")
    results = []
    realistic_count = 0
    
    for i, event in enumerate(test_events, 1):
        print(f"\n{i}. {event['name']} ({event['date']})")
        print(f"   Input: {event['registered']} reg, {event['temp']}°F, {event['weather']}, sunset {event['sunset']}")
        print(f"   Expected: {event['expected_ratio']}")
        
        result = predict_event(event)
        if result:
            is_realistic, analysis = analyze_prediction(event, result)
            print(f"   Result: {analysis}")
            
            if 'warnings' in result and result['warnings']:
                print(f"   Warnings: {result['warnings']}")
            
            results.append({
                'event': event,
                'result': result,
                'realistic': is_realistic,
                'ratio': result["predicted_rsvp_count"] / event["registered"]
            })
            
            if is_realistic:
                realistic_count += 1
        else:
            print(f"   Result: ✗ FAILED")
            results.append({
                'event': event,
                'result': None,
                'realistic': False,
                'ratio': 0
            })
    
    # Summary analysis
    print(f"\n=== SUMMARY ANALYSIS ===")
    total_events = len(test_events)
    success_rate = len([r for r in results if r['result']]) / total_events * 100
    realistic_rate = realistic_count / total_events * 100
    
    print(f"API Success Rate: {success_rate:.1f}% ({len([r for r in results if r['result']])}/{total_events})")
    print(f"Realistic Predictions: {realistic_rate:.1f}% ({realistic_count}/{total_events})")
    
    if results:
        ratios = [r['ratio'] for r in results if r['result']]
        if ratios:
            avg_ratio = sum(ratios) / len(ratios)
            print(f"Average RSVP/Registered Ratio: {avg_ratio:.3f}")
            print(f"Historical Expected Ratio: ~0.95")
            
            if 0.8 <= avg_ratio <= 1.2:
                print("✓ Average ratio is realistic!")
            else:
                print("⚠ Average ratio seems off from historical data")
    
    # Detailed breakdown
    print(f"\n--- Event Type Analysis ---")
    special_events = [r for r in results if r['event']['special'] and r['result']]
    normal_events = [r for r in results if not r['event']['special'] and r['result']]
    rainy_events = [r for r in results if r['event']['weather'] == 'Rain' and r['result']]
    
    if special_events:
        special_avg = sum(r['ratio'] for r in special_events) / len(special_events)
        print(f"Special Events Avg Ratio: {special_avg:.3f} (expected: ~1.1)")
    
    if normal_events:
        normal_avg = sum(r['ratio'] for r in normal_events) / len(normal_events)
        print(f"Normal Events Avg Ratio: {normal_avg:.3f} (expected: ~0.95)")
    
    if rainy_events:
        rainy_avg = sum(r['ratio'] for r in rainy_events) / len(rainy_events)
        print(f"Rainy Events Avg Ratio: {rainy_avg:.3f} (expected: ~0.85)")
    
    # Final assessment
    print(f"\n=== FINAL ASSESSMENT ===")
    if success_rate >= 90 and realistic_rate >= 70:
        print("🎉 EXCELLENT: API is working well with realistic predictions!")
    elif success_rate >= 80 and realistic_rate >= 50:
        print("✅ GOOD: API is functional with mostly reasonable predictions")
    elif success_rate >= 70:
        print("⚠️  FAIR: API works but predictions may need adjustment")
    else:
        print("❌ POOR: API has significant issues that need attention")

if __name__ == "__main__":
    main()

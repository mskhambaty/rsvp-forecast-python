#!/usr/bin/env python3
"""
Test the new practical API with realistic scenarios
"""
import requests
import json
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000"

# Test events for late June - early July (your current need)
test_events = [
    {
        "name": "Community Iftar",
        "date": "2025-06-26",
        "registered": 520,
        "temp": 78,
        "weather": "Clear",
        "special": False,
        "sunset": "20:25",
        "expected_range": [450, 550]
    },
    {
        "name": "Urs Mubarak Celebration", 
        "date": "2025-06-28",
        "registered": 680,
        "temp": 82,
        "weather": "Clear",
        "special": True,
        "sunset": "20:26",
        "expected_range": [600, 750]
    },
    {
        "name": "Weekly Community Dinner",
        "date": "2025-06-30",
        "registered": 450,
        "temp": 75,
        "weather": "Rain",
        "special": False,
        "sunset": "20:27",
        "expected_range": [380, 470]
    },
    {
        "name": "Youth Program Night",
        "date": "2025-07-02",
        "registered": 320,
        "temp": 79,
        "weather": "Clear",
        "special": False,
        "sunset": "20:27",
        "expected_range": [280, 350]
    },
    {
        "name": "Independence Day Special",
        "date": "2025-07-04",
        "registered": 750,
        "temp": 85,
        "weather": "Clear",
        "special": True,
        "sunset": "20:27",
        "expected_range": [650, 800]
    },
    {
        "name": "Sunday Family Gathering",
        "date": "2025-07-06",
        "registered": 600,
        "temp": 77,
        "weather": "Clear",
        "special": False,
        "sunset": "20:26",
        "expected_range": [550, 650]
    }
]

def test_practical_api():
    """Test the new practical API"""
    print("=== TESTING PRACTICAL RSVP FORECAST API ===")
    
    # Test API connection
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API running: {data['message']} v{data['version']}")
            print(f"✓ Models loaded: {data['models_loaded']}")
        else:
            print(f"✗ API connection failed: {response.status_code}")
            return
    except:
        print("✗ Cannot connect to API. Make sure it's running on port 8000")
        return
    
    # Get model info
    print(f"\n--- Model Information ---")
    try:
        response = requests.get(f"{BASE_URL}/model_info")
        if response.status_code == 200:
            info = response.json()
            print(f"Model: {info['model_type']}")
            print(f"Training events: {info['training_events']}")
            print(f"Average error: {info['average_error']}")
            print(f"Base ratio: {info['base_attendance_ratio']}")
            
            print(f"\nKey insights:")
            for insight in info['insights']:
                print(f"  • {insight}")
        else:
            print(f"✗ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Model info error: {e}")
    
    # Test predictions
    print(f"\n--- Event Predictions ---")
    results = []
    
    for i, event in enumerate(test_events, 1):
        print(f"\n{i}. {event['name']} ({event['date']})")
        print(f"   Input: {event['registered']} reg, {event['temp']}°F, {event['weather']}, sunset {event['sunset']}")
        
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
            response = requests.post(f"{BASE_URL}/predict", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                primary = result["predictions"]["primary"]
                ratio = result["attendance_ratio"]
                lower = result["confidence_interval"]["lower_bound"]
                upper = result["confidence_interval"]["upper_bound"]
                
                print(f"   Prediction: {primary} attendees (ratio: {ratio:.3f})")
                print(f"   Confidence: {lower} - {upper} people")
                
                # Check if realistic
                expected_min, expected_max = event["expected_range"]
                is_realistic = expected_min <= primary <= expected_max
                status = "✓" if is_realistic else "⚠"
                
                print(f"   Assessment: {status} {'Realistic' if is_realistic else 'Outside expected range'}")
                
                # Show insights
                if result["insights"]:
                    print(f"   Insights: {', '.join(result['insights'])}")
                
                results.append({
                    'event': event['name'],
                    'predicted': primary,
                    'registered': event['registered'],
                    'ratio': ratio,
                    'realistic': is_realistic,
                    'confidence_range': (lower, upper)
                })
                
            else:
                print(f"   ✗ Prediction failed: {response.status_code}")
                print(f"     Error: {response.text}")
                
        except Exception as e:
            print(f"   ✗ Request error: {e}")
    
    # Summary analysis
    print(f"\n=== SUMMARY ANALYSIS ===")
    if results:
        successful = len(results)
        realistic = len([r for r in results if r['realistic']])
        avg_ratio = sum(r['ratio'] for r in results) / len(results)
        
        print(f"Successful predictions: {successful}/{len(test_events)}")
        print(f"Realistic predictions: {realistic}/{successful}")
        print(f"Average attendance ratio: {avg_ratio:.3f}")
        
        print(f"\nPrediction breakdown:")
        for result in results:
            status = "✓" if result['realistic'] else "⚠"
            print(f"  {status} {result['event']}: {result['predicted']} / {result['registered']} = {result['ratio']:.3f}")
        
        if realistic >= successful * 0.8:
            print(f"\n🎉 EXCELLENT: {realistic/successful*100:.0f}% of predictions are realistic!")
        elif realistic >= successful * 0.6:
            print(f"\n✅ GOOD: {realistic/successful*100:.0f}% of predictions are realistic")
        else:
            print(f"\n⚠️ FAIR: Only {realistic/successful*100:.0f}% of predictions are realistic")
        
        # Practical recommendations
        print(f"\n--- Planning Recommendations ---")
        for result in results:
            lower, upper = result['confidence_range']
            print(f"{result['event']}: Plan for {result['predicted']} (range: {lower}-{upper})")
    
    else:
        print("❌ No successful predictions")

if __name__ == "__main__":
    test_practical_api()

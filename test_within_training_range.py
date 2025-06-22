#!/usr/bin/env python3
"""
Test API with dates within the training range (Feb-June 2025)
"""
import requests
import json

BASE_URL = "http://localhost:8000"

# Test events within training range (Feb 27 - June 14, 2025)
test_events = [
    {
        "date": "2025-03-15",
        "name": "Community Sherullah Event",
        "registered": 520,
        "temp": 45,
        "weather": "Clear",
        "special": False,
        "sunset": "19:00",
        "expected_ratio": "~1.0"
    },
    {
        "date": "2025-04-10", 
        "name": "Urs Celebration",
        "registered": 680,
        "temp": 55,
        "weather": "Clear", 
        "special": True,
        "sunset": "19:30",
        "expected_ratio": "~1.1 (special event)"
    },
    {
        "date": "2025-05-05",
        "name": "Educational Program",
        "registered": 450,
        "temp": 65,
        "weather": "Rain",
        "special": False,
        "sunset": "19:45",
        "expected_ratio": "~0.85 (rain impact)"
    },
    {
        "date": "2025-06-01",
        "name": "Eid Celebration",
        "registered": 750,
        "temp": 75,
        "weather": "Clear",
        "special": True,
        "sunset": "20:15",
        "expected_ratio": "~1.1 (special event)"
    }
]

def test_within_training_range():
    """Test predictions within training date range"""
    print("=== TESTING WITHIN TRAINING RANGE (Feb-June 2025) ===")
    
    results = []
    
    for i, event in enumerate(test_events, 1):
        print(f"\n{i}. {event['name']} ({event['date']})")
        print(f"   Input: {event['registered']} reg, {event['temp']}°F, {event['weather']}, sunset {event['sunset']}")
        print(f"   Expected: {event['expected_ratio']}")
        
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
                result = response.json()
                predicted = result["predicted_rsvp_count"]
                registered = event["registered"]
                ratio = predicted / registered if registered > 0 else 0
                
                print(f"   Result: {predicted} RSVP / {registered} Reg = {ratio:.3f}")
                
                if 'warnings' in result and result['warnings']:
                    print(f"   Warnings: {result['warnings']}")
                
                # Check if ratio is realistic
                if event["special"] and event["weather"] != "Rain":
                    expected_min, expected_max = 0.9, 1.3
                elif event["weather"] == "Rain":
                    expected_min, expected_max = 0.7, 1.0
                else:
                    expected_min, expected_max = 0.8, 1.1
                
                is_realistic = expected_min <= ratio <= expected_max
                status = "✓" if is_realistic else "⚠"
                
                if predicted > 0:
                    print(f"   Assessment: {status} {'Realistic' if is_realistic else 'Outside expected range'}")
                else:
                    print(f"   Assessment: ✗ Zero prediction")
                
                results.append({
                    'success': True,
                    'predicted': predicted,
                    'registered': registered,
                    'ratio': ratio,
                    'realistic': is_realistic
                })
            else:
                print(f"   Result: ✗ API Error {response.status_code}: {response.text}")
                results.append({'success': False})
                
        except Exception as e:
            print(f"   Result: ✗ Exception: {e}")
            results.append({'success': False})
    
    # Summary
    print(f"\n=== SUMMARY ===")
    successful = [r for r in results if r.get('success', False)]
    realistic = [r for r in successful if r.get('realistic', False)]
    positive = [r for r in successful if r.get('predicted', 0) > 0]
    
    print(f"Successful predictions: {len(successful)}/{len(results)}")
    print(f"Positive predictions: {len(positive)}/{len(successful) if successful else 1}")
    print(f"Realistic predictions: {len(realistic)}/{len(successful) if successful else 1}")
    
    if positive:
        avg_ratio = sum(r['ratio'] for r in positive) / len(positive)
        print(f"Average ratio (positive predictions): {avg_ratio:.3f}")
        
        if len(positive) == len(successful):
            print("✅ All predictions are positive!")
        else:
            print("⚠️ Some predictions are still zero")
    else:
        print("❌ No positive predictions - model still has issues")
    
    return len(positive) > 0

if __name__ == "__main__":
    success = test_within_training_range()
    
    if success:
        print("\n🎉 SUCCESS: Model works within training range!")
        print("The issue was predicting outside the training date range.")
    else:
        print("\n❌ FAILURE: Model doesn't work even within training range.")
        print("This indicates a fundamental model issue.")

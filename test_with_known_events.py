#!/usr/bin/env python3
"""
Test with known event names from training data
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def get_available_events():
    """Get list of available events from model"""
    try:
        response = requests.get(f"{BASE_URL}/model_info")
        if response.status_code == 200:
            data = response.json()
            return data.get('available_events', [])
        return []
    except:
        return []

def test_with_known_events():
    """Test predictions using known event names"""
    print("=== TESTING WITH KNOWN EVENT NAMES ===")
    
    # Get available events
    available_events = get_available_events()
    print(f"Available events in training data: {len(available_events)}")
    
    if available_events:
        print("Sample available events:")
        for i, event in enumerate(available_events[:5]):
            print(f"  {i+1}. {event}")
    
    # Test with known events
    test_cases = [
        {
            "name": "Test with Sherullah Raat event",
            "payload": {
                "event_date": "2025-06-26",
                "registered_count": 520,
                "weather_temperature": 78,
                "weather_type": "Clear",
                "special_event": False,
                "event_name": "Sherullah Raat - 3/1",  # Known from training
                "sunset_time": "20:25"
            },
            "expected": "Should work - known event"
        },
        {
            "name": "Test with Eid event",
            "payload": {
                "event_date": "2025-06-28",
                "registered_count": 680,
                "weather_temperature": 75,
                "weather_type": "Clear",
                "special_event": True,
                "event_name": "Eid-e-Gadheer-e-Khum",  # Known from training
                "sunset_time": "20:26"
            },
            "expected": "Should work - known special event"
        },
        {
            "name": "Test with Milad event",
            "payload": {
                "event_date": "2025-07-02",
                "registered_count": 450,
                "weather_temperature": 83,
                "weather_type": "Clear",
                "special_event": True,
                "event_name": "Milad Mubarak of Syedna Taher Saifuddin RA Niyaz",  # Known from training
                "sunset_time": "20:27"
            },
            "expected": "Should work - known special event"
        },
        {
            "name": "Test with rainy weather",
            "payload": {
                "event_date": "2025-06-30",
                "registered_count": 400,
                "weather_temperature": 45,
                "weather_type": "Rain",
                "special_event": False,
                "event_name": "Sherullah Raat - 3/5",  # Known rainy event from training
                "sunset_time": "20:27"
            },
            "expected": "Should work - known event with rain"
        }
    ]
    
    print(f"\n--- Testing Known Events ---")
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Event: {test_case['payload']['event_name']}")
        print(f"   Input: {test_case['payload']['registered_count']} reg, {test_case['payload']['weather_temperature']}°F, {test_case['payload']['weather_type']}")
        print(f"   Expected: {test_case['expected']}")
        
        try:
            response = requests.post(f"{BASE_URL}/predict_event_rsvp", json=test_case['payload'])
            
            if response.status_code == 200:
                result = response.json()
                predicted = result["predicted_rsvp_count"]
                registered = test_case['payload']['registered_count']
                ratio = predicted / registered if registered > 0 else 0
                
                print(f"   Result: ✓ {predicted} RSVP / {registered} Reg = {ratio:.3f}")
                
                if 'warnings' in result and result['warnings']:
                    print(f"   Warnings: {result['warnings']}")
                
                # Check if ratio is realistic
                if 0.7 <= ratio <= 1.3:
                    print(f"   Assessment: ✓ Realistic ratio")
                elif predicted == 0:
                    print(f"   Assessment: ✗ Zero prediction - model issue")
                else:
                    print(f"   Assessment: ⚠ Ratio outside expected range")
                
                results.append({
                    'success': True,
                    'predicted': predicted,
                    'registered': registered,
                    'ratio': ratio,
                    'realistic': 0.7 <= ratio <= 1.3
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
    
    print(f"Successful predictions: {len(successful)}/{len(results)}")
    print(f"Realistic predictions: {len(realistic)}/{len(successful) if successful else 1}")
    
    if successful:
        avg_ratio = sum(r['ratio'] for r in successful) / len(successful)
        print(f"Average ratio: {avg_ratio:.3f}")
        
        if len(realistic) == len(successful):
            print("✅ All successful predictions are realistic!")
        elif len(realistic) > len(successful) / 2:
            print("✅ Most predictions are realistic")
        else:
            print("⚠️ Many predictions are unrealistic")
    else:
        print("❌ No successful predictions - major model issue")

if __name__ == "__main__":
    test_with_known_events()

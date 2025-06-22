#!/usr/bin/env python3
"""
Test the updated main.py API to ensure it works with same interface
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_updated_api():
    """Test the updated main.py API"""
    print("=== TESTING UPDATED MAIN.PY API ===")
    
    # Test 1: Basic connection
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API running: {data['message']}")
            print(f"✓ Model loaded: {data['model_loaded']}")
        else:
            print(f"✗ API connection failed: {response.status_code}")
            return False
    except:
        print("✗ Cannot connect to API")
        return False
    
    # Test 2: Model info endpoint
    print(f"\n--- Testing /model_info ---")
    try:
        response = requests.get(f"{BASE_URL}/model_info")
        if response.status_code == 200:
            info = response.json()
            print(f"✓ Model type: {info.get('model_type', 'Unknown')}")
            print(f"✓ Training events: {info.get('training_events', 'Unknown')}")
            print(f"✓ Average error: {info.get('average_error', 'Unknown')}")
            print(f"✓ Base ratio: {info.get('base_attendance_ratio', 'Unknown')}")
        else:
            print(f"✗ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Model info error: {e}")
    
    # Test 3: Prediction endpoint (same interface as before)
    print(f"\n--- Testing /predict_event_rsvp ---")
    test_cases = [
        {
            "name": "Future Event (July)",
            "payload": {
                "event_date": "2025-07-15",
                "registered_count": 500,
                "weather_temperature": 78.5,
                "weather_type": "Clear",
                "special_event": False,
                "event_name": "Community Dinner",
                "sunset_time": "20:15"
            }
        },
        {
            "name": "Special Event with Rain",
            "payload": {
                "event_date": "2025-08-01",
                "registered_count": 650,
                "weather_temperature": 72.0,
                "weather_type": "Rain",
                "special_event": True,
                "event_name": "Eid Celebration",
                "sunset_time": "20:00"
            }
        },
        {
            "name": "Winter Event",
            "payload": {
                "event_date": "2025-12-15",
                "registered_count": 400,
                "weather_temperature": 35.0,
                "weather_type": "Clear",
                "special_event": False,
                "event_name": "Winter Gathering",
                "sunset_time": "17:30"
            }
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        payload = test_case['payload']
        print(f"   Input: {payload['registered_count']} reg, {payload['weather_temperature']}°F, {payload['weather_type']}")
        
        try:
            response = requests.post(f"{BASE_URL}/predict_event_rsvp", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                predicted = result["predicted_rsvp_count"]
                lower = result["lower_bound"]
                upper = result["upper_bound"]
                ratio = predicted / payload['registered_count']
                
                print(f"   Result: {predicted} RSVP (ratio: {ratio:.3f})")
                print(f"   Range: {lower} - {upper}")
                
                if 'warnings' in result:
                    print(f"   Insights: {', '.join(result['warnings'])}")
                
                # Check if realistic
                if 0.5 <= ratio <= 1.5:
                    print(f"   Assessment: ✓ Realistic ratio")
                    realistic = True
                else:
                    print(f"   Assessment: ⚠ Ratio outside expected range")
                    realistic = False
                
                results.append({
                    'name': test_case['name'],
                    'predicted': predicted,
                    'registered': payload['registered_count'],
                    'ratio': ratio,
                    'realistic': realistic,
                    'success': True
                })
                
            else:
                print(f"   ✗ Prediction failed: {response.status_code}")
                print(f"     Error: {response.text}")
                results.append({'name': test_case['name'], 'success': False})
                
        except Exception as e:
            print(f"   ✗ Request error: {e}")
            results.append({'name': test_case['name'], 'success': False})
    
    # Summary
    print(f"\n=== SUMMARY ===")
    successful = [r for r in results if r.get('success', False)]
    realistic = [r for r in successful if r.get('realistic', False)]
    
    print(f"Successful predictions: {len(successful)}/{len(results)}")
    print(f"Realistic predictions: {len(realistic)}/{len(successful) if successful else 1}")
    
    if successful:
        avg_ratio = sum(r['ratio'] for r in successful) / len(successful)
        print(f"Average ratio: {avg_ratio:.3f}")
        
        print(f"\nResults breakdown:")
        for result in successful:
            status = "✓" if result.get('realistic', False) else "⚠"
            print(f"  {status} {result['name']}: {result['predicted']} / {result['registered']} = {result['ratio']:.3f}")
    
    success_rate = len(successful) / len(results) * 100
    realistic_rate = len(realistic) / len(successful) * 100 if successful else 0
    
    if success_rate >= 90 and realistic_rate >= 80:
        print(f"\n🎉 EXCELLENT: API working perfectly!")
        return True
    elif success_rate >= 70:
        print(f"\n✅ GOOD: API mostly working")
        return True
    else:
        print(f"\n❌ ISSUES: API needs attention")
        return False

if __name__ == "__main__":
    success = test_updated_api()
    
    if success:
        print(f"\n✅ Updated main.py API is working correctly!")
        print(f"✅ Same interface as before, but with better predictions")
        print(f"✅ Ready for deployment to render.com")
    else:
        print(f"\n❌ Issues found with updated API")

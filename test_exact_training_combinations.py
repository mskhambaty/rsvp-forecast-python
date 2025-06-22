#!/usr/bin/env python3
"""
Test with exact feature combinations from training data
"""
import requests
import json
import pandas as pd

BASE_URL = "http://localhost:8000"

def test_exact_training_combinations():
    """Test with exact combinations from training data"""
    print("=== TESTING WITH EXACT TRAINING COMBINATIONS ===")
    
    # These are exact combinations from the training data
    training_combinations = [
        {
            "name": "March 1 Training Data",
            "payload": {
                "event_date": "2025-03-01",  # Exact from training
                "registered_count": 744,     # Exact from training
                "weather_temperature": 28,   # Exact from training
                "weather_type": "Clear",     # Exact from training (no rain)
                "special_event": False,      # Exact from training (NaN = False)
                "event_name": "Sherullah Raat - 3/1",  # Exact from training
                "sunset_time": "18:42"       # Exact from training
            },
            "expected_rsvp": 880,  # Actual from training data
            "description": "Exact reproduction of training data"
        },
        {
            "name": "March 8 Training Data (Special)",
            "payload": {
                "event_date": "2025-03-08",
                "registered_count": 939,
                "weather_temperature": 40,
                "weather_type": "Clear",
                "special_event": True,       # Yes in training
                "event_name": "Sherullah Raat - 3/8",
                "sunset_time": "18:50"
            },
            "expected_rsvp": 968,
            "description": "Special event from training data"
        },
        {
            "name": "March 5 Training Data (Rain)",
            "payload": {
                "event_date": "2025-03-05",
                "registered_count": 722,
                "weather_temperature": 43,
                "weather_type": "Rain",      # Rain in training
                "special_event": False,
                "event_name": "Sherullah Raat - 3/5",
                "sunset_time": "18:47"
            },
            "expected_rsvp": 616,
            "description": "Rainy event from training data"
        },
        {
            "name": "June 14 Training Data (Late Sunset)",
            "payload": {
                "event_date": "2025-06-14",
                "registered_count": 475,
                "weather_temperature": 75,
                "weather_type": "Clear",
                "special_event": True,
                "event_name": "Eid-e-Gadheer-e-Khum",
                "sunset_time": "20:28"
            },
            "expected_rsvp": 344,
            "description": "Late sunset event from training data"
        }
    ]
    
    print(f"Testing {len(training_combinations)} exact training combinations...")
    
    results = []
    
    for i, test_case in enumerate(training_combinations, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Description: {test_case['description']}")
        print(f"   Expected RSVP: {test_case['expected_rsvp']}")
        
        try:
            response = requests.post(f"{BASE_URL}/predict_event_rsvp", json=test_case['payload'])
            
            if response.status_code == 200:
                result = response.json()
                predicted = result["predicted_rsvp_count"]
                expected = test_case['expected_rsvp']
                difference = abs(predicted - expected)
                accuracy = (1 - difference / expected) * 100 if expected > 0 else 0
                
                print(f"   Predicted: {predicted} RSVP")
                print(f"   Difference: {difference} ({accuracy:.1f}% accuracy)")
                
                if 'warnings' in result and result['warnings']:
                    print(f"   Warnings: {result['warnings']}")
                
                # Assessment
                if difference <= 50:
                    print(f"   Assessment: ✓ Excellent reproduction")
                elif difference <= 100:
                    print(f"   Assessment: ✓ Good reproduction")
                elif difference <= 200:
                    print(f"   Assessment: ⚠ Fair reproduction")
                else:
                    print(f"   Assessment: ✗ Poor reproduction")
                
                results.append({
                    'success': True,
                    'predicted': predicted,
                    'expected': expected,
                    'difference': difference,
                    'accuracy': accuracy
                })
            else:
                print(f"   Result: ✗ API Error {response.status_code}: {response.text}")
                results.append({'success': False})
                
        except Exception as e:
            print(f"   Result: ✗ Exception: {e}")
            results.append({'success': False})
    
    # Summary analysis
    print(f"\n=== TRAINING DATA REPRODUCTION ANALYSIS ===")
    successful = [r for r in results if r.get('success', False)]
    
    if successful:
        avg_accuracy = sum(r['accuracy'] for r in successful) / len(successful)
        avg_difference = sum(r['difference'] for r in successful) / len(successful)
        
        print(f"Successful predictions: {len(successful)}/{len(results)}")
        print(f"Average accuracy: {avg_accuracy:.1f}%")
        print(f"Average difference: {avg_difference:.0f} RSVP")
        
        excellent = len([r for r in successful if r['difference'] <= 50])
        good = len([r for r in successful if 50 < r['difference'] <= 100])
        fair = len([r for r in successful if 100 < r['difference'] <= 200])
        poor = len([r for r in successful if r['difference'] > 200])
        
        print(f"\nAccuracy breakdown:")
        print(f"  Excellent (≤50 diff): {excellent}")
        print(f"  Good (51-100 diff): {good}")
        print(f"  Fair (101-200 diff): {fair}")
        print(f"  Poor (>200 diff): {poor}")
        
        if avg_accuracy >= 90:
            print("✅ EXCELLENT: Model reproduces training data very well")
        elif avg_accuracy >= 80:
            print("✅ GOOD: Model reproduces training data well")
        elif avg_accuracy >= 70:
            print("⚠️ FAIR: Model has some issues with training data reproduction")
        else:
            print("❌ POOR: Model cannot reproduce its own training data")
            print("   This indicates a fundamental problem with the model")
    else:
        print("❌ CRITICAL: No successful predictions even with exact training data")
        print("   The model is completely broken")

if __name__ == "__main__":
    test_exact_training_combinations()

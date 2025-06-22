#!/usr/bin/env python3
"""
Create a simplified model that generalizes better
"""
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.serialize import model_to_json
import os
import json

print("Starting simplified model creation process...")

try:
    # Load historical data
    print("Loading historical data...")
    data = pd.read_csv('historical_rsvp_data.csv')
    print(f"Loaded {len(data)} rows of data")
    
    # --- Data Preparation ---
    print("Processing data...")
    data['ds'] = pd.to_datetime(data['ds'] + '-2025', format='%d-%b-%Y', errors='coerce')
    data['y'] = pd.to_numeric(data['y'], errors='coerce')
    data.dropna(subset=['ds', 'y'], inplace=True)
    print(f"After initial processing: {len(data)} valid rows")
    
    # Check RSVP to Registered ratios
    data['ratio'] = data['y'] / pd.to_numeric(data['RegisteredCount'], errors='coerce')
    print(f"RSVP/Registered ratio - Mean: {data['ratio'].mean():.3f}, Median: {data['ratio'].median():.3f}")

    # --- Simplified Feature Engineering ---
    print("Engineering simplified features...")
    
    # Core numeric regressors (these generalize well)
    data['RegisteredCount_reg'] = pd.to_numeric(data['RegisteredCount'], errors='coerce')
    data['WeatherType_reg'] = np.where(data['WeatherType'] == 'Rain', 1, 0)
    data['SpecialEvent_reg'] = np.where(data['SpecialEvent'] == 'Yes', 1, 0)
    
    # Sunset features (numeric - generalizes well)
    def process_sunset_time(sunset_str):
        if pd.isna(sunset_str):
            return None
        try:
            hour, minute = map(int, str(sunset_str).split(':'))
            return hour * 60 + minute
        except:
            return None
    
    data['SunsetMinutes_reg'] = data['SunsetTime'].apply(process_sunset_time)
    
    # Simplified categorical features (reduce overfitting)
    # Instead of specific event names, use broader categories
    def categorize_event_type(event_name):
        if pd.isna(event_name):
            return 'Other'
        event_str = str(event_name).lower()
        if 'sherullah' in event_str:
            return 'Sherullah'
        elif 'urs' in event_str:
            return 'Urs'
        elif 'eid' in event_str or 'milad' in event_str:
            return 'Celebration'
        elif 'raat' in event_str or 'daris' in event_str:
            return 'Educational'
        else:
            return 'Other'
    
    data['EventType'] = data['EventName'].apply(categorize_event_type)
    
    # Simplified temperature categories (instead of exact temperatures)
    def categorize_temperature(temp):
        if pd.isna(temp):
            return 'Unknown'
        temp = float(temp)
        if temp < 40:
            return 'Cold'
        elif temp < 60:
            return 'Cool'
        elif temp < 80:
            return 'Warm'
        else:
            return 'Hot'
    
    data['TempCategory'] = data['WeatherTemperature'].apply(categorize_temperature)
    
    # Day of week (keep this as it's useful and not overfitted)
    data['DayOfWeek'] = data['ds'].dt.day_name()
    
    # Simplified sunset categories
    def categorize_sunset_simple(sunset_str):
        if pd.isna(sunset_str):
            return 'Unknown'
        try:
            hour = int(str(sunset_str).split(':')[0])
            if hour < 19:
                return 'Early'
            elif hour < 20:
                return 'Normal'
            else:
                return 'Late'
        except:
            return 'Unknown'
    
    data['SunsetCategory'] = data['SunsetTime'].apply(categorize_sunset_simple)
    
    # One-hot encode simplified categorical features
    categorical_features = ['EventType', 'TempCategory', 'DayOfWeek', 'SunsetCategory']
    for col in categorical_features:
        data[col] = data[col].astype(str)
    
    data_encoded = pd.get_dummies(data, columns=categorical_features, prefix=categorical_features)
    
    # --- Create Final DataFrame for Prophet ---
    final_cols = ['ds', 'y']
    # Add numeric regressors
    final_cols.extend(['RegisteredCount_reg', 'WeatherType_reg', 'SpecialEvent_reg', 'SunsetMinutes_reg'])
    # Add simplified categorical features
    dummy_cols = [col for col in data_encoded.columns if any(col.startswith(prefix + '_') for prefix in categorical_features)]
    final_cols.extend(dummy_cols)
    
    final_df = data_encoded[final_cols].dropna()
    print(f"Simplified model has {len(final_df)} rows and {len(final_df.columns)} columns.")
    
    # Show feature categories
    print(f"\nFeature categories:")
    for prefix in categorical_features:
        cols = [col for col in final_df.columns if col.startswith(prefix + '_')]
        print(f"  {prefix}: {len(cols)} categories")
    
    # --- Save Model Columns ---
    regressor_names = [col for col in final_df.columns if col not in ['ds', 'y']]
    with open('model_columns_simplified.json', 'w') as fout:
        json.dump(regressor_names, fout)
    print(f"Saved {len(regressor_names)} simplified regressor column names")

    # --- Train Prophet Model ---
    print("Creating and training simplified Prophet model...")
    model = Prophet(
        daily_seasonality=False,    # Reduce complexity
        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode='additive'  # Less complex than multiplicative
    )

    for regressor in regressor_names:
        model.add_regressor(regressor)

    model.fit(final_df)
    
    # --- Save Model ---
    print("Serializing simplified model to JSON...")
    with open('serialized_model_simplified.json', 'w') as fout:
        fout.write(model_to_json(model))
    
    file_size = os.path.getsize('serialized_model_simplified.json')
    print(f"Simplified model saved ({file_size} bytes)")
    
    # Test the simplified model with training data
    print("\n--- Testing Simplified Model ---")
    test_row = final_df.iloc[0:1].copy()  # First row
    forecast = model.predict(test_row)
    predicted = max(int(round(forecast['yhat'].values[0])), 0)
    actual = int(test_row['y'].values[0])
    
    print(f"Test prediction: {predicted} vs actual {actual} (diff: {abs(predicted - actual)})")
    
    if abs(predicted - actual) < 100:
        print("✓ Simplified model working correctly")
    else:
        print("⚠ Simplified model may have issues")
    
    print("\nSimplified model creation completed successfully!")
    print("This model should generalize better to new events and scenarios.")

except Exception as e:
    print(f"\nError creating simplified model: {str(e)}")
    import traceback
    traceback.print_exc()

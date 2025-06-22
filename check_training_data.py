#!/usr/bin/env python3
"""
Check the actual training data to understand the issue
"""
import pandas as pd
import numpy as np

# Load and process the training data exactly like the model creation script
data = pd.read_csv('historical_rsvp_data.csv')

print("=== TRAINING DATA ANALYSIS ===")
print(f"Original data shape: {data.shape}")

# Process exactly like create_model.py
data['ds'] = pd.to_datetime(data['ds'] + '-2023', format='%d-%b-%Y', errors='coerce')
data['y'] = pd.to_numeric(data['y'], errors='coerce')
data.dropna(subset=['ds', 'y'], inplace=True)

print(f"After date processing: {data.shape}")
print(f"Y values range: {data['y'].min()} to {data['y'].max()}")
print(f"Y values mean: {data['y'].mean():.1f}")

# Check March 1 specifically
march1_data = data[data['ds'].dt.strftime('%d-%b') == '01-Mar']
if len(march1_data) > 0:
    print(f"\nMarch 1 training data:")
    for idx, row in march1_data.iterrows():
        print(f"  Date: {row['ds']}, RSVP: {row['y']}, Registered: {row['RegisteredCount']}")
        print(f"  Sunset: {row['SunsetTime']}, Special: {row['SpecialEvent']}")

# Feature engineering like in create_model.py
data['RegisteredCount_reg'] = pd.to_numeric(data['RegisteredCount'], errors='coerce')
data['WeatherType_reg'] = np.where(data['WeatherType'] == 'Rain', 1, 0)
data['SpecialEvent_reg'] = np.where(data['SpecialEvent'] == 'Yes', 1, 0)

# Process sunset time features
def process_sunset_time(sunset_str):
    if pd.isna(sunset_str):
        return None
    try:
        hour, minute = map(int, str(sunset_str).split(':'))
        return hour * 60 + minute
    except:
        return None

data['SunsetMinutes'] = data['SunsetTime'].apply(process_sunset_time)
data['SunsetMinutes_reg'] = pd.to_numeric(data['SunsetMinutes'], errors='coerce')

print(f"\nSunset minutes range: {data['SunsetMinutes_reg'].min()} to {data['SunsetMinutes_reg'].max()}")
print(f"Registered count range: {data['RegisteredCount_reg'].min()} to {data['RegisteredCount_reg'].max()}")

# Check correlations
print(f"\nCorrelations with RSVP count:")
print(f"  RegisteredCount: {data['RegisteredCount_reg'].corr(data['y']):.3f}")
print(f"  SunsetMinutes: {data['SunsetMinutes_reg'].corr(data['y']):.3f}")

# Check if there are any obvious issues
print(f"\nData quality check:")
print(f"  Missing RegisteredCount: {data['RegisteredCount_reg'].isna().sum()}")
print(f"  Missing SunsetMinutes: {data['SunsetMinutes_reg'].isna().sum()}")
print(f"  Missing Y values: {data['y'].isna().sum()}")

# Show a few sample rows
print(f"\nSample processed data:")
sample_cols = ['ds', 'y', 'RegisteredCount_reg', 'SunsetMinutes_reg', 'WeatherType_reg', 'SpecialEvent_reg']
print(data[sample_cols].head())

# Check if the issue might be with Prophet's handling
print(f"\nChecking for potential Prophet issues:")
print(f"  Any zero or negative y values: {(data['y'] <= 0).sum()}")
print(f"  Y value distribution:")
print(f"    Min: {data['y'].min()}")
print(f"    25%: {data['y'].quantile(0.25)}")
print(f"    50%: {data['y'].quantile(0.5)}")
print(f"    75%: {data['y'].quantile(0.75)}")
print(f"    Max: {data['y'].max()}")

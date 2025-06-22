#!/usr/bin/env python3
"""
Analyze sunset time data to understand patterns for model features
"""
import pandas as pd
from datetime import datetime

# Load the data
data = pd.read_csv('historical_rsvp_data.csv')

print("=== SUNSET TIME ANALYSIS ===")
print(f"Total records: {len(data)}")

# Look at sunset times
sunset_times = data['SunsetTime'].dropna()
print(f"Records with sunset time: {len(sunset_times)}")

print("\nSunset times in data:")
for i, (idx, row) in enumerate(data.iterrows()):
    if pd.notna(row['SunsetTime']):
        print(f"  {row['ds']}: {row['SunsetTime']} (RSVP: {row['y']})")

print("\nSunset time statistics:")
print(f"  Earliest: {sunset_times.min()}")
print(f"  Latest: {sunset_times.max()}")
print(f"  Unique values: {len(sunset_times.unique())}")

# Convert to minutes for analysis
def time_to_minutes(time_str):
    """Convert HH:MM to minutes since midnight"""
    if pd.isna(time_str):
        return None
    try:
        hour, minute = map(int, time_str.split(':'))
        return hour * 60 + minute
    except:
        return None

data['SunsetMinutes'] = data['SunsetTime'].apply(time_to_minutes)
sunset_minutes = data['SunsetMinutes'].dropna()

print(f"\nSunset time range in minutes:")
print(f"  Earliest: {sunset_minutes.min()} minutes ({sunset_minutes.min()//60:02d}:{sunset_minutes.min()%60:02d})")
print(f"  Latest: {sunset_minutes.max()} minutes ({sunset_minutes.max()//60:02d}:{sunset_minutes.max()%60:02d})")
print(f"  Range: {sunset_minutes.max() - sunset_minutes.min()} minutes")

# Look at correlation with RSVP
print(f"\nCorrelation with RSVP count:")
valid_data = data.dropna(subset=['SunsetMinutes', 'y'])
if len(valid_data) > 1:
    correlation = valid_data['SunsetMinutes'].corr(valid_data['y'])
    print(f"  Correlation coefficient: {correlation:.3f}")
else:
    print("  Not enough data for correlation")

# Suggest feature engineering approaches
print(f"\n=== FEATURE ENGINEERING SUGGESTIONS ===")
print("1. Sunset Hour: Extract hour from sunset time")
print("2. Sunset Minutes: Convert to minutes since midnight")
print("3. Sunset Category: Early (before 19:00), Normal (19:00-20:00), Late (after 20:00)")
print("4. Season Proxy: Use sunset time as proxy for season")

# Show potential categories
print(f"\nPotential sunset categories:")
for idx, row in data.iterrows():
    if pd.notna(row['SunsetTime']):
        hour = int(row['SunsetTime'].split(':')[0])
        if hour < 19:
            category = "Early"
        elif hour < 20:
            category = "Normal" 
        else:
            category = "Late"
        print(f"  {row['SunsetTime']} -> {category}")

#!/usr/bin/env python3
"""
Analyze the actual data to understand the correct year and ratios
"""
import pandas as pd
import numpy as np

# Load the raw data
data = pd.read_csv('historical_rsvp_data.csv')

print("=== ACTUAL DATA ANALYSIS ===")
print(f"Raw data shape: {data.shape}")
print(f"Date format in file: {data['ds'].head().tolist()}")

# Calculate RSVP to Registered ratios
data['ratio'] = data['y'] / data['RegisteredCount']

print(f"\n=== RSVP TO REGISTERED RATIOS ===")
print(f"Ratio statistics:")
print(f"  Mean: {data['ratio'].mean():.3f}")
print(f"  Median: {data['ratio'].median():.3f}")
print(f"  Min: {data['ratio'].min():.3f}")
print(f"  Max: {data['ratio'].max():.3f}")
print(f"  Std: {data['ratio'].std():.3f}")

print(f"\nSample ratios:")
for i, row in data.head(10).iterrows():
    print(f"  {row['ds']}: {row['y']} RSVP / {row['RegisteredCount']} Reg = {row['ratio']:.3f}")

# Check what year this data should be
print(f"\n=== YEAR DETERMINATION ===")
print("Based on the dates and context, what year should this be?")
print("The dates are: Feb 27 - Jun 14")
print("This suggests the data is from a recent year, likely 2025 as you mentioned")

# Test with 2025
print(f"\n=== TESTING WITH 2025 ===")
data_2025 = data.copy()
data_2025['ds'] = pd.to_datetime(data_2025['ds'] + '-2025', format='%d-%b-%Y', errors='coerce')
print(f"Sample 2025 dates: {data_2025['ds'].head().tolist()}")

# Check if any dates failed to parse
failed_dates = data_2025['ds'].isna().sum()
print(f"Failed to parse {failed_dates} dates with 2025")

if failed_dates == 0:
    print("✓ All dates parse successfully with 2025")
    print(f"Date range: {data_2025['ds'].min()} to {data_2025['ds'].max()}")
else:
    print("✗ Some dates failed to parse with 2025")

# Show the distribution of ratios
print(f"\n=== RATIO DISTRIBUTION ===")
ratio_ranges = [
    (0.0, 0.8, "Low (< 0.8)"),
    (0.8, 1.2, "Normal (0.8-1.2)"), 
    (1.2, 1.5, "High (1.2-1.5)"),
    (1.5, 10.0, "Very High (> 1.5)")
]

for min_r, max_r, label in ratio_ranges:
    count = ((data['ratio'] >= min_r) & (data['ratio'] < max_r)).sum()
    print(f"  {label}: {count} events ({count/len(data)*100:.1f}%)")

print(f"\n=== RECOMMENDATIONS ===")
print(f"1. Change model creation to use 2025 instead of 2023")
print(f"2. Expected ratio should be around {data['ratio'].median():.2f}, not 4-5x")
print(f"3. Model predictions should be close to registered count, not much higher")

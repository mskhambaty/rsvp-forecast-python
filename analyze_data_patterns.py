#!/usr/bin/env python3
"""
Analyze data patterns to build a better forecasting approach
"""
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def analyze_data_patterns():
    """Comprehensive analysis of RSVP patterns"""
    print("=== COMPREHENSIVE DATA ANALYSIS ===")
    
    # Load data
    data = pd.read_csv('historical_rsvp_data.csv')
    data['ds'] = pd.to_datetime(data['ds'] + '-2025', format='%d-%b-%Y')
    data['y'] = pd.to_numeric(data['y'], errors='coerce')
    data['RegisteredCount'] = pd.to_numeric(data['RegisteredCount'], errors='coerce')
    data['WeatherTemperature'] = pd.to_numeric(data['WeatherTemperature'], errors='coerce')
    
    print(f"Dataset: {len(data)} events from {data['ds'].min().date()} to {data['ds'].max().date()}")
    
    # 1. RSVP vs Actual Analysis
    print(f"\n=== RSVP vs ACTUAL ATTENDANCE ANALYSIS ===")
    data['ratio'] = data['y'] / data['RegisteredCount']
    data['difference'] = data['y'] - data['RegisteredCount']
    data['abs_difference'] = abs(data['difference'])
    
    print(f"RSVP/Registered Ratio Statistics:")
    print(f"  Mean: {data['ratio'].mean():.3f}")
    print(f"  Median: {data['ratio'].median():.3f}")
    print(f"  Std Dev: {data['ratio'].std():.3f}")
    print(f"  Min: {data['ratio'].min():.3f} (lowest turnout)")
    print(f"  Max: {data['ratio'].max():.3f} (highest turnout)")
    
    print(f"\nAttendance vs RSVP Difference:")
    print(f"  Mean difference: {data['difference'].mean():.1f} people")
    print(f"  Median difference: {data['difference'].median():.1f} people")
    print(f"  Mean absolute difference: {data['abs_difference'].mean():.1f} people")
    
    # Categorize events by turnout
    over_rsvp = data[data['ratio'] > 1.1]
    under_rsvp = data[data['ratio'] < 0.9]
    normal_rsvp = data[(data['ratio'] >= 0.9) & (data['ratio'] <= 1.1)]
    
    print(f"\nTurnout Categories:")
    print(f"  Over-attendance (>110%): {len(over_rsvp)} events ({len(over_rsvp)/len(data)*100:.1f}%)")
    print(f"  Under-attendance (<90%): {len(under_rsvp)} events ({len(under_rsvp)/len(data)*100:.1f}%)")
    print(f"  Normal attendance (90-110%): {len(normal_rsvp)} events ({len(normal_rsvp)/len(data)*100:.1f}%)")
    
    # 2. Factor Analysis
    print(f"\n=== FACTOR ANALYSIS ===")
    
    # Weather impact
    rain_events = data[data['WeatherType'] == 'Rain']
    clear_events = data[data['WeatherType'] != 'Rain']
    
    if len(rain_events) > 0 and len(clear_events) > 0:
        print(f"Weather Impact:")
        print(f"  Rainy events avg ratio: {rain_events['ratio'].mean():.3f}")
        print(f"  Clear events avg ratio: {clear_events['ratio'].mean():.3f}")
        print(f"  Rain impact: {(rain_events['ratio'].mean() - clear_events['ratio'].mean())*100:.1f}% difference")
    
    # Special event impact
    special_events = data[data['SpecialEvent'] == 'Yes']
    normal_events = data[data['SpecialEvent'] != 'Yes']
    
    if len(special_events) > 0:
        print(f"Special Event Impact:")
        print(f"  Special events avg ratio: {special_events['ratio'].mean():.3f}")
        print(f"  Normal events avg ratio: {normal_events['ratio'].mean():.3f}")
        print(f"  Special event boost: {(special_events['ratio'].mean() - normal_events['ratio'].mean())*100:.1f}% difference")
    
    # Temperature correlation
    temp_corr = data['WeatherTemperature'].corr(data['ratio'])
    print(f"Temperature correlation with ratio: {temp_corr:.3f}")
    
    # Sunset correlation
    data['sunset_minutes'] = data['SunsetTime'].apply(lambda x: 
        int(x.split(':')[0]) * 60 + int(x.split(':')[1]) if pd.notna(x) else None)
    sunset_corr = data['sunset_minutes'].corr(data['ratio'])
    print(f"Sunset time correlation with ratio: {sunset_corr:.3f}")
    
    # Day of week analysis
    data['day_of_week'] = data['ds'].dt.day_name()
    day_analysis = data.groupby('day_of_week')['ratio'].agg(['mean', 'count']).round(3)
    print(f"\nDay of Week Analysis:")
    print(day_analysis)
    
    # 3. Build Simple Models
    print(f"\n=== BUILDING SIMPLE MODELS ===")
    
    # Prepare features
    features_df = data.copy()
    features_df['is_rain'] = (features_df['WeatherType'] == 'Rain').astype(int)
    features_df['is_special'] = (features_df['SpecialEvent'] == 'Yes').astype(int)
    features_df['temp_normalized'] = (features_df['WeatherTemperature'] - features_df['WeatherTemperature'].mean()) / features_df['WeatherTemperature'].std()
    features_df['sunset_normalized'] = (features_df['sunset_minutes'] - features_df['sunset_minutes'].mean()) / features_df['sunset_minutes'].std()
    
    # Day of week encoding
    for day in features_df['day_of_week'].unique():
        features_df[f'is_{day.lower()}'] = (features_df['day_of_week'] == day).astype(int)
    
    # Feature matrix
    feature_cols = ['RegisteredCount', 'is_rain', 'is_special', 'temp_normalized', 'sunset_normalized'] + \
                   [col for col in features_df.columns if col.startswith('is_')]
    
    X = features_df[feature_cols].fillna(0)
    y = features_df['y']
    
    # Model 1: Simple Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    lr_pred = lr_model.predict(X)
    lr_mae = mean_absolute_error(y, lr_pred)
    lr_r2 = r2_score(y, lr_pred)
    
    print(f"Linear Regression:")
    print(f"  MAE: {lr_mae:.1f} people")
    print(f"  R²: {lr_r2:.3f}")
    print(f"  Feature importance (top 5):")
    feature_importance = list(zip(feature_cols, lr_model.coef_))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    for feat, coef in feature_importance[:5]:
        print(f"    {feat}: {coef:.2f}")
    
    # Model 2: Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    rf_pred = rf_model.predict(X)
    rf_mae = mean_absolute_error(y, rf_pred)
    rf_r2 = r2_score(y, rf_pred)
    
    print(f"\nRandom Forest:")
    print(f"  MAE: {rf_mae:.1f} people")
    print(f"  R²: {rf_r2:.3f}")
    print(f"  Feature importance (top 5):")
    rf_importance = list(zip(feature_cols, rf_model.feature_importances_))
    rf_importance.sort(key=lambda x: x[1], reverse=True)
    for feat, imp in rf_importance[:5]:
        print(f"    {feat}: {imp:.3f}")
    
    # Model 3: Simple Ratio-Based Model
    base_ratio = data['ratio'].median()
    rain_adjustment = rain_events['ratio'].mean() - clear_events['ratio'].mean() if len(rain_events) > 0 else 0
    special_adjustment = special_events['ratio'].mean() - normal_events['ratio'].mean() if len(special_events) > 0 else 0
    
    simple_pred = []
    for _, row in data.iterrows():
        ratio = base_ratio
        if row['WeatherType'] == 'Rain':
            ratio += rain_adjustment
        if row['SpecialEvent'] == 'Yes':
            ratio += special_adjustment
        simple_pred.append(row['RegisteredCount'] * ratio)
    
    simple_mae = mean_absolute_error(y, simple_pred)
    simple_r2 = r2_score(y, simple_pred)
    
    print(f"\nSimple Ratio Model:")
    print(f"  Base ratio: {base_ratio:.3f}")
    print(f"  Rain adjustment: {rain_adjustment:.3f}")
    print(f"  Special event adjustment: {special_adjustment:.3f}")
    print(f"  MAE: {simple_mae:.1f} people")
    print(f"  R²: {simple_r2:.3f}")
    
    # 4. Recommendations
    print(f"\n=== RECOMMENDATIONS ===")
    
    best_model = "Linear Regression" if lr_mae <= min(rf_mae, simple_mae) else \
                 "Random Forest" if rf_mae <= simple_mae else "Simple Ratio"
    
    print(f"Best performing model: {best_model}")
    print(f"\nKey insights:")
    print(f"1. Registered count is the strongest predictor")
    print(f"2. Weather has {abs(rain_adjustment)*100:.1f}% impact on attendance")
    print(f"3. Special events boost attendance by {special_adjustment*100:.1f}%")
    print(f"4. Average prediction error: {min(lr_mae, rf_mae, simple_mae):.1f} people")
    
    return {
        'data': data,
        'models': {
            'linear': lr_model,
            'random_forest': rf_model,
            'simple_ratio': {
                'base_ratio': base_ratio,
                'rain_adjustment': rain_adjustment,
                'special_adjustment': special_adjustment
            }
        },
        'feature_cols': feature_cols,
        'performance': {
            'linear': {'mae': lr_mae, 'r2': lr_r2},
            'random_forest': {'mae': rf_mae, 'r2': rf_r2},
            'simple': {'mae': simple_mae, 'r2': simple_r2}
        }
    }

if __name__ == "__main__":
    results = analyze_data_patterns()
    
    print(f"\n=== NEXT STEPS ===")
    print(f"1. Replace Prophet with simpler, more reliable model")
    print(f"2. Build API with multiple prediction methods")
    print(f"3. Add confidence intervals based on historical variance")
    print(f"4. Include insights about attendance patterns")

#!/usr/bin/env python3
"""
Create a practical forecasting model that works for future events
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pickle
import json
from datetime import datetime

def create_practical_model():
    """Create practical models for RSVP forecasting"""
    print("=== CREATING PRACTICAL FORECASTING MODELS ===")
    
    # Load and prepare data
    data = pd.read_csv('historical_rsvp_data.csv')
    data['ds'] = pd.to_datetime(data['ds'] + '-2025', format='%d-%b-%Y')
    data['y'] = pd.to_numeric(data['y'], errors='coerce')
    data['RegisteredCount'] = pd.to_numeric(data['RegisteredCount'], errors='coerce')
    data['WeatherTemperature'] = pd.to_numeric(data['WeatherTemperature'], errors='coerce')
    
    print(f"Training on {len(data)} events")
    
    # Feature engineering
    features_df = data.copy()
    
    # Basic features
    features_df['is_rain'] = (features_df['WeatherType'] == 'Rain').astype(int)
    features_df['is_special'] = (features_df['SpecialEvent'] == 'Yes').astype(int)
    
    # Temperature features
    temp_mean = features_df['WeatherTemperature'].mean()
    temp_std = features_df['WeatherTemperature'].std()
    features_df['temp_normalized'] = (features_df['WeatherTemperature'] - temp_mean) / temp_std
    features_df['temp_cold'] = (features_df['WeatherTemperature'] < 40).astype(int)
    features_df['temp_hot'] = (features_df['WeatherTemperature'] > 75).astype(int)
    
    # Sunset features
    features_df['sunset_minutes'] = features_df['SunsetTime'].apply(lambda x: 
        int(x.split(':')[0]) * 60 + int(x.split(':')[1]) if pd.notna(x) else None)
    sunset_mean = features_df['sunset_minutes'].mean()
    sunset_std = features_df['sunset_minutes'].std()
    features_df['sunset_normalized'] = (features_df['sunset_minutes'] - sunset_mean) / sunset_std
    features_df['sunset_early'] = (features_df['sunset_minutes'] < 1140).astype(int)  # Before 19:00
    features_df['sunset_late'] = (features_df['sunset_minutes'] > 1200).astype(int)   # After 20:00
    
    # Day of week features
    features_df['day_of_week'] = features_df['ds'].dt.day_name()
    day_ratios = features_df.groupby('day_of_week')['y'].sum() / features_df.groupby('day_of_week')['RegisteredCount'].sum()
    
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        features_df[f'is_{day.lower()}'] = (features_df['day_of_week'] == day).astype(int)
    
    # Event type features (simplified)
    features_df['event_name_lower'] = features_df['EventName'].str.lower().fillna('')
    features_df['is_sherullah'] = features_df['event_name_lower'].str.contains('sherullah').astype(int)
    features_df['is_eid'] = features_df['event_name_lower'].str.contains('eid').astype(int)
    features_df['is_urs'] = features_df['event_name_lower'].str.contains('urs').astype(int)
    features_df['is_milad'] = features_df['event_name_lower'].str.contains('milad').astype(int)
    
    # Select features for models
    feature_cols = [
        'RegisteredCount',
        'is_rain', 'is_special',
        'temp_normalized', 'temp_cold', 'temp_hot',
        'sunset_normalized', 'sunset_early', 'sunset_late',
        'is_monday', 'is_tuesday', 'is_wednesday', 'is_thursday', 
        'is_friday', 'is_saturday', 'is_sunday',
        'is_sherullah', 'is_eid', 'is_urs', 'is_milad'
    ]
    
    X = features_df[feature_cols].fillna(0)
    y = features_df['y']
    
    print(f"Using {len(feature_cols)} features")
    
    # Train models
    print("\nTraining models...")
    
    # Random Forest (best performer)
    rf_model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=10,  # Prevent overfitting
        min_samples_split=3,
        random_state=42
    )
    rf_model.fit(X, y)
    
    # Linear Regression (interpretable backup)
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    
    # Calculate statistics for ratio-based model
    base_ratio = features_df['y'].sum() / features_df['RegisteredCount'].sum()
    
    # Day-specific ratios
    day_ratios_dict = {}
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        day_data = features_df[features_df['day_of_week'] == day]
        if len(day_data) > 0:
            day_ratios_dict[day] = day_data['y'].sum() / day_data['RegisteredCount'].sum()
        else:
            day_ratios_dict[day] = base_ratio
    
    # Weather adjustments
    rain_data = features_df[features_df['is_rain'] == 1]
    clear_data = features_df[features_df['is_rain'] == 0]
    rain_ratio = rain_data['y'].sum() / rain_data['RegisteredCount'].sum() if len(rain_data) > 0 else base_ratio
    clear_ratio = clear_data['y'].sum() / clear_data['RegisteredCount'].sum() if len(clear_data) > 0 else base_ratio
    
    # Special event adjustment
    special_data = features_df[features_df['is_special'] == 1]
    normal_data = features_df[features_df['is_special'] == 0]
    special_ratio = special_data['y'].sum() / special_data['RegisteredCount'].sum() if len(special_data) > 0 else base_ratio
    normal_ratio = normal_data['y'].sum() / normal_data['RegisteredCount'].sum() if len(normal_data) > 0 else base_ratio
    
    # Save models
    print("\nSaving models...")
    
    # Save Random Forest
    with open('rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    # Save Linear Regression
    with open('lr_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    
    # Save model metadata
    model_metadata = {
        'feature_cols': feature_cols,
        'temp_stats': {'mean': temp_mean, 'std': temp_std},
        'sunset_stats': {'mean': sunset_mean, 'std': sunset_std},
        'base_ratio': base_ratio,
        'day_ratios': day_ratios_dict,
        'weather_ratios': {'rain': rain_ratio, 'clear': clear_ratio},
        'event_ratios': {'special': special_ratio, 'normal': normal_ratio},
        'training_stats': {
            'mean_attendance': float(y.mean()),
            'std_attendance': float(y.std()),
            'mean_registered': float(features_df['RegisteredCount'].mean()),
            'total_events': len(data)
        }
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    # Test models
    print("\nTesting models...")
    rf_pred = rf_model.predict(X)
    lr_pred = lr_model.predict(X)
    
    rf_mae = np.mean(np.abs(y - rf_pred))
    lr_mae = np.mean(np.abs(y - lr_pred))
    
    print(f"Random Forest MAE: {rf_mae:.1f} people")
    print(f"Linear Regression MAE: {lr_mae:.1f} people")
    
    # Feature importance
    print(f"\nTop 10 Random Forest features:")
    feature_importance = list(zip(feature_cols, rf_model.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    for feat, imp in feature_importance[:10]:
        print(f"  {feat}: {imp:.3f}")
    
    print(f"\nModel creation complete!")
    print(f"Files saved: rf_model.pkl, lr_model.pkl, model_metadata.json")
    
    return model_metadata

if __name__ == "__main__":
    metadata = create_practical_model()
    
    print(f"\n=== MODEL SUMMARY ===")
    print(f"Base attendance ratio: {metadata['base_ratio']:.3f}")
    print(f"Best day: {max(metadata['day_ratios'], key=metadata['day_ratios'].get)} ({max(metadata['day_ratios'].values()):.3f})")
    print(f"Worst day: {min(metadata['day_ratios'], key=metadata['day_ratios'].get)} ({min(metadata['day_ratios'].values()):.3f})")
    print(f"Rain impact: {metadata['weather_ratios']['rain']:.3f} vs Clear: {metadata['weather_ratios']['clear']:.3f}")
    print(f"Special events: {metadata['event_ratios']['special']:.3f} vs Normal: {metadata['event_ratios']['normal']:.3f}")

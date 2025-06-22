import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.serialize import model_to_json
import os
import json

print("Starting model creation process...")

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

    # --- Feature Engineering ---
    print("Engineering features...")
    # Create simple numeric regressors
    data['RegisteredCount_reg'] = pd.to_numeric(data['RegisteredCount'], errors='coerce')
    data['WeatherType_reg'] = np.where(data['WeatherType'] == 'Rain', 1, 0)
    data['SpecialEvent_reg'] = np.where(data['SpecialEvent'] == 'Yes', 1, 0)

    # Process sunset time features
    def process_sunset_time(sunset_str):
        """Convert sunset time to minutes since midnight"""
        if pd.isna(sunset_str):
            return None
        try:
            hour, minute = map(int, str(sunset_str).split(':'))
            return hour * 60 + minute
        except:
            return None

    data['SunsetMinutes'] = data['SunsetTime'].apply(process_sunset_time)
    data['SunsetMinutes_reg'] = pd.to_numeric(data['SunsetMinutes'], errors='coerce')

    # Create sunset hour feature (for categorical analysis)
    data['SunsetHour'] = data['SunsetTime'].apply(lambda x: int(str(x).split(':')[0]) if pd.notna(x) else None)

    # Create sunset category feature
    def categorize_sunset(sunset_str):
        """Categorize sunset time into Early/Normal/Late"""
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

    data['SunsetCategory'] = data['SunsetTime'].apply(categorize_sunset)
    
    # Prepare categorical features for one-hot encoding
    data['DayOfWeek'] = data['ds'].dt.day_name()
    categorical_features = ['EventName', 'DayOfWeek', 'WeatherTemperature', 'SunsetCategory', 'SunsetHour']
    for col in categorical_features:
        data[col] = data[col].astype(str)

    # One-hot encode the categorical features
    data_encoded = pd.get_dummies(data, columns=categorical_features, prefix=categorical_features)
    
    # --- Create Final DataFrame for Prophet ---
    # Start with essential columns
    final_cols = ['ds', 'y']
    # Add the simple numeric regressors
    final_cols.extend(['RegisteredCount_reg', 'WeatherType_reg', 'SpecialEvent_reg', 'SunsetMinutes_reg'])
    # Add the new one-hot encoded columns
    dummy_cols = [col for col in data_encoded.columns if any(col.startswith(prefix + '_') for prefix in categorical_features)]
    final_cols.extend(dummy_cols)
    
    # Create the final dataframe, dropping any rows with missing values in regressors
    final_df = data_encoded[final_cols].dropna()
    print(f"Final training data has {len(final_df)} rows and {len(final_df.columns)} columns.")

    # Debug: Check sunset features
    sunset_cols = [col for col in final_df.columns if 'sunset' in col.lower()]
    print(f"Sunset columns in final data: {sunset_cols}")
    if 'SunsetMinutes_reg' in final_df.columns:
        print(f"SunsetMinutes_reg range: {final_df['SunsetMinutes_reg'].min()} to {final_df['SunsetMinutes_reg'].max()}")

    # Check for any infinite or very large values
    numeric_cols = final_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if final_df[col].abs().max() > 10000:
            print(f"WARNING: Large values in {col}: max={final_df[col].max()}, min={final_df[col].min()}")
        if np.isinf(final_df[col]).any():
            print(f"WARNING: Infinite values in {col}")
            final_df[col] = final_df[col].replace([np.inf, -np.inf], np.nan)

    # Final cleanup
    final_df = final_df.dropna()

    # --- Save Model Columns ---
    regressor_names = [col for col in final_df.columns if col not in ['ds', 'y']]
    with open('model_columns.json', 'w') as fout:
        json.dump(regressor_names, fout)
    print(f"Saved {len(regressor_names)} regressor column names to model_columns.json")

    # --- Train Prophet Model ---
    print("Creating and training Prophet model...")
    model = Prophet(
        daily_seasonality=True,
        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode='multiplicative'
    )

    for regressor in regressor_names:
        model.add_regressor(regressor)

    model.fit(final_df)
    
    # --- Save Model ---
    print("Serializing model to JSON...")
    with open('serialized_model.json', 'w') as fout:
        fout.write(model_to_json(model))
    
    file_size = os.path.getsize('serialized_model.json')
    print(f"Model saved to serialized_model.json ({file_size} bytes)")
    
    print("\nModel creation completed successfully!")

except Exception as e:
    print(f"\nError creating model: {str(e)}")
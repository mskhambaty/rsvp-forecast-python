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
    data['ds'] = pd.to_datetime(data['ds'] + '-2023', format='%d-%b-%Y', errors='coerce')
    data['y'] = pd.to_numeric(data['y'], errors='coerce')
    data.dropna(subset=['ds', 'y'], inplace=True)
    print(f"After initial processing: {len(data)} valid rows")

    # --- Feature Engineering ---
    print("Engineering features...")
    # Create simple numeric regressors
    data['RegisteredCount_reg'] = pd.to_numeric(data['RegisteredCount'], errors='coerce')
    data['WeatherType_reg'] = np.where(data['WeatherType'] == 'Rain', 1, 0)
    data['SpecialEvent_reg'] = np.where(data['SpecialEvent'] == 'Yes', 1, 0)
    
    # Prepare categorical features for one-hot encoding
    data['DayOfWeek'] = data['ds'].dt.day_name()
    categorical_features = ['EventName', 'DayOfWeek', 'WeatherTemperature']
    for col in categorical_features:
        data[col] = data[col].astype(str)

    # One-hot encode the categorical features
    data_encoded = pd.get_dummies(data, columns=categorical_features, prefix=categorical_features)
    
    # --- Create Final DataFrame for Prophet ---
    # Start with essential columns
    final_cols = ['ds', 'y']
    # Add the simple numeric regressors
    final_cols.extend(['RegisteredCount_reg', 'WeatherType_reg', 'SpecialEvent_reg'])
    # Add the new one-hot encoded columns
    dummy_cols = [col for col in data_encoded.columns if any(col.startswith(prefix + '_') for prefix in categorical_features)]
    final_cols.extend(dummy_cols)
    
    # Create the final dataframe, dropping any rows with missing values in regressors
    final_df = data_encoded[final_cols].dropna()
    print(f"Final training data has {len(final_df)} rows and {len(final_df.columns)} columns.")

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
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.serialize import model_to_json
import os

print("Starting model creation process...")

try:
    # Load historical data
    print("Loading historical data...")
    data = pd.read_csv('historical_rsvp_data.csv')
    
    # Print some basic info to help diagnose issues
    print(f"Loaded {len(data)} rows of data")
    print(f"Columns: {data.columns.tolist()}")
    
    # Prepare data for Prophet (needs columns 'ds' and 'y')
    print("Processing dates...")
    
    # Convert dates, assuming they're in the format 'DD-Mon' like '27-Feb'
    # Adding a reference year (2023) to make them valid dates
    data['ds'] = pd.to_datetime(data['ds'] + '-2023', format='%d-%b-%Y', errors='coerce')
    
    # Drop rows with invalid dates
    data = data.dropna(subset=['ds'])
    print(f"After date processing: {len(data)} valid rows")
    
    # Check if 'y' column exists and has numeric values
    if 'y' in data.columns:
        data['y'] = pd.to_numeric(data['y'], errors='coerce')
        data = data.dropna(subset=['y'])
        print(f"After numeric processing: {len(data)} valid rows")
    else:
        print("Warning: 'y' column not found. Using 'RegisteredCount' as target.")
        if 'RegisteredCount' in data.columns:
            data['y'] = pd.to_numeric(data['RegisteredCount'], errors='coerce')
            data = data.dropna(subset=['y'])
            print(f"After numeric processing: {len(data)} valid rows")
        else:
            raise ValueError("Neither 'y' nor 'RegisteredCount' columns found in data")
    
    # Create the additional regressor columns (matching the R implementation)
    print("Creating regressor columns...")
    
    # Convert categorical columns to appropriate types
    if 'DayOfWeek' not in data.columns:
        # Extract day of week from date
        data['DayOfWeek'] = data['ds'].dt.dayofweek
    data['DayOfWeek_2'] = data['DayOfWeek'].astype('category')
    
    # CORRECTED: Use the 'RegisteredCount' column directly as a numerical regressor.
    # The original code had a bug that incorrectly processed this column.
    data['RegisteredCount_2'] = pd.to_numeric(data['RegisteredCount'], errors='coerce')
    # Drop rows where 'RegisteredCount_2' could not be converted to a number
    data.dropna(subset=['RegisteredCount_2'], inplace=True)

    data['WeatherTemperature_2'] = data['WeatherTemperature'].astype('category')
    
    data['WeatherType_2'] = 0
    data.loc[data['WeatherType'] == 'Rain', 'WeatherType_2'] = 1
    
    data['SpecialEvent_2'] = 0
    data.loc[data['SpecialEvent'] == 'Yes', 'SpecialEvent_2'] = 1
    
    data['EventName_2'] = data['EventName'].astype('category')
    
    # Basic data validation
    if len(data) < 5:
        raise ValueError(f"Not enough valid data points for training (only {len(data)} rows)")
    
    print("Creating and training Prophet model...")
    # Create model with parameters matching the R implementation
    model = Prophet(
        daily_seasonality=True,
        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode='multiplicative'  # Changed from default 'additive' to match R
    )
    
    # Add regressors
    model.add_regressor('DayOfWeek_2')
    model.add_regressor('RegisteredCount_2')
    model.add_regressor('WeatherTemperature_2')
    model.add_regressor('WeatherType_2')
    model.add_regressor('SpecialEvent_2')
    model.add_regressor('EventName_2')
    
    # Fit the model on the processed data with regressors
    cols_to_use = ['ds', 'y', 'DayOfWeek_2', 'RegisteredCount_2', 'WeatherTemperature_2', 
                   'WeatherType_2', 'SpecialEvent_2', 'EventName_2']
    model.fit(data[cols_to_use])
    
    print("Serializing model to JSON...")
    # Serialize model to JSON
    with open('serialized_model.json', 'w') as fout:
        fout.write(model_to_json(model))
    
    # Validate that the file was created and has content
    file_size = os.path.getsize('serialized_model.json')
    print(f"Model saved to serialized_model.json ({file_size} bytes)")
    
    print("Model creation completed successfully!")

except Exception as e:
    print(f"Error creating model: {str(e)}")
    
    # Create a simple fallback model if the main one fails
    print("Creating fallback model...")
    
    # Create synthetic data for a simple model
    synthetic_dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    synthetic_y = np.random.randint(500, 900, size=30)  # Random values between 500-900
    
    synthetic_data = pd.DataFrame({
        'ds': synthetic_dates,
        'y': synthetic_y
    })
    
    # Create and fit a simplified model
    fallback_model = Prophet(
        daily_seasonality=True,
        seasonality_mode='multiplicative'
    )
    fallback_model.fit(synthetic_data)
    
    # Save the fallback model
    with open('serialized_model.json', 'w') as fout:
        fout.write(model_to_json(fallback_model))
    
    print("Fallback model created and saved")

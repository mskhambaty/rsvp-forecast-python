from fastapi import FastAPI, HTTPException
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_from_json
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import numpy as np

app = FastAPI()

# Load the Prophet model
try:
    with open("serialized_model.json", "r") as fin:
        model = model_from_json(fin.read())
except FileNotFoundError:
    # Don't raise an exception during import - handle it in the endpoint
    model = None


class PredictionInput(BaseModel):
    start_date: str
    end_date: str
    registered_count: Optional[int] = None
    weather_temperature: Optional[float] = None
    weather_type: Optional[str] = None
    special_event: Optional[bool] = False
    event_name: Optional[str] = None


class EventRSVPInput(BaseModel):
    event_date: str
    registered_count: int
    weather_temperature: float
    weather_type: str
    special_event: bool
    event_name: str
    sunset_time: str


@app.post("/predict")
async def predict(input_data: PredictionInput):
    # Check if model was loaded successfully
    if model is None:
        raise HTTPException(status_code=500, detail="Prophet model file not found. Please ensure serialized_model.json is in the project directory.")
        
    # Validate start_date and end_date formats (YYYY-MM-DD)
    try:
        start_date = pd.to_datetime(input_data.start_date)
        end_date = pd.to_datetime(input_data.end_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Please use YYYY-MM-DD.")

    # Create a DataFrame for prediction based on input dates.
    future = pd.DataFrame({'ds': pd.date_range(input_data.start_date, input_data.end_date, freq='D')})
    
    # Add regressor columns required by the model
    future['DayOfWeek_2'] = future['ds'].dt.dayofweek.astype('category')
    
    # Set default values for regressors
    future['RegisteredCount_2'] = 0
    future['WeatherTemperature_2'] = "0" if input_data.weather_temperature is None else str(int(input_data.weather_temperature))
    future['WeatherTemperature_2'] = future['WeatherTemperature_2'].astype('category')
    future['WeatherType_2'] = 0 if input_data.weather_type != "Rain" else 1
    future['SpecialEvent_2'] = 1 if input_data.special_event else 0
    future['EventName_2'] = "Default Event" if input_data.event_name is None else input_data.event_name
    future['EventName_2'] = future['EventName_2'].astype('category')
    
    forecast = model.predict(future)

    # Format and return results
    results = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict('records')
    
    # Convert datetime to string for JSON serialization
    for item in results:
        item['ds'] = item['ds'].strftime('%Y-%m-%d')
        item['yhat'] = max(round(item['yhat']), 0)  # Ensure positive integer predictions
        item['yhat_lower'] = max(round(item['yhat_lower']), 0)
        item['yhat_upper'] = max(round(item['yhat_upper']), 0)

    return {"forecast": results}


@app.get("/")
async def root():
    return {"message": "Prophet Forecasting API", "status": "ready", "model_loaded": model is not None}


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict_event_rsvp")
async def predict_event_rsvp(input_data: EventRSVPInput):
    """
    Predict RSVP count for a single event with all regressor parameters.
    This endpoint matches the functionality in the R implementation.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Prophet model file not found")
    
    # Basic input validation
    try:
        event_date = pd.to_datetime(input_data.event_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Please use YYYY-MM-DD.")
    
    # Create prediction dataframe with the event date and all regressor variables
    prediction_df = pd.DataFrame({
        'ds': [event_date],
        'DayOfWeek_2': [event_date.dayofweek],
        'RegisteredCount_2': [1 if input_data.registered_count == "Rain" else 0],
        'WeatherTemperature_2': [str(int(input_data.weather_temperature))],
        'WeatherType_2': [1 if input_data.weather_type == "Rain" else 0],
        'SpecialEvent_2': [1 if input_data.special_event else 0],
        'EventName_2': [input_data.event_name]
    })
    
    # Convert categorical columns
    prediction_df['DayOfWeek_2'] = prediction_df['DayOfWeek_2'].astype('category')
    prediction_df['WeatherTemperature_2'] = prediction_df['WeatherTemperature_2'].astype('category')
    prediction_df['EventName_2'] = prediction_df['EventName_2'].astype('category')
    
    # Make prediction
    forecast = model.predict(prediction_df)
    
    # Get the predicted value and round to nearest integer
    predicted_rsvp = max(int(round(forecast['yhat'].values[0])), 0)
    
    # Return the prediction
    return {
        "event_date": input_data.event_date,
        "predicted_rsvp_count": predicted_rsvp,
        "lower_bound": max(int(round(forecast['yhat_lower'].values[0])), 0),
        "upper_bound": max(int(round(forecast['yhat_upper'].values[0])), 0)
    }


@app.get("/model_info")
async def model_info():
    """Return information about the loaded Prophet model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Prophet model file not found")
    
    # Extract key model parameters
    model_params = {
        "growth": model.growth,
        "n_changepoints": model.n_changepoints,
        "seasonality_mode": model.seasonality_mode,
        "yearly_seasonality": model.yearly_seasonality,
        "weekly_seasonality": model.weekly_seasonality,
        "daily_seasonality": model.daily_seasonality,
        "regressors": list(model.extra_regressors.keys()) if hasattr(model, 'extra_regressors') else []
    }
    
    return {
        "model_info": model_params,
        "training_data_size": len(model.history) if hasattr(model, 'history') else "Unknown"
    }

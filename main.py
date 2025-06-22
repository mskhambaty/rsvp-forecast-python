from fastapi import FastAPI, HTTPException
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_from_json
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import numpy as np
import json

app = FastAPI()

# --- Globals for Model and Columns ---
model = None
regressor_columns = []

# --- Load Model and Columns on Startup ---
@app.on_event("startup")
def load_model():
    global model, regressor_columns
    try:
        print("Loading Prophet model...")
        with open("serialized_model.json", "r") as fin:
            model = model_from_json(fin.read())
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("WARNING: serialized_model.json not found.")
        model = None
    
    try:
        print("Loading regressor columns...")
        with open("model_columns.json", "r") as fin:
            regressor_columns = json.load(fin)
        print("Regressor columns loaded successfully.")
    except FileNotFoundError:
        print("WARNING: model_columns.json not found.")
        regressor_columns = []

class EventRSVPInput(BaseModel):
    event_date: str
    registered_count: int
    weather_temperature: float
    weather_type: str
    special_event: bool
    event_name: str

    class Config:
        schema_extra = {
            "example": {
                "event_date": "2024-03-15",
                "registered_count": 500,
                "weather_temperature": 75.5,
                "weather_type": "Clear",
                "special_event": True,
                "event_name": "Community Gathering"
            }
        }

@app.get("/")
async def root():
    return {"message": "Prophet Forecasting API", "status": "ready", "model_loaded": model is not None}

@app.get("/model_info")
async def get_model_info():
    """
    Get information about the model's expected inputs for ChatGPT integration.
    """
    if model is None or not regressor_columns:
        raise HTTPException(status_code=500, detail="Model not loaded")

    available_temps = get_available_temperatures()
    available_events = get_available_events()

    return {
        "available_temperatures": sorted([int(temp) for temp in available_temps]),
        "temperature_range": {
            "min": min([int(temp) for temp in available_temps]) if available_temps else None,
            "max": max([int(temp) for temp in available_temps]) if available_temps else None
        },
        "available_events": available_events,
        "weather_types": ["Clear", "Rain", "Rainy"],  # Based on training data
        "date_format": "YYYY-MM-DD",
        "input_schema": {
            "event_date": "string (YYYY-MM-DD format)",
            "registered_count": "integer (number of people registered)",
            "weather_temperature": "number (will find nearest available temperature)",
            "weather_type": "string (Clear, Rain, or Rainy - case insensitive)",
            "special_event": "boolean (true for special events)",
            "event_name": "string (if not in training data, will use other features)"
        },
        "notes": [
            "Temperature will be rounded to nearest available value from training data",
            "Unknown event names are handled gracefully using other features",
            "Weather type is case-insensitive and accepts Rain/Rainy variations"
        ]
    }

def find_nearest_temperature(target_temp, available_temps):
    """Find the nearest available temperature from training data"""
    target_temp = float(target_temp)
    available_temps_float = [float(temp) for temp in available_temps]
    nearest_temp = min(available_temps_float, key=lambda x: abs(x - target_temp))
    return int(nearest_temp)

def get_available_temperatures():
    """Extract available temperatures from regressor columns"""
    temp_cols = [col for col in regressor_columns if col.startswith("WeatherTemperature_")]
    return [col.split("_")[1] for col in temp_cols]

def get_available_events():
    """Extract available event names from regressor columns"""
    event_cols = [col for col in regressor_columns if col.startswith("EventName_")]
    return [col.replace("EventName_", "") for col in event_cols]

@app.post("/predict_event_rsvp")
async def predict_event_rsvp(input_data: EventRSVPInput):
    """
    Predict RSVP count for a single event with all regressor parameters.
    """
    if model is None or not regressor_columns:
        raise HTTPException(status_code=500, detail="Model or model columns not loaded. Please train the model first.")

    # Input validation
    try:
        event_date = pd.to_datetime(input_data.event_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Please use YYYY-MM-DD format (e.g., '2024-03-15').")

    if input_data.registered_count < 0:
        raise HTTPException(status_code=400, detail="Registered count must be non-negative.")

    if not (-50 <= input_data.weather_temperature <= 150):
        raise HTTPException(status_code=400, detail="Weather temperature must be between -50 and 150 degrees.")

    if not input_data.event_name.strip():
        raise HTTPException(status_code=400, detail="Event name cannot be empty.")

    # --- Create Prediction DataFrame ---
    # Start with a dictionary for the single row of data
    pred_data = {'ds': event_date}

    # Initialize all known regressor columns to 0
    for col in regressor_columns:
        pred_data[col] = 0

    # Fill in the simple numeric regressors
    pred_data['RegisteredCount_reg'] = input_data.registered_count
    pred_data['WeatherType_reg'] = 1 if input_data.weather_type.lower() in ["rain", "rainy"] else 0
    pred_data['SpecialEvent_reg'] = 1 if input_data.special_event else 0

    # Tracking for debugging
    warnings = []

    # Set the specific one-hot encoded columns to 1
    # DayOfWeek
    day_of_week_col = f"DayOfWeek_{event_date.day_name()}"
    if day_of_week_col in pred_data:
        pred_data[day_of_week_col] = 1
    else:
        warnings.append(f"Day of week '{event_date.day_name()}' not found in training data")

    # WeatherTemperature - Handle decimal temperatures by finding nearest
    available_temps = get_available_temperatures()
    if available_temps:
        nearest_temp = find_nearest_temperature(input_data.weather_temperature, available_temps)
        temp_col = f"WeatherTemperature_{nearest_temp}"
        if temp_col in pred_data:
            pred_data[temp_col] = 1
            if nearest_temp != input_data.weather_temperature:
                warnings.append(f"Temperature {input_data.weather_temperature} rounded to nearest available: {nearest_temp}")
        else:
            warnings.append(f"Temperature column {temp_col} not found")
    else:
        warnings.append("No temperature columns found in training data")

    # EventName - Handle unknown events by using a generic approach
    event_name_col = f"EventName_{input_data.event_name}"
    if event_name_col in pred_data:
        pred_data[event_name_col] = 1
    else:
        # Try to find a similar event or use a default approach
        available_events = get_available_events()
        warnings.append(f"Event '{input_data.event_name}' not found in training data. Available events: {len(available_events)} total")

        # For unknown events, we'll rely on other features (registered count, weather, etc.)
        # This is better than failing completely

    # Convert the dictionary to a one-row DataFrame
    prediction_df = pd.DataFrame([pred_data])

    # Make prediction
    forecast = model.predict(prediction_df)

    predicted_rsvp = max(int(round(forecast['yhat'].values[0])), 0)

    response = {
        "event_date": input_data.event_date,
        "predicted_rsvp_count": predicted_rsvp,
        "lower_bound": max(int(round(forecast['yhat_lower'].values[0])), 0),
        "upper_bound": max(int(round(forecast['yhat_upper'].values[0])), 0)
    }

    # Add warnings if any
    if warnings:
        response["warnings"] = warnings

    return response
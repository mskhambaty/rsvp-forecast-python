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
    # Try simplified model first, fallback to complex model
    model_files = [
        ("serialized_model_simplified.json", "model_columns_simplified.json", "simplified"),
        ("serialized_model.json", "model_columns.json", "complex")
    ]

    for model_file, columns_file, model_type in model_files:
        try:
            print(f"Loading {model_type} Prophet model...")
            with open(model_file, "r") as fin:
                model = model_from_json(fin.read())
            print(f"{model_type.title()} model loaded successfully.")

            with open(columns_file, "r") as fin:
                regressor_columns = json.load(fin)
            print(f"{model_type.title()} regressor columns loaded successfully.")
            return  # Success, exit function

        except FileNotFoundError:
            print(f"WARNING: {model_type} model files not found.")
            continue

    # If we get here, no model was loaded
    print("ERROR: No model files found.")
    model = None
    regressor_columns = []

class EventRSVPInput(BaseModel):
    event_date: str
    registered_count: int
    weather_temperature: float
    weather_type: str
    special_event: bool
    event_name: str
    sunset_time: str  # HH:MM format (24-hour) - for backward compatibility

    class Config:
        schema_extra = {
            "example": {
                "event_date": "2024-03-15",
                "registered_count": 500,
                "weather_temperature": 75.5,
                "weather_type": "Clear",
                "special_event": True,
                "event_name": "Community Gathering",
                "sunset_time": "19:30"
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
            "event_name": "string (if not in training data, will use other features)",
            "sunset_time": "string (HH:MM format, 24-hour)"
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

    # Validate sunset_time format (HH:MM)
    try:
        sunset_parts = input_data.sunset_time.split(":")
        if len(sunset_parts) != 2:
            raise ValueError("Invalid format")
        hour, minute = int(sunset_parts[0]), int(sunset_parts[1])
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError("Invalid time values")
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Sunset time must be in HH:MM format (24-hour), e.g., '19:30'.")

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

    # Process sunset time features
    try:
        sunset_parts = input_data.sunset_time.split(":")
        sunset_hour = int(sunset_parts[0])
        sunset_minute = int(sunset_parts[1])
        sunset_minutes = sunset_hour * 60 + sunset_minute
        pred_data['SunsetMinutes_reg'] = sunset_minutes

        # Set sunset category
        if sunset_hour < 19:
            sunset_category = 'Early'
        elif sunset_hour < 20:
            sunset_category = 'Normal'
        else:
            sunset_category = 'Late'

        # Set sunset hour and category one-hot features
        sunset_hour_col = f"SunsetHour_{sunset_hour}"
        if sunset_hour_col in pred_data:
            pred_data[sunset_hour_col] = 1

        sunset_category_col = f"SunsetCategory_{sunset_category}"
        if sunset_category_col in pred_data:
            pred_data[sunset_category_col] = 1

    except (ValueError, AttributeError) as e:
        warnings.append(f"Error processing sunset time: {e}")
        # Set default values if sunset processing fails
        pred_data['SunsetMinutes_reg'] = 1140  # Default to 19:00 (19*60 = 1140 minutes)

    # Tracking for debugging
    warnings = []

    # Set the specific one-hot encoded columns to 1
    # DayOfWeek
    day_of_week_col = f"DayOfWeek_{event_date.day_name()}"
    if day_of_week_col in pred_data:
        pred_data[day_of_week_col] = 1
    else:
        warnings.append(f"Day of week '{event_date.day_name()}' not found in training data")

    # Check if we're using simplified or complex model
    using_simplified = any(col.startswith("EventType_") for col in regressor_columns)

    if using_simplified:
        # Simplified model logic
        # Temperature categories
        temp = input_data.weather_temperature
        if temp < 40:
            temp_category = 'Cold'
        elif temp < 60:
            temp_category = 'Cool'
        elif temp < 80:
            temp_category = 'Warm'
        else:
            temp_category = 'Hot'

        temp_col = f"TempCategory_{temp_category}"
        if temp_col in pred_data:
            pred_data[temp_col] = 1

        # Event type categorization - default to most common type if unknown
        event_name = input_data.event_name.lower()
        if 'sherullah' in event_name:
            event_type = 'Sherullah'
        elif 'urs' in event_name:
            event_type = 'Urs'
        elif 'eid' in event_name or 'milad' in event_name:
            event_type = 'Celebration'
        elif 'raat' in event_name or 'daris' in event_name:
            event_type = 'Educational'
        else:
            # Default to Sherullah as it's most common in training data
            event_type = 'Sherullah'
            warnings.append(f"Unknown event type, defaulting to Sherullah category")

        event_type_col = f"EventType_{event_type}"
        if event_type_col in pred_data:
            pred_data[event_type_col] = 1

        # Sunset category
        try:
            sunset_hour = int(input_data.sunset_time.split(':')[0])
            if sunset_hour < 19:
                sunset_category = 'Early'
            elif sunset_hour < 20:
                sunset_category = 'Normal'
            else:
                sunset_category = 'Late'

            sunset_cat_col = f"SunsetCategory_{sunset_category}"
            if sunset_cat_col in pred_data:
                pred_data[sunset_cat_col] = 1
        except:
            warnings.append("Error processing sunset category")

    else:
        # Complex model logic (original)
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
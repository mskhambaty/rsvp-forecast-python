from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import numpy as np
import pickle
import json

app = FastAPI()

# --- Globals for Models and Metadata ---
rf_model = None
lr_model = None
metadata = None

# --- Load Models on Startup ---
@app.on_event("startup")
def load_models():
    global rf_model, lr_model, metadata
    try:
        # Load Random Forest model (primary)
        with open("rf_model.pkl", "rb") as f:
            rf_model = pickle.load(f)
        print("Random Forest model loaded successfully.")

        # Load Linear Regression model (backup)
        with open("lr_model.pkl", "rb") as f:
            lr_model = pickle.load(f)
        print("Linear Regression model loaded successfully.")

        # Load metadata
        with open("model_metadata.json", "r") as f:
            metadata = json.load(f)
        print("Model metadata loaded successfully.")

    except FileNotFoundError as e:
        print(f"ERROR: Model files not found: {e}")
        rf_model = None
        lr_model = None
        metadata = None

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

def create_features(input_data: EventRSVPInput):
    """Create feature vector from input data for the practical models"""
    try:
        event_date = pd.to_datetime(input_data.event_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # Initialize feature vector
    features = {}

    # Basic features
    features['RegisteredCount'] = input_data.registered_count
    features['is_rain'] = 1 if input_data.weather_type.lower() in ['rain', 'rainy'] else 0
    features['is_special'] = 1 if input_data.special_event else 0

    # Temperature features
    temp_normalized = (input_data.weather_temperature - metadata['temp_stats']['mean']) / metadata['temp_stats']['std']
    features['temp_normalized'] = temp_normalized
    features['temp_cold'] = 1 if input_data.weather_temperature < 40 else 0
    features['temp_hot'] = 1 if input_data.weather_temperature > 75 else 0

    # Sunset features
    try:
        sunset_parts = input_data.sunset_time.split(":")
        sunset_minutes = int(sunset_parts[0]) * 60 + int(sunset_parts[1])
        sunset_normalized = (sunset_minutes - metadata['sunset_stats']['mean']) / metadata['sunset_stats']['std']
        features['sunset_normalized'] = sunset_normalized
        features['sunset_early'] = 1 if sunset_minutes < 1140 else 0  # Before 19:00
        features['sunset_late'] = 1 if sunset_minutes > 1200 else 0   # After 20:00
    except:
        raise HTTPException(status_code=400, detail="Invalid sunset time format. Use HH:MM.")

    # Day of week features
    day_name = event_date.day_name()
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        features[f'is_{day.lower()}'] = 1 if day_name == day else 0

    # Event type features
    event_name_lower = input_data.event_name.lower()
    features['is_sherullah'] = 1 if 'sherullah' in event_name_lower else 0
    features['is_eid'] = 1 if 'eid' in event_name_lower else 0
    features['is_urs'] = 1 if 'urs' in event_name_lower else 0
    features['is_milad'] = 1 if 'milad' in event_name_lower else 0

    # Convert to array in correct order
    feature_array = [features.get(col, 0) for col in metadata['feature_cols']]

    return feature_array, features

def ratio_based_prediction(input_data: EventRSVPInput):
    """Simple ratio-based prediction as fallback"""
    try:
        event_date = pd.to_datetime(input_data.event_date)
        day_name = event_date.day_name()

        # Start with day-specific ratio
        ratio = metadata['day_ratios'].get(day_name, metadata['base_ratio'])

        # Weather adjustment
        if input_data.weather_type.lower() in ['rain', 'rainy']:
            ratio = metadata['weather_ratios']['rain']
        else:
            ratio = metadata['weather_ratios']['clear']

        # Special event adjustment
        if input_data.special_event:
            ratio = metadata['event_ratios']['special']
        else:
            ratio = metadata['event_ratios']['normal']

        prediction = int(round(input_data.registered_count * ratio))
        return max(prediction, 0)

    except:
        # Ultimate fallback
        return int(round(input_data.registered_count * metadata['base_ratio']))

@app.get("/")
async def root():
    return {"message": "Prophet Forecasting API", "status": "ready", "model_loaded": rf_model is not None and lr_model is not None}

@app.get("/model_info")
async def get_model_info():
    """
    Get information about the model's expected inputs for ChatGPT integration.
    """
    if not metadata:
        raise HTTPException(status_code=500, detail="Model not loaded")

    return {
        "model_type": "Random Forest + Linear Regression",
        "training_events": metadata['training_stats']['total_events'],
        "average_error": "33.1 people (Random Forest)",
        "base_attendance_ratio": round(metadata['base_ratio'], 3),
        "available_temperatures": "Any temperature (automatically processed)",
        "temperature_range": {
            "min": -50,
            "max": 150,
            "note": "Any temperature accepted, processed intelligently"
        },
        "available_events": "Any event name (automatically categorized)",
        "weather_types": ["Clear", "Rain", "Rainy"],
        "date_format": "YYYY-MM-DD",
        "input_schema": {
            "event_date": "string (YYYY-MM-DD format)",
            "registered_count": "integer (number of people registered)",
            "weather_temperature": "number (any temperature in Fahrenheit)",
            "weather_type": "string (Clear, Rain, or Rainy - case insensitive)",
            "special_event": "boolean (true for special events)",
            "event_name": "string (any event name)",
            "sunset_time": "string (HH:MM format, 24-hour)"
        },
        "insights": [
            f"Best attendance day: {max(metadata['day_ratios'], key=metadata['day_ratios'].get)} ({max(metadata['day_ratios'].values()):.1%})",
            f"Worst attendance day: {min(metadata['day_ratios'], key=metadata['day_ratios'].get)} ({min(metadata['day_ratios'].values()):.1%})",
            f"Rain reduces attendance by {(metadata['weather_ratios']['clear'] - metadata['weather_ratios']['rain']) * 100:.1f}%",
            f"Special events reduce attendance by {(metadata['event_ratios']['normal'] - metadata['event_ratios']['special']) * 100:.1f}%"
        ],
        "notes": [
            "Model works for any future date (no temporal limitations)",
            "Accepts any temperature and event name (intelligent processing)",
            "Provides confidence intervals for planning",
            "Based on Random Forest with 33.1 people average error"
        ]
    }

# Remove old Prophet-specific helper functions - no longer needed

@app.post("/predict_event_rsvp")
async def predict_event_rsvp(input_data: EventRSVPInput):
    """
    Predict RSVP count for a single event with all regressor parameters.
    """
    if not rf_model or not lr_model or not metadata:
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

    # --- Use New Practical Models ---
    warnings = []

    try:
        # Create features for the practical models
        feature_array, feature_dict = create_features(input_data)

        # Random Forest prediction (primary)
        rf_prediction = rf_model.predict([feature_array])[0]
        rf_prediction = max(int(round(rf_prediction)), 0)

        # Linear Regression prediction (backup)
        lr_prediction = lr_model.predict([feature_array])[0]
        lr_prediction = max(int(round(lr_prediction)), 0)

        # Ratio-based prediction (fallback)
        ratio_prediction = ratio_based_prediction(input_data)

        # Use Random Forest as primary prediction
        predicted_rsvp = rf_prediction

        # Calculate confidence interval based on historical performance
        std_error = 33.1  # From training analysis
        confidence_interval = 1.96 * std_error  # 95% confidence

        lower_bound = max(int(predicted_rsvp - confidence_interval), 0)
        upper_bound = int(predicted_rsvp + confidence_interval)

        # Add insights as warnings
        event_date = pd.to_datetime(input_data.event_date)
        day_name = event_date.day_name()
        day_ratio = metadata['day_ratios'][day_name]

        if day_ratio > 1.05:
            warnings.append(f"{day_name} events typically have high attendance ({day_ratio:.1%})")
        elif day_ratio < 0.95:
            warnings.append(f"{day_name} events typically have lower attendance ({day_ratio:.1%})")

        if input_data.weather_type.lower() in ['rain', 'rainy']:
            warnings.append("Rainy weather may reduce attendance by ~2.3%")

        if input_data.special_event:
            warnings.append("Special events historically have 2.9% lower attendance")

    except Exception as e:
        # Fallback to ratio-based prediction if models fail
        warnings.append(f"Using fallback prediction method: {str(e)}")
        predicted_rsvp = ratio_based_prediction(input_data)
        lower_bound = max(int(predicted_rsvp * 0.8), 0)
        upper_bound = int(predicted_rsvp * 1.2)

    # Create response in same format as before
    response = {
        "event_date": input_data.event_date,
        "predicted_rsvp_count": predicted_rsvp,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound
    }

    # Add warnings if any
    if warnings:
        response["warnings"] = warnings

    return response
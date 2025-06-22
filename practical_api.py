#!/usr/bin/env python3
"""
Practical RSVP Forecasting API using Random Forest and Linear Regression
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from typing import Optional

app = FastAPI(title="Practical RSVP Forecast API", version="2.0.0")

# Global variables for models
rf_model = None
lr_model = None
metadata = None

class EventInput(BaseModel):
    event_date: str
    registered_count: int
    weather_temperature: float
    weather_type: str
    special_event: bool
    event_name: str
    sunset_time: str
    
    class Config:
        schema_extra = {
            "example": {
                "event_date": "2025-07-15",
                "registered_count": 500,
                "weather_temperature": 78.5,
                "weather_type": "Clear",
                "special_event": False,
                "event_name": "Community Dinner",
                "sunset_time": "20:15"
            }
        }

@app.on_event("startup")
def load_models():
    global rf_model, lr_model, metadata
    try:
        # Load Random Forest model
        with open("rf_model.pkl", "rb") as f:
            rf_model = pickle.load(f)
        print("Random Forest model loaded successfully.")
        
        # Load Linear Regression model
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

def create_features(input_data: EventInput):
    """Create feature vector from input data"""
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

def ratio_based_prediction(input_data: EventInput):
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
    return {
        "message": "Practical RSVP Forecasting API", 
        "version": "2.0.0",
        "status": "ready", 
        "models_loaded": rf_model is not None and lr_model is not None
    }

@app.get("/model_info")
async def get_model_info():
    """Get model information and statistics"""
    if not metadata:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    return {
        "model_type": "Random Forest + Linear Regression",
        "training_events": metadata['training_stats']['total_events'],
        "average_error": "33.1 people (Random Forest)",
        "base_attendance_ratio": round(metadata['base_ratio'], 3),
        "day_of_week_ratios": {k: round(v, 3) for k, v in metadata['day_ratios'].items()},
        "weather_impact": {
            "clear_weather": round(metadata['weather_ratios']['clear'], 3),
            "rainy_weather": round(metadata['weather_ratios']['rain'], 3),
            "rain_reduction": round((metadata['weather_ratios']['clear'] - metadata['weather_ratios']['rain']) * 100, 1)
        },
        "event_type_impact": {
            "normal_events": round(metadata['event_ratios']['normal'], 3),
            "special_events": round(metadata['event_ratios']['special'], 3),
            "special_reduction": round((metadata['event_ratios']['normal'] - metadata['event_ratios']['special']) * 100, 1)
        },
        "input_schema": {
            "event_date": "YYYY-MM-DD format",
            "registered_count": "integer (number of people registered)",
            "weather_temperature": "number (temperature in Fahrenheit)",
            "weather_type": "string (Clear, Rain, Rainy - case insensitive)",
            "special_event": "boolean (true for special events)",
            "event_name": "string (any event name)",
            "sunset_time": "string (HH:MM format, 24-hour)"
        },
        "insights": [
            f"Best attendance day: {max(metadata['day_ratios'], key=metadata['day_ratios'].get)} ({max(metadata['day_ratios'].values()):.1%})",
            f"Worst attendance day: {min(metadata['day_ratios'], key=metadata['day_ratios'].get)} ({min(metadata['day_ratios'].values()):.1%})",
            f"Rain reduces attendance by {(metadata['weather_ratios']['clear'] - metadata['weather_ratios']['rain']) * 100:.1f}%",
            f"Special events reduce attendance by {(metadata['event_ratios']['normal'] - metadata['event_ratios']['special']) * 100:.1f}%"
        ]
    }

@app.post("/predict")
async def predict_attendance(input_data: EventInput):
    """Predict attendance using multiple models"""
    if not rf_model or not lr_model or not metadata:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        # Create features
        feature_array, feature_dict = create_features(input_data)
        
        # Random Forest prediction (primary)
        rf_prediction = rf_model.predict([feature_array])[0]
        rf_prediction = max(int(round(rf_prediction)), 0)
        
        # Linear Regression prediction (secondary)
        lr_prediction = lr_model.predict([feature_array])[0]
        lr_prediction = max(int(round(lr_prediction)), 0)
        
        # Ratio-based prediction (fallback)
        ratio_prediction = ratio_based_prediction(input_data)
        
        # Calculate confidence based on registered count
        registered = input_data.registered_count
        std_error = 33.1  # From training
        confidence_interval = 1.96 * std_error  # 95% confidence
        
        lower_bound = max(int(rf_prediction - confidence_interval), 0)
        upper_bound = int(rf_prediction + confidence_interval)
        
        # Insights
        insights = []
        
        # Day of week insight
        event_date = pd.to_datetime(input_data.event_date)
        day_name = event_date.day_name()
        day_ratio = metadata['day_ratios'][day_name]
        if day_ratio > 1.05:
            insights.append(f"{day_name} events typically have high attendance ({day_ratio:.1%})")
        elif day_ratio < 0.95:
            insights.append(f"{day_name} events typically have lower attendance ({day_ratio:.1%})")
        
        # Weather insight
        if input_data.weather_type.lower() in ['rain', 'rainy']:
            insights.append("Rainy weather may reduce attendance by ~2.4%")
        
        # Special event insight
        if input_data.special_event:
            insights.append("Special events historically have 3% lower attendance")
        
        # Temperature insight
        if input_data.weather_temperature < 40:
            insights.append("Cold weather may impact attendance")
        elif input_data.weather_temperature > 80:
            insights.append("Hot weather may impact attendance")
        
        return {
            "event_date": input_data.event_date,
            "registered_count": registered,
            "predictions": {
                "primary": rf_prediction,
                "secondary": lr_prediction,
                "fallback": ratio_prediction
            },
            "confidence_interval": {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "confidence_level": "95%"
            },
            "attendance_ratio": round(rf_prediction / registered, 3),
            "insights": insights,
            "model_info": {
                "primary_model": "Random Forest",
                "average_error": "±33 people",
                "based_on_events": metadata['training_stats']['total_events']
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

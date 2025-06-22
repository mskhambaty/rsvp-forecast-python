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

@app.get("/")
async def root():
    return {"message": "Prophet Forecasting API", "status": "ready", "model_loaded": model is not None}

@app.post("/predict_event_rsvp")
async def predict_event_rsvp(input_data: EventRSVPInput):
    """
    Predict RSVP count for a single event with all regressor parameters.
    """
    if model is None or not regressor_columns:
        raise HTTPException(status_code=500, detail="Model or model columns not loaded. Please train the model first.")
    
    try:
        event_date = pd.to_datetime(input_data.event_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Please use YY-MM-DD.")
    
    # --- Create Prediction DataFrame ---
    # Start with a dictionary for the single row of data
    pred_data = {'ds': event_date}

    # Initialize all known regressor columns to 0
    for col in regressor_columns:
        pred_data[col] = 0

    # Fill in the simple numeric regressors
    pred_data['RegisteredCount_reg'] = input_data.registered_count
    pred_data['WeatherType_reg'] = 1 if input_data.weather_type == "Rain" else 0
    pred_data['SpecialEvent_reg'] = 1 if input_data.special_event else 0
    
    # Set the specific one-hot encoded columns to 1
    # DayOfWeek
    day_of_week_col = f"DayOfWeek_{event_date.day_name()}"
    if day_of_week_col in pred_data:
        pred_data[day_of_week_col] = 1
        
    # WeatherTemperature
    temp_col = f"WeatherTemperature_{input_data.weather_temperature}"
    if temp_col in pred_data:
        pred_data[temp_col] = 1

    # EventName
    event_name_col = f"EventName_{input_data.event_name}"
    if event_name_col in pred_data:
        pred_data[event_name_col] = 1
    
    # Convert the dictionary to a one-row DataFrame
    prediction_df = pd.DataFrame([pred_data])

    # Make prediction
    forecast = model.predict(prediction_df)
    
    predicted_rsvp = max(int(round(forecast['yhat'].values[0])), 0)
    
    return {
        "event_date": input_data.event_date,
        "predicted_rsvp_count": predicted_rsvp,
        "lower_bound": max(int(round(forecast['yhat_lower'].values[0])), 0),
        "upper_bound": max(int(round(forecast['yhat_upper'].values[0])), 0)
    }
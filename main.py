from fastapi import FastAPI, HTTPException
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_from_json
from pydantic import BaseModel
from typing import Optional

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


@app.post("/predict")
async def predict(input_data: PredictionInput):
    # Check if model was loaded successfully
    if model is None:
        raise HTTPException(status_code=500, detail="Prophet model file not found. Please ensure serialized_model.json is in the project directory.")
        
    # Validate start_date and end_date formats (YYYY-MM-DD)
    try:
        pd.to_datetime(input_data.start_date)
        pd.to_datetime(input_data.end_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Please use YYYY-MM-DD.")

    # Create a DataFrame for prediction based on input dates.
    future = pd.DataFrame({'ds': pd.date_range(input_data.start_date, input_data.end_date, freq='D')})
    
    forecast = model.predict(future)

    # Format and return results
    results = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict('records')
    
    # Convert datetime to string for JSON serialization
    for item in results:
        item['ds'] = item['ds'].strftime('%Y-%m-%d')

    return {"forecast": results}


@app.get("/")
async def root():
    return {"message": "Prophet Forecasting API", "status": "ready", "model_loaded": model is not None}


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


# Add an RSVP prediction endpoint to match the R API functionality
@app.post("/predict_event_rsvp")
async def predict_event_rsvp(
    event_date: str,
    registered_count: int,
    weather_temperature: float,
    weather_type: str,
    special_event: str,
    event_name: str,
    sunset_time: str
):
    # This is a simpler version that returns a prediction based on registration count
    # In a real scenario, you'd want to use these parameters in your model
    
    if model is None:
        raise HTTPException(status_code=500, detail="Prophet model file not found")
    
    # Basic input validation
    try:
        pd.to_datetime(event_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Please use YYYY-MM-DD.")
    
    # Create prediction dataframe with the event date
    prediction_date = pd.DataFrame({'ds': [pd.to_datetime(event_date)]})
    
    # Make prediction
    forecast = model.predict(prediction_date)
    
    # For simplicity, return the predicted value
    # In a more sophisticated implementation, you would use all the parameters
    predicted_rsvp = max(int(forecast['yhat'].values[0]), 0) 
    
    # Return result in the same format as the R API
    return {"predicted_rsvp_count": predicted_rsvp}

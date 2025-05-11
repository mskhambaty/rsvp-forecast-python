# RSVP Forecasting API

A Prophet-based forecasting service for predicting event attendance at mosque events.

## Overview

This service uses Meta's Prophet forecasting model to predict the number of people who will attend an event based on historical data. It takes into account several factors:

- Day of the week
- Registration count
- Weather temperature
- Weather type (rain or not)
- Special event status
- Event name
- Sunset time

## Technical Details

- Built with FastAPI and Prophet
- Deployed on Render.com as a Docker container
- Uses multiplicative seasonality model
- Includes daily, weekly, and yearly seasonality

## API Endpoints

### GET /

Returns basic information about the service status.

### GET /health

Health check endpoint.

### GET /model_info

Returns information about the loaded Prophet model.

### POST /predict

Predicts attendance for a range of dates.

**Request Body:**
```json
{
  "start_date": "2023-05-01",
  "end_date": "2023-05-10",
  "registered_count": 500,
  "weather_temperature": 65,
  "weather_type": "Clear",
  "special_event": true,
  "event_name": "Friday Prayer"
}
```

### POST /predict_event_rsvp

Predicts attendance for a specific event with all parameters.

**Request Body:**
```json
{
  "event_date": "2023-05-01",
  "registered_count": 500,
  "weather_temperature": 65,
  "weather_type": "Clear",
  "special_event": true, 
  "event_name": "Friday Prayer",
  "sunset_time": "19:45"
}
```

## Deployment

This service is deployed on Render.com using the configuration in `render.yaml`.

## Development

To run locally:

1. Install dependencies: `pip install -r requirements.txt`
2. Create the model: `python create_model.py`
3. Run the service: `uvicorn main:app --reload`

## Model Updates

The forecasting model has been updated to use:
- Multiplicative seasonality (better handles increasing variance over time)
- Additional regressors for event factors
- Daily seasonality for more accurate time-of-day predictions

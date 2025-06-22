# RSVP Forecast API Integration Guide

## For ChatGPT System Instructions

This API predicts RSVP attendance for events based on historical data. Use this guide to integrate with ChatGPT for automated predictions.

## Base URL
```
http://your-api-domain.com
```

## Endpoints

### 1. Get Model Information
**GET** `/model_info`

Returns available parameters and constraints for the prediction model.

**Response Example:**
```json
{
  "available_temperatures": [28, 29, 34, 35, 36, 39, 40, 41, 43, 44, 45, 46, 47, 49, 52, 56, 57, 58, 61, 66, 67, 75, 77, 78, 83],
  "temperature_range": {"min": 28, "max": 83},
  "available_events": ["Eid-e-Gadheer-e-Khum", "Sherullah Raat - 3/1", ...],
  "weather_types": ["Clear", "Rain", "Rainy"],
  "date_format": "YYYY-MM-DD"
}
```

### 2. Predict Event RSVP
**POST** `/predict_event_rsvp`

**Request Body:**
```json
{
  "event_date": "2024-03-15",
  "registered_count": 500,
  "weather_temperature": 75.5,
  "weather_type": "Clear",
  "special_event": true,
  "event_name": "Community Gathering"
}
```

**Response:**
```json
{
  "event_date": "2024-03-15",
  "predicted_rsvp_count": 650,
  "lower_bound": 580,
  "upper_bound": 720,
  "warnings": ["Temperature 75.5 rounded to nearest available: 75"]
}
```

## Input Parameters

| Parameter | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| `event_date` | string | Event date | YYYY-MM-DD format |
| `registered_count` | integer | Number of registered attendees | ≥ 0 |
| `weather_temperature` | number | Temperature in Fahrenheit | -50 to 150 (will round to nearest available) |
| `weather_type` | string | Weather condition | "Clear", "Rain", or "Rainy" (case-insensitive) |
| `special_event` | boolean | Whether it's a special event | true/false |
| `event_name` | string | Name of the event | Any non-empty string |

## ChatGPT Integration Notes

### Temperature Handling
- The API accepts decimal temperatures (e.g., 75.5°F)
- It automatically rounds to the nearest temperature from training data
- Available temperatures: 28-83°F (see `/model_info` for exact values)

### Weather Types
- Case-insensitive: "rain", "Rain", "RAIN" all work
- Accepts variations: "Rain", "Rainy" both map to rainy weather
- Default assumption: anything not rain-related is "Clear"

### Event Names
- Unknown event names are handled gracefully
- The model will use other features (date, weather, registered count) for prediction
- For better accuracy, use similar event names from training data when possible

### Error Handling
- Invalid dates return 400 error with clear message
- Out-of-range values return 400 error with constraints
- Model loading issues return 500 error

## Example ChatGPT Workflow

1. **Get weather data** from external API (OpenWeatherMap, etc.)
2. **Extract relevant info**: temperature, conditions
3. **Format the request** according to the schema above
4. **Call the prediction API**
5. **Handle warnings** in the response (temperature rounding, unknown events)

## Sample cURL Command
```bash
curl -X POST "http://your-api-domain.com/predict_event_rsvp" \
  -H "Content-Type: application/json" \
  -d '{
    "event_date": "2024-03-15",
    "registered_count": 500,
    "weather_temperature": 72.3,
    "weather_type": "Clear",
    "special_event": false,
    "event_name": "Weekly Community Meeting"
  }'
```

## Response Interpretation

- **predicted_rsvp_count**: Most likely attendance
- **lower_bound**: Conservative estimate (80% confidence interval)
- **upper_bound**: Optimistic estimate (80% confidence interval)
- **warnings**: Array of any adjustments made to inputs

## Best Practices

1. Always check `/model_info` first to understand available parameters
2. Handle warnings in responses to inform users of any adjustments
3. Use reasonable fallbacks for unknown event names
4. Validate dates are in the future for event planning
5. Consider the confidence interval (lower_bound to upper_bound) for planning

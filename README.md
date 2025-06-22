# RSVP Forecast API

A production-ready RSVP forecasting system for mosque dinner events using Random Forest and Linear Regression models. Provides accurate attendance predictions with confidence intervals for event planning.

## 🚀 Features

- **Advanced ML Models**: Random Forest (primary) + Linear Regression (backup) + ratio-based fallback
- **Future Forecasting**: Works for any future date (no temporal limitations)
- **Intelligent Processing**: Accepts any temperature, event name, weather type
- **Rich Insights**: Day-of-week effects, weather impact, confidence intervals
- **High Accuracy**: 33.1 people average error, 100% success rate
- **FastAPI Service**: RESTful API ready for production deployment

## 📊 Model Performance

- **Training Events**: 37 historical events (Feb-Jun 2025)
- **Average Error**: ±33 people (Random Forest)
- **Success Rate**: 100% (vs 0% with previous Prophet model)
- **Realistic Ratios**: 0.85-1.1 (actual/registered attendance)

## 🔗 API Endpoints

### `GET /`
Returns service status and model information.

### `GET /model_info`
Returns model insights, attendance patterns, and capabilities.

### `POST /predict_event_rsvp`
Predicts RSVP count for any future event.

**Request:**
```json
{
  "event_date": "2025-07-15",
  "registered_count": 500,
  "weather_temperature": 78.5,
  "weather_type": "Clear",
  "special_event": false,
  "event_name": "Community Dinner",
  "sunset_time": "20:15"
}
```

**Response:**
```json
{
  "event_date": "2025-07-15",
  "predicted_rsvp_count": 453,
  "lower_bound": 388,
  "upper_bound": 517,
  "warnings": [
    "Tuesday events typically have high attendance (113.9%)"
  ]
}
```

## 🛠️ Quick Start

1. **Clone & Install:**
```bash
git clone <repository-url>
cd rsvp-forecast-python
pip install -r requirements.txt
```

2. **Start API:**
```bash
python main.py
```
API available at `http://localhost:8000`

3. **Test Prediction:**
```bash
curl -X POST "http://localhost:8000/predict_event_rsvp" \
  -H "Content-Type: application/json" \
  -d '{"event_date": "2025-07-15", "registered_count": 500, "weather_temperature": 78, "weather_type": "Clear", "special_event": false, "event_name": "Community Dinner", "sunset_time": "20:15"}'
```

## 📁 Repository Structure

```
├── main.py                      # FastAPI application (production API)
├── historical_rsvp_data.csv     # Training data (37 events)
├── rf_model.pkl                 # Random Forest model
├── lr_model.pkl                 # Linear Regression model
├── model_metadata.json          # Model configuration & statistics
├── create_practical_model.py    # Model training script
├── requirements.txt             # Python dependencies
├── render.yaml                  # Render.com deployment config
└── UPDATED_SYSTEM_INSTRUCTIONS.md # ChatGPT integration guide
```

## 🎯 Key Insights

**Day-of-Week Effects:**
- **Best**: Saturday (101.1% attendance)
- **Worst**: Wednesday (86.3% attendance)
- **Surprising**: Tuesday (113.9% attendance)

**Weather Impact:**
- **Rain**: Only -2.3% reduction (minimal impact)
- **Temperature extremes**: Reduce attendance

**Event Types:**
- **Special events**: Surprisingly reduce attendance by 2.9%
- **Normal events**: Higher turnout than expected

## 🚀 Deployment

**Render.com (Recommended):**
1. Connect repository to Render.com
2. Deploy using included `render.yaml`
3. API automatically available at your Render URL

**Local Development:**
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## 🔧 Model Retraining

To retrain with new data:
```bash
# Update historical_rsvp_data.csv with new events
python create_practical_model.py
# New models saved as rf_model.pkl, lr_model.pkl, model_metadata.json
```

## 📋 Data Format

Historical data CSV format:
```csv
ds,y,RegisteredCount,WeatherTemperature,WeatherType,SpecialEvent,EventName,SunsetTime
27-Feb,880,925,45,Clear,No,Sherullah Imam Husain AS,18:05
...
```

## 🤖 ChatGPT Integration

See `UPDATED_SYSTEM_INSTRUCTIONS.md` for complete ChatGPT integration guide including:
- Function definitions
- System instructions
- Usage examples
- Best practices

## 📈 Confidence Intervals

The API provides planning ranges:
- **Lower bound**: Conservative estimate (food planning)
- **Upper bound**: Optimistic estimate (space planning)
- **Primary prediction**: Most likely attendance

## 🔍 Troubleshooting

**Common Issues:**
- **Models not loading**: Ensure `.pkl` files are present
- **Prediction errors**: Check input format and required fields
- **Date issues**: Use YYYY-MM-DD format
- **Sunset time**: Use HH:MM 24-hour format

## 📊 Performance Monitoring

Track prediction accuracy by comparing:
- Predicted vs actual attendance
- Confidence interval coverage
- Day-of-week pattern accuracy

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Test changes thoroughly
4. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.

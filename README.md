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

## 📧 Automated RSVP Forecast Emails

The repository includes an automated system that sends daily RSVP forecast emails for upcoming Chicago Jamaat events.

### 🚀 Features

- **Daily Automation**: Runs automatically at 12:00 UTC (7:00 AM CDT / 6:00 AM CST) via GitHub Actions
- **Smart Event Detection**: Fetches upcoming events 0-2 days ahead from Chicago Jamaat API
- **Weather Integration**: Incorporates real-time weather data from Open-Meteo API
- **ML-Powered Forecasts**: Uses the Render-hosted model with retry logic for cold starts
- **Rich Email Reports**: HTML-formatted emails with attendance predictions, weather info, and thaals estimates
- **Zoho SMTP**: Sends via Zoho Mail for reliable delivery

### 🔧 Setup Instructions

#### Required GitHub Secrets
Add these to your repository's **Settings > Secrets and variables > Actions > Secrets**:

```
SMTP_USERNAME=your_zoho_email@domain.com
SMTP_PASSWORD=your_zoho_app_password
```

Optional secrets:
```
SMTP_FROM=custom_from_email@domain.com  # Defaults to SMTP_USERNAME
JAMAAT_API_TOKEN=your_api_token_if_required
RENDER_BASE_URL=https://your-custom-render-url.onrender.com  # Optional override
```

#### Optional GitHub Variables
Configure these in **Settings > Secrets and variables > Actions > Variables**:

```
SMTP_TO=rsvps@panxpan.com  # Email recipient (default)
MOSQUE_LAT=41.7670         # Mosque latitude (Willowbrook, IL)
MOSQUE_LON=-87.9428        # Mosque longitude
EVENT_WINDOW_DAYS=2        # Days ahead to check (0-2 = today, tomorrow, day after)
THAALS_DIVISOR=8          # People per thaal for estimation
LOG_LEVEL=INFO            # Logging verbosity
```

#### Zoho SMTP Setup
1. **Enable 2FA** on your Zoho account
2. **Generate App Password**: Zoho Control Panel > Security > App Passwords
3. **Use App Password** as `SMTP_PASSWORD` (not your regular password)

### 🏃‍♂️ Running Locally

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

3. **Run Forecast Emailer**:
   ```bash
   python src/alert_forecast.py
   ```

### 📊 Email Content

Each forecast email includes:

- **Event Details**: Name, date, registered count
- **RSVP Predictions**: ML-generated attendance forecast with confidence range
- **Weather Information**: Temperature, precipitation, sunset time  
- **Thaals Estimate**: Calculated as `ceil(predicted_attendance / 8)`
- **Smart Notes**: Day-of-week effects, weather impact insights

### 🔧 Troubleshooting

**No emails being sent?**
- Check GitHub Actions logs for error details
- Verify SMTP credentials are correct (use app password for Zoho)
- Ensure the Render API is responding (may have cold start delays)

**Missing events?**
- Check if events exist in the 0-2 day window
- Verify Chicago Jamaat API is accessible
- Check if API token is required and properly set

**Forecast errors?**
- Render API may be in cold start (automatically retried)
- Check weather API availability 
- Verify event data format matches expected structure

**Email formatting issues?**
- HTML emails require proper MIME setup (automatically handled)
- Check recipient email server supports HTML content

### ⚙️ Customization

**Change email schedule**: Edit `.github/workflows/send-forecast-emails.yml` cron expression

**Modify forecast window**: Set `EVENT_WINDOW_DAYS` variable (0 = today only, 2 = today + 2 days ahead)

**Adjust thaals calculation**: Set `THAALS_DIVISOR` variable (default: 8 people per thaal)

**Update email recipient**: Set `SMTP_TO` variable or modify the default in code

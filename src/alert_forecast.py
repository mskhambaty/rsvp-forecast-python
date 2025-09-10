import os
import sys
import math
import time
import json
import smtplib
import logging
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

CHICAGO_TZ = ZoneInfo("America/Chicago")

# Configuration defaults
DEFAULT_SMTP_HOST = "smtp.zoho.com"
DEFAULT_SMTP_PORT = 587
DEFAULT_SMTP_USE_TLS = True

JAMAAT_BASE_URL = "https://www.chicagojamaat.org/api"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
RENDER_BASE_URL = os.getenv("RENDER_BASE_URL", "https://rsvp-forecast-python.onrender.com")

MOSQUE_LAT = float(os.getenv("MOSQUE_LAT", "41.7670"))
MOSQUE_LON = float(os.getenv("MOSQUE_LON", "-87.9428"))

# How many days ahead to include (0 = today, 1 = tomorrow, 2 = day after)
EVENT_WINDOW_DAYS = int(os.getenv("EVENT_WINDOW_DAYS", "2"))

# For Thaals calculation: attendees per thaal
THAALS_DIVISOR = int(os.getenv("THAALS_DIVISOR", "8"))

# Retry configuration for forecast API
FORECAST_RETRIES = int(os.getenv("FORECAST_RETRIES", "3"))
FORECAST_RETRY_DELAY_SECONDS = float(os.getenv("FORECAST_RETRY_DELAY_SECONDS", "12"))

# Email recipients/config
SMTP_HOST = os.getenv("SMTP_HOST", DEFAULT_SMTP_HOST)
SMTP_PORT = int(os.getenv("SMTP_PORT", str(DEFAULT_SMTP_PORT)))
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() in ["1", "true", "yes"]
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USERNAME)
SMTP_TO = os.getenv("SMTP_TO", "rsvps@panxpan.com")

# Chicago Jamaat API token (optional; set if required)
JAMAAT_API_TOKEN = os.getenv("JAMAAT_API_TOKEN", "").strip()

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Event:
    name: str
    date: date
    registered_count: int
    instance_id: Optional[str] = None
    special_event: bool = False


@dataclass
class Weather:
    temperature_f_max: float
    precipitation_sum: float
    sunset_time_hhmm: str
    weather_type: str  # "" or "Rain"


@dataclass
class ForecastResult:
    predicted_rsvp_count: int
    confidence_low: Optional[int]
    confidence_high: Optional[int]
    notes: Optional[str]


def load_env() -> None:
    # Load .env if present
    load_dotenv()
    logger.info("Environment configuration loaded")


def get_target_dates() -> List[date]:
    """Get the list of target dates to check for events (0-EVENT_WINDOW_DAYS days ahead)"""
    # Get Chicago time to ensure we're using the correct local date
    chicago_now = datetime.now(CHICAGO_TZ)
    today = chicago_now.date()
    target_dates = []
    for i in range(EVENT_WINDOW_DAYS + 1):  # 0, 1, 2 days ahead for default EVENT_WINDOW_DAYS=2
        target_dates.append(today + timedelta(days=i))
    logger.info(f"Target dates for events (Chicago timezone): {target_dates}")
    return target_dates


def fetch_upcoming_events() -> List[Event]:
    """Fetch upcoming events from Chicago Jamaat API"""
    target_dates = get_target_dates()
    events = []
    
    try:
        # Build headers if API token is provided
        headers = {}
        if JAMAAT_API_TOKEN:
            headers["Authorization"] = f"Bearer {JAMAAT_API_TOKEN}"
        
        # Fetch events from API
        url = f"{JAMAAT_BASE_URL}/jamaat/event-registrations"
        logger.info(f"Fetching events from: {url}")
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        logger.info(f"Fetched {len(data)} event registrations")
        
        # Process events and filter by target dates
        for event_data in data:
            # Extract event date - handle various possible date keys
            event_date_str = None
            for date_key in ['event_date', 'eventDate', 'date', 'registration_date']:
                if date_key in event_data:
                    event_date_str = event_data[date_key]
                    break
            
            if not event_date_str:
                logger.warning(f"No date found in event data: {event_data}")
                continue
            
            # Parse date (handle YYYY-MM-DD format)
            try:
                if 'T' in event_date_str:
                    event_date = datetime.fromisoformat(event_date_str.replace('Z', '+00:00')).date()
                else:
                    event_date = datetime.strptime(event_date_str, '%Y-%m-%d').date()
            except ValueError:
                logger.warning(f"Could not parse date '{event_date_str}' for event: {event_data}")
                continue
            
            # Filter by target dates
            if event_date in target_dates:
                # Extract event details
                event_name = event_data.get('event_name', event_data.get('eventName', f"Event on {event_date}"))
                registered_count = event_data.get('totalRegistrations', event_data.get('total_registrations', 0))
                instance_id = event_data.get('instance_id', event_data.get('instanceId'))
                
                # Detect special events based on name
                special_event = any(keyword in event_name.lower() for keyword in [
                    'ashara', 'muharram', 'ramadan', 'eid', 'majlis', 'special'
                ])
                
                event = Event(
                    name=event_name,
                    date=event_date,
                    registered_count=registered_count,
                    instance_id=instance_id,
                    special_event=special_event
                )
                events.append(event)
                logger.info(f"Found upcoming event: {event.name} on {event.date} ({event.registered_count} registered)")
    
    except requests.RequestException as e:
        logger.error(f"Failed to fetch events from Chicago Jamaat API: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching events: {e}")
        return []
    
    # Group events by (date, name) to handle multiple events per day
    unique_events = {}
    for event in events:
        key = (event.date, event.name)
        if key not in unique_events:
            unique_events[key] = event
        else:
            # Combine registrations if same event name on same day
            unique_events[key].registered_count += event.registered_count
    
    final_events = list(unique_events.values())
    logger.info(f"Found {len(final_events)} unique upcoming events in target date range")
    return final_events


def get_weather_for_date(event_date: date) -> Optional[Weather]:
    """Get weather data from Open-Meteo for a specific date"""
    try:
        # Open-Meteo forecast API parameters
        params = {
            'latitude': MOSQUE_LAT,
            'longitude': MOSQUE_LON,
            'daily': 'temperature_2m_max,precipitation_sum,sunset',
            'timezone': 'America/Chicago',
            'temperature_unit': 'fahrenheit',
            'start_date': event_date.strftime('%Y-%m-%d'),
            'end_date': event_date.strftime('%Y-%m-%d')
        }
        
        response = requests.get(OPEN_METEO_URL, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'daily' not in data or not data['daily'].get('time'):
            logger.warning(f"No weather data available for {event_date}")
            return None
        
        # Extract data for the specific date
        daily_data = data['daily']
        temp_max = daily_data['temperature_2m_max'][0] if daily_data['temperature_2m_max'] else 70.0
        precip_sum = daily_data['precipitation_sum'][0] if daily_data['precipitation_sum'] else 0.0
        sunset_iso = daily_data['sunset'][0] if daily_data['sunset'] else None
        
        # Parse sunset time to HH:MM format
        if sunset_iso:
            sunset_dt = datetime.fromisoformat(sunset_iso.replace('Z', '+00:00'))
            sunset_chicago = sunset_dt.astimezone(CHICAGO_TZ)
            sunset_time = sunset_chicago.strftime('%H:%M')
        else:
            sunset_time = "19:00"  # Default fallback
        
        # Determine weather type
        weather_type = "Rain" if precip_sum > 0 else ""
        
        weather = Weather(
            temperature_f_max=temp_max,
            precipitation_sum=precip_sum,
            sunset_time_hhmm=sunset_time,
            weather_type=weather_type
        )
        
        logger.info(f"Weather for {event_date}: {temp_max}°F, {precip_sum}mm precip, sunset {sunset_time}")
        return weather
    
    except Exception as e:
        logger.error(f"Failed to get weather for {event_date}: {e}")
        return None


def predict_event_rsvp_with_retry(event: Event, weather: Weather) -> Optional[ForecastResult]:
    """Call the Render model API with retry logic for cold starts"""
    
    payload = {
        "event_date": event.date.strftime('%Y-%m-%d'),
        "registered_count": event.registered_count,
        "weather_temperature": weather.temperature_f_max,
        "weather_type": weather.weather_type,
        "special_event": event.special_event,
        "event_name": event.name,
        "sunset_time": weather.sunset_time_hhmm
    }
    
    logger.info(f"Predicting RSVP for {event.name} on {event.date}")
    
    for attempt in range(FORECAST_RETRIES):
        try:
            response = requests.post(
                f"{RENDER_BASE_URL}/predict_event_rsvp",
                json=payload,
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result_data = response.json()
                
                predicted = result_data.get("predicted_rsvp_count", result_data.get("predicted_rsvp", 0))
                lower = result_data.get("lower_bound", None)
                upper = result_data.get("upper_bound", None)
                warnings = result_data.get("warnings", [])
                
                notes = "; ".join(warnings) if warnings else None
                
                forecast = ForecastResult(
                    predicted_rsvp_count=predicted,
                    confidence_low=lower,
                    confidence_high=upper,
                    notes=notes
                )
                
                logger.info(f"Forecast successful: {predicted} RSVPs (range: {lower}-{upper})")
                return forecast
            
            elif response.status_code == 500 and attempt < FORECAST_RETRIES - 1:
                logger.warning(f"API cold start detected (500 error), retrying in {FORECAST_RETRY_DELAY_SECONDS}s... (attempt {attempt + 1}/{FORECAST_RETRIES})")
                time.sleep(FORECAST_RETRY_DELAY_SECONDS)
                continue
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                break
        
        except requests.RequestException as e:
            logger.error(f"Request failed (attempt {attempt + 1}): {e}")
            if attempt < FORECAST_RETRIES - 1:
                time.sleep(FORECAST_RETRY_DELAY_SECONDS)
                continue
            break
    
    logger.error(f"Failed to get forecast for {event.name} after {FORECAST_RETRIES} attempts")
    return None


def format_html_email(forecasts: List[Tuple[Event, Weather, ForecastResult]]) -> str:
    """Format the forecast results into HTML email content"""
    
    if not forecasts:
        return "<p>No upcoming events found in the forecast window.</p>"
    
    # Sort forecasts by date
    forecasts.sort(key=lambda x: x[0].date)
    
    # Email header
    date_range = f"{forecasts[0][0].date.strftime('%m/%d/%Y')} – {forecasts[-1][0].date.strftime('%m/%d/%Y')}" if len(forecasts) > 1 else forecasts[0][0].date.strftime('%m/%d/%Y')
    
    html = f"""
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        .header {{ background-color: #f4f4f4; padding: 10px; margin-bottom: 20px; }}
        .event {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
        .event-title {{ font-size: 18px; font-weight: bold; color: #333; }}
        .event-date {{ color: #666; font-size: 14px; }}
        .forecast {{ background-color: #e8f5e8; padding: 10px; margin: 10px 0; border-radius: 3px; }}
        .weather {{ background-color: #f0f8ff; padding: 8px; margin: 5px 0; border-radius: 3px; }}
        .thaals {{ background-color: #fff3cd; padding: 8px; margin: 5px 0; border-radius: 3px; }}
        .notes {{ font-style: italic; color: #666; margin-top: 10px; }}
        .footer {{ margin-top: 20px; padding-top: 15px; border-top: 1px solid #ddd; font-size: 12px; color: #888; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>Upcoming Dinner Event Forecasts: {date_range}</h2>
        <p>Automated RSVP forecast generated at {datetime.now(CHICAGO_TZ).strftime('%Y-%m-%d %H:%M:%S CDT')}</p>
    </div>
"""
    
    # Add each event forecast
    for event, weather, forecast in forecasts:
        # Calculate thaals
        estimated_thaals = math.ceil(forecast.predicted_rsvp_count / THAALS_DIVISOR)
        
        # Format date with day of week
        formatted_date = event.date.strftime('%A, %B %d, %Y')
        
        html += f"""
    <div class="event">
        <div class="event-title">{event.name}</div>
        <div class="event-date">{formatted_date}</div>
        
        <div class="forecast">
            <strong>📊 RSVP Forecast:</strong> {forecast.predicted_rsvp_count} attendees<br>
            <strong>📈 Range:</strong> {forecast.confidence_low or 'N/A'} - {forecast.confidence_high or 'N/A'} attendees<br>
            <strong>👥 Registered:</strong> {event.registered_count} people
        </div>
        
        <div class="weather">
            <strong>🌤️ Weather:</strong> {weather.temperature_f_max:.0f}°F, {weather.weather_type or 'Clear'}<br>
            <strong>🌅 Sunset:</strong> {weather.sunset_time_hhmm}<br>
            {"<strong>🌧️ Precipitation:</strong> " + f"{weather.precipitation_sum:.1f}mm<br>" if weather.precipitation_sum > 0 else ""}
        </div>
        
        <div class="thaals">
            <strong>🍽️ Estimated Thaals Needed:</strong> {estimated_thaals} (@ {THAALS_DIVISOR} people per thaal)
        </div>
        
        {f'<div class="notes"><strong>📝 Notes:</strong> {forecast.notes}</div>' if forecast.notes else ''}
    </div>
"""
    
    html += f"""
    <div class="footer">
        <p>This forecast is generated using machine learning models trained on historical RSVP data.<br>
        Weather data from Open-Meteo • RSVP predictions from Chicago Jamaat ML Model</p>
        <p>Questions? Contact the development team.</p>
    </div>
</body>
</html>
"""
    
    return html


def send_email(subject: str, html_content: str) -> bool:
    """Send HTML email via SMTP"""
    
    if not SMTP_USERNAME or not SMTP_PASSWORD:
        logger.error("SMTP credentials not configured")
        return False
    
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = SMTP_FROM
        msg['To'] = SMTP_TO
        
        # Add HTML content
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)
        
        # Connect to SMTP server
        logger.info(f"Connecting to SMTP server: {SMTP_HOST}:{SMTP_PORT}")
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        
        if SMTP_USE_TLS:
            server.starttls()
        
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        
        # Send email
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Email sent successfully to {SMTP_TO}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False


def main():
    """Main function to run the forecast emailer"""
    
    logger.info("Starting RSVP Forecast Emailer")
    
    # Load environment
    load_env()
    
    # Fetch upcoming events
    events = fetch_upcoming_events()
    
    if not events:
        logger.info("No upcoming events found - skipping email")
        return
    
    # Generate forecasts
    successful_forecasts = []
    
    for event in events:
        logger.info(f"Processing event: {event.name} on {event.date}")
        
        # Get weather
        weather = get_weather_for_date(event.date)
        if not weather:
            logger.warning(f"Skipping {event.name} - no weather data")
            continue
        
        # Get forecast
        forecast = predict_event_rsvp_with_retry(event, weather)
        if not forecast:
            logger.warning(f"Skipping {event.name} - forecast failed")
            continue
        
        successful_forecasts.append((event, weather, forecast))
    
    if not successful_forecasts:
        logger.warning("No successful forecasts generated - skipping email")
        return
    
    # Format and send email
    date_range = f"{successful_forecasts[0][0].date.strftime('%m/%d/%Y')} – {successful_forecasts[-1][0].date.strftime('%m/%d/%Y')}" if len(successful_forecasts) > 1 else successful_forecasts[0][0].date.strftime('%m/%d/%Y')
    subject = f"Upcoming Dinner Event Forecasts: {date_range}"
    
    html_content = format_html_email(successful_forecasts)
    
    success = send_email(subject, html_content)
    
    if success:
        logger.info("Forecast email process completed successfully")
    else:
        logger.error("Forecast email process failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
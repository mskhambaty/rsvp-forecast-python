import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def load_and_prepare():
    print("=== CREATING PRACTICAL FORECASTING MODELS ===")
    print("Loading events from historical_rsvp_data.csv")
    df = pd.read_csv("historical_rsvp_data.csv")

    required_columns = ['ds', 'y', 'RegisteredCount', 'WeatherTemperature', 'SunsetTime']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"ERROR: Required column '{col}' not found in CSV.")

    df = df.dropna(subset=required_columns)
    df['SunsetHour'] = df['SunsetTime'].str.split(":").str[0].astype(int)
    df['EventMonth'] = pd.to_datetime(df['ds'], errors='coerce').dt.month.fillna(0).astype(int)
    df['EventWeekday'] = pd.to_datetime(df['ds'], errors='coerce').dt.weekday.fillna(0).astype(int)

    return df

def create_practical_model():
    df = load_and_prepare()

    X = df[['RegisteredCount', 'WeatherTemperature', 'SunsetHour', 'EventMonth', 'EventWeekday']]
    y = df['y']  # Updated from 'ActualCount'

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    print("✅ Random Forest model trained.")

    lr_model = LinearRegression()
    lr_model.fit(X, y)
    print("✅ Linear Regression model trained.")

    with open("rf_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)

    with open("lr_model.pkl", "wb") as f:
        pickle.dump(lr_model, f)

    print("✅ Models saved as rf_model.pkl and lr_model.pkl")

if __name__ == "__main__":
    create_practical_model()

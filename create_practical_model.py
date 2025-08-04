import json
import pandas as pd
import pickle
from datetime import date
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

def load_and_prepare(csv_path: str = "historical_rsvp_data.csv") -> pd.DataFrame:
    print("=== Loading and preparing data")
    df = pd.read_csv(csv_path)

    required = ['ds', 'y', 'RegisteredCount',
                'WeatherTemperature', 'SunsetTime']
    missing = set(required) - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required)
    df['SunsetHour'] = df['SunsetTime'].str.split(":").str[0].astype(int)
    ts = pd.to_datetime(df['ds'], format='%Y-%m-%d', errors='coerce')
    df['EventMonth'] = ts.dt.month.fillna(0).astype(int)
    df['EventWeekday'] = ts.dt.weekday.fillna(0).astype(int)

    return df

def create_practical_model(csv_path: str = "historical_rsvp_data.csv"):
    df = load_and_prepare(csv_path)

    X = df[['RegisteredCount', 'WeatherTemperature',
             'SunsetHour', 'EventMonth', 'EventWeekday']]
    y = df['y']
    features = list(X.columns)

    print("Training random‑forest")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    print("Training linear regression")
    lr = LinearRegression()
    lr.fit(X, y)

    meta = {
        "features": features,
        "target": "y",
        "model_version": date.today().isoformat()
    }

    print("Saving models and metadata")
    with open("rf_model.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open("lr_model.pkl", "wb") as f:
        pickle.dump(lr, f)
    with open("model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Completed―version {meta['model_version']}")

if __name__ == "__main__":
    create_practical_model()

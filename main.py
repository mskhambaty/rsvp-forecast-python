#!/usr/bin/env python3
"""
FastAPI microservice to serve RSVP-based attendance forecasts.
POST /predict_event_rsvp expects:
{
  "registered_count": int,
  "days_to_event": int,
  "venue_good": bool,
  "venue_bad": bool,
  "special_event": bool,
  ...and any other features in model_metadata.json
}
"""

import os, json, pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

app = FastAPI()
FEATURES = []
rf_model, lr_model = None, None

class Event(BaseModel):
    registered_count: int
    days_to_event: int
    venue_good: bool
    venue_bad: bool
    special_event: bool
    # add other features exactly matching those in FEATURES

class ForecastOut(BaseModel):
    rf_prediction: float
    linreg_prediction: float
    feature_values: dict

@app.on_event("startup")
def load_models():
    global FEATURES, rf_model, lr_model
    with open("model_metadata.json") as f:
        md = json.load(f)
    FEATURES = md.get("feature_order", [])
    missing = [fname for fname in ("rf_model.pkl", "lr_model.pkl") if not os.path.exists(fname)]
    if missing:
        raise RuntimeError("Missing model files: " + ", ".join(missing))

    with open("rf_model.pkl", "rb") as f: rf_model = pickle.load(f)
    with open("lr_model.pkl", "rb") as f: lr_model = pickle.load(f)

    print(f"API Startup: loaded RF & LR models with features = {FEATURES}")

@app.post("/predict_event_rsvp", response_model=ForecastOut)
def predict_event_rsvp(evt: Event):
    fv = {}
    for feat in FEATURES:
        try:
            fv[feat] = getattr(evt, feat)
        except AttributeError:
            raise HTTPException(status_code=422,
                                detail=f"Missing feature in request—expected '{feat}'")
    X = np.array([[fv[f] for f in FEATURES]], dtype=float)

    rf_pred = float(rf_model.predict(X)[0])
    lr_pred = float(lr_model.predict(X)[0])

    return ForecastOut(
        rf_prediction=rf_pred,
        linreg_prediction=lr_pred,
        feature_values=fv
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

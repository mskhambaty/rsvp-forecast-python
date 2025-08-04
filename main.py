#!/usr/bin/env python3
"""
FastAPI microservice to serve RSVP‑based forecasts.
POST /forecast  with JSON of event info, returns predicted attendance (y).
"""

import os
import json
import pickle
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

app = FastAPI()
FEATURES = []
rf_model = None
lr_model = None

class Event(BaseModel):
    registered_count: int
    days_to_event: int
    venue_good: bool
    venue_bad: bool
    special_event: bool
    # … all your other features from FEATURES list

class ForecastOut(BaseModel):
    rf_prediction: float
    linreg_prediction: float
    feature_values: dict

@app.on_event("startup")
def load_models():
    global FEATURES, rf_model, lr_model
    with open('features.json') as f:
        md = json.load(f)
    FEATURES = md.get('feature_order', [])
    for fname in ['model_rf.pkl', 'model_linreg.pkl']:
        if not os.path.exists(fname):
            raise RuntimeError(f"Missing {fname}—did you run the training step?")
    with open('model_rf.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('model_linreg.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    print("Models loaded, Features:", FEATURES)

@app.post('/forecast', response_model=ForecastOut)
def forecast(event: Event):
    # Build feature vector in correct order
    fv = {feat: getattr(event, feat) for feat in FEATURES}
    try:
        X = np.array([[fv[feat] for feat in FEATURES]], dtype=float)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error building feature vector: {e}")

    rf_pred = rf_model.predict(X)[0]
    lr_pred = lr_model.predict(X)[0]

    return ForecastOut(
        rf_prediction=float(rf_pred),
        linreg_prediction=float(lr_pred),
        feature_values=fv
    )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv("PORT", 8000)))

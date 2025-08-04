#!/usr/bin/env python3
"""
FastAPI micro-service for RSVP forecasting.

• Dynamic Pydantic schema is auto-generated from model_metadata.json
• /predict_event_rsvp validates against that schema
• Models loaded once via FastAPI lifespan context
"""
from __future__ import annotations

import json
import logging
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from pydantic import Field, create_model

# ---------------------------------------------------------------------------- #
# Static boot-time steps – need metadata to build the schema                   #
# ---------------------------------------------------------------------------- #
METADATA_PATH = Path("model_metadata.json")
RF_MODEL_PATH = Path("rf_model.pkl")
LR_MODEL_PATH = Path("lr_model.pkl")

if not METADATA_PATH.exists():
    raise FileNotFoundError("model_metadata.json missing - run create_practical_model.py first")

meta = json.loads(METADATA_PATH.read_text())
FEATURES: list[str] = meta["feature_cols"]

# Build dynamic input model
field_definitions: Dict[str, tuple] = {
    f: (float, Field(...)) for f in FEATURES  # float accepts int, enforces number
}
Event = create_model(  # type: ignore[call-arg]
    "Event",
    **field_definitions,
    __config__=type("Cfg", (), {"extra": "forbid"}),
)

# ---------------------------------------------------------------------------- #
# Lifespan – load models once, expose via app.state                            #
# ---------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("rsvp_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.rf = pickle.loads(RF_MODEL_PATH.read_bytes())
        app.state.lr = pickle.loads(LR_MODEL_PATH.read_bytes())
        app.state.features = FEATURES
        app.state.meta = meta
        logger.info("Models loaded (version %s)", meta["model_version"])
        yield
    finally:
        logger.info("Shutting down")


app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------------------------- #
# Routes                                                                       #
# ---------------------------------------------------------------------------- #
@app.get("/health")
def health():
    return {"status": "ok", "model_version": app.state.meta["model_version"]}


@app.get("/model_info")
def model_info():
    return app.state.meta


@app.post("/predict_event_rsvp")
def predict_event_rsvp(event: Event):
    """
    Validate JSON against dynamic Event schema, run both models,
    return point forecast + simple min/max band, clamped ≥ 0.
    """
    X_row = pd.DataFrame([[getattr(event, f) for f in app.state.features]],
                         columns=app.state.features)

    try:
        rf_pred = float(app.state.rf.predict(X_row)[0])
        lr_pred = float(app.state.lr.predict(X_row)[0])
    except Exception as exc:  # model or feature mis-match
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    point = max(0, round(rf_pred))
    lo = max(0, min(point, round(lr_pred)))
    hi = max(point, round(lr_pred))

    return {
        "predicted_rsvp": point,
        "lower_bound": lo,
        "upper_bound": hi,
        "model_version": app.state.meta["model_version"],
    }

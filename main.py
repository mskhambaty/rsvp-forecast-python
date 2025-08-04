import json
import logging
import pickle
from datetime import date
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model, Field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rsvp_forecast")

# Utility to load models and metadata at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    md = Path("model_metadata.json")
    if not md.exists():
        logger.error("model_metadata.json missing")
        raise FileNotFoundError("Required metadata file")
    metadata = json.loads(md.read_text())
    features = metadata["features"]

    # Dynamically build Pydantic model
    fields = {}
    for feat in features:
        ft = int if feat.endswith("Count") or feat == "SunsetHour" else float
        fields[feat] = (ft, Field(..., ge=0))
    # Add days_until_event if used by your `predict` logic
    fields["days_to_event"] = (int, Field(..., ge=0))
    Event = create_model(
        "Event", **fields,
        __config__=type("C", (), {"extra": "forbid"})
    )

    app.state.Event = Event
    app.state.features = features
    app.state.rf = pickle.loads(Path("rf_model.pkl").read_bytes())
    app.state.lr = pickle.loads(Path("lr_model.pkl").read_bytes())
    app.state.meta = metadata

    logger.info(f"Models loaded from version {metadata.get('model_version')}")
    yield
    # (Optional cleanup)

app = FastAPI(lifespan=lifespan)

@app.get("/")
def health():
    return {"status": "ok", "model_version": app.state.meta.get("model_version")}

@app.get("/model_info")
def model_info():
    return app.state.meta

@app.post("/predict_event_rsvp")
async def predict_event_rsvp(event: None):
    """
    Validate event via schema, prepare features, predict,
    clamp to zero, produce interval, return JSON.
    """
    Event = app.state.Event
    try:
        ev = Event.model_validate(event) if isinstance(event, dict) else event
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    # Prepare X row
    x = [getattr(ev, f) for f in app.state.features]
    try:
        pred = app.state.rf.predict([x])[0]
    except Exception as e:
        logger.exception("RF failed")
        raise HTTPException(status_code=500, detail="Prediction failed")

    pred = max(0, round(pred))
    lr_pred = round(app.state.lr.predict([x])[0])
    lo = max(0, min(pred, lr_pred))
    hi = pred + abs(pred - lr_pred)

    return {
        "predicted_rsvp_count": pred,
        "lower_bound": lo,
        "upper_bound": hi,
        "model_version": app.state.meta.get("model_version"),
    }

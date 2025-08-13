# backend/app/main.py
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from pathlib import Path
import json

from typing import Optional

from .predict_service import predict_for_request
from .predict_spike_service import spike_for_request


from enum import Enum
from fastapi import Query

class CrimeType(str, Enum):
    all = "all"
    violent = "violent"
    property = "property"

class TimeOfDay(str, Enum):
    all = "all"
    day = "day"
    night = "night"

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
NEIGH_PATH = DATA_DIR / "neighborhoods.geojson"

app = FastAPI(title="SF Crime Mapper API")

# simple CORS (we’ll use a Vite proxy, but this keeps it flexible)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/neighborhoods")
def neighborhoods():
    if not NEIGH_PATH.exists():
        raise HTTPException(status_code=404, detail="neighborhoods.geojson not found")
    data = json.loads(NEIGH_PATH.read_text())
    # ensure each feature has an id for Mapbox feature-state
    for f in data.get("features", []):
        if "id" not in f:
            props = f.get("properties", {})
            for k in ("neighborhood", "name", "nid"):
                if props.get(k):
                    f["id"] = str(props[k])
                    break
    return JSONResponse(data)


@app.get("/api/predict")
def predict(
    date: str,
    crime_type: CrimeType = Query(default=CrimeType.all),
    time_of_day: TimeOfDay = Query(default=TimeOfDay.all),
):
    try:
        day = datetime.fromisoformat(date).date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Use ISO date like 2025-08-12")

    preds = predict_for_request(day, crime_type.value, time_of_day.value)
    return {
        "date": day.isoformat(),
        "filters": {"crime_type": crime_type, "time_of_day": time_of_day},
        "predictions": preds,
    }
    
    
@app.get("/api/spike")
def spike(date: str):
    try:
        day = datetime.fromisoformat(date).date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Use ISO date like 2025-08-12")
    res = spike_for_request(day)
    return {"requested_week": day.isoformat(), **res}



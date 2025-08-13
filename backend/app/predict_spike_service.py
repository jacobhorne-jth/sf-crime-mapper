# backend/app/predict_spike_service.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta, datetime
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Dict, Optional

APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / "models" / "xgb_spike"
COUNTS = APP_DIR / "data" / "processed" / "counts_weekly.csv"

# ---------- utilities ----------

def week_monday(d: date) -> date:
    """Return the Monday for the week containing date d."""
    return d - timedelta(days=d.weekday())

def _load_meta() -> Optional[dict]:
    p = MODELS_DIR / "meta.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

META: Optional[dict] = _load_meta()

# ---------- model container ----------

@dataclass
class SpikeInfo:
    model: object          # XGBClassifier, but typed as object to avoid hard import
    last_week: date

SPIKE_MODELS: Dict[str, SpikeInfo] = {}
_LOADED = False  # ensures we attempt to load only once

def _ensure_loaded() -> None:
    """
    Load all neighborhood XGB models on first use.
    Never raise if xgboost/scikit-learn isn't available or files are missing.
    """
    global _LOADED
    if _LOADED:
        return
    _LOADED = True

    if not META:
        return

    # Try importing lazily so a missing sklearn/xgboost doesn't crash module import
    try:
        from xgboost import XGBClassifier
    except Exception:
        # No xgboost (or its sklearn dependency) – keep models unavailable
        return

    for p in MODELS_DIR.glob("*.json"):
        if p.name == "meta.json":
            continue
        nid = p.stem
        # Keep only neighborhoods listed in meta
        if "neighborhoods" in META and nid not in META["neighborhoods"]:
            continue
        try:
            m = XGBClassifier()
            m.load_model(str(p))  # expects models saved via XGBClassifier.save_model(...)
            info = META["neighborhoods"][nid]
            SPIKE_MODELS[nid] = SpikeInfo(
                model=m,
                last_week=datetime.fromisoformat(info["last_week"]).date(),
            )
        except Exception:
            # If a single neighborhood model fails to load, skip it—others still work
            continue

# ---------- feature builder ----------

def _feature_row(
    hist: pd.DataFrame,
    target_week: date,
    feats: list[str],
    n_lags: int,
) -> Optional[np.ndarray]:
    """Build features for target_week using ONLY data prior to target_week."""
    if hist.empty:
        return None

    h = hist.sort_values("week_start").copy()
    h = h[h["week_start"] < pd.Timestamp(target_week)]  # strictly before target
    y = h["count"].astype(float).values
    if len(y) < max(n_lags, 8):
        return None

    # Lags
    lags = [y[-k] for k in range(1, n_lags + 1)]

    # Rolling means
    ma4 = float(pd.Series(y).rolling(4).mean().iloc[-1])
    ma8 = float(pd.Series(y).rolling(8).mean().iloc[-1])

    # Seasonality (weekly)
    wk = pd.Timestamp(target_week).isocalendar().week
    sin52 = float(np.sin(2 * np.pi * wk / 52.0))
    cos52 = float(np.cos(2 * np.pi * wk / 52.0))

    # Simple trend proxy
    trend = float(len(y) + 1)

    vec = {}
    for i, v in enumerate(lags, start=1):
        vec[f"lag{i}"] = v
    vec.update({"ma4": ma4, "ma8": ma8, "sin52": sin52, "cos52": cos52, "trend": trend})

    return np.array([vec[f] for f in feats], dtype=float)

# ---------- API surface ----------

def spike_for_request(target_day: date) -> dict:
    """
    Return spike probabilities for the requested week.
    If request is beyond next observed week, clamp to next_week.
    Response:
      {
        "served_week": "YYYY-MM-DD" | null,
        "predictions": [
          {"neighborhood_id": "...", "prob": 0.1234, "risk": 12.3},
          ...
        ],
        "available": true|false
      }
    """
    _ensure_loaded()

    # models or meta missing → safe, empty response
    if not META or not SPIKE_MODELS or not COUNTS.exists():
        return {"served_week": None, "predictions": [], "available": False}

    wk = week_monday(target_day)
    feats = META["features"]
    n_lags = int(META["n_lags"])

    df = pd.read_csv(COUNTS, parse_dates=["week_start"])
    base = df[
        (df["crime_type"] == "all") & (df["time_of_day"] == "all")
    ][["neighborhood_id", "week_start", "count"]].copy()

    last_obs = base["week_start"].max().date()
    next_week = last_obs + timedelta(days=7)
    served = wk if wk <= next_week else next_week

    out = []
    for nid, info in SPIKE_MODELS.items():
        hist = base[base["neighborhood_id"] == nid]
        row = _feature_row(hist, served, feats, n_lags)
        if row is None:
            continue
        try:
            # XGBClassifier.predict_proba returns [ [p0, p1] ]
            prob = float(info.model.predict_proba(row.reshape(1, -1))[0, 1])
        except Exception:
            continue
        risk = round(100.0 * max(0.0, min(1.0, prob)), 1)
        out.append(
            {"neighborhood_id": nid, "prob": round(prob, 4), "risk": risk}
        )

    return {"served_week": served.isoformat(), "predictions": out, "available": True}

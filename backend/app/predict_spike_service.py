# backend/app/predict_spike_service.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta, datetime
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / "models" / "xgb_spike"
COUNTS = APP_DIR / "data" / "processed" / "counts_weekly.csv"

@dataclass
class SpikeInfo:
    model: XGBClassifier
    last_week: date

def week_monday(d: date) -> date:
    return d - timedelta(days=d.weekday())

def _load_meta():
    p = MODELS_DIR / "meta.json"
    if not p.exists(): return None
    return json.loads(p.read_text())

META = _load_meta()
SPIKE_MODELS: dict[str, SpikeInfo] = {}

def _load_models():
    if not META: return
    for p in MODELS_DIR.glob("*.json"):
        if p.name == "meta.json": continue
        nid = p.stem
        if nid not in META["neighborhoods"]: continue
        m = XGBClassifier()
        m.load_model(str(p))
        info = META["neighborhoods"][nid]
        SPIKE_MODELS[nid] = SpikeInfo(
            model=m,
            last_week=datetime.fromisoformat(info["last_week"]).date()
        )

_load_models()

def _feature_row(hist: pd.DataFrame, target_week: date, feats: list[str], n_lags: int) -> np.ndarray | None:
    """Build features for 'target_week' using ONLY data before it."""
    if hist.empty: return None
    h = hist.sort_values("week_start").copy()
    # keep only rows BEFORE target
    h = h[h["week_start"] < pd.Timestamp(target_week)]
    y = h["count"].astype(float).values
    if len(y) < max(n_lags, 8): return None

    lags = [y[-k] for k in range(1, n_lags + 1)]
    ma4 = float(pd.Series(y).rolling(4).mean().iloc[-1])
    ma8 = float(pd.Series(y).rolling(8).mean().iloc[-1])

    wk = pd.Timestamp(target_week).isocalendar().week
    sin52 = float(np.sin(2 * np.pi * wk / 52.0))
    cos52 = float(np.cos(2 * np.pi * wk / 52.0))
    trend = float(len(y) + 1)  # step index

    vec = {}
    for i, v in enumerate(lags, start=1):
        vec[f"lag{i}"] = v
    vec.update({"ma4": ma4, "ma8": ma8, "sin52": sin52, "cos52": cos52, "trend": trend})

    return np.array([vec[f] for f in feats], dtype=float)

def spike_for_request(target_day: date) -> dict:
    """Return spike probabilities for the requested week. If the request is beyond next week,
    we serve the closest supported week (past weeks and next week are supported)."""
    if not META or not SPIKE_MODELS or not COUNTS.exists():
        return {"served_week": None, "predictions": []}

    wk = week_monday(target_day)
    feats = META["features"]; n_lags = int(META["n_lags"])

    df = pd.read_csv(COUNTS, parse_dates=["week_start"])
    base = df[(df["crime_type"]=="all") & (df["time_of_day"]=="all")][["neighborhood_id","week_start","count"]].copy()

    # determine a universally supported week: min(max(requested, min+lags), max+7)
    last_obs = base["week_start"].max().date()
    next_week = last_obs + timedelta(days=7)
    # we support any past week (where features can be built) and next_week. Beyond that => clamp to next_week
    served = wk if wk <= next_week else next_week

    out = []
    for nid, info in SPIKE_MODELS.items():
        hist = base[base["neighborhood_id"] == nid]
        row = _feature_row(hist, served, feats, n_lags)
        if row is None:
            continue
        prob = float(info.model.predict_proba(row.reshape(1, -1))[0, 1])
        risk = round(100.0 * max(0.0, min(1.0, prob)), 1)
        out.append({"neighborhood_id": nid, "prob": round(prob, 4), "risk": risk})

    return {"served_week": served.isoformat(), "predictions": out}

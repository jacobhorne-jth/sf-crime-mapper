from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
import json, math
import pandas as pd
from prophet.serialize import model_from_json
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"prophet\..*")

APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / "models" / "prophet"
FACTORS_PATH = APP_DIR / "models" / "segment_factors.json"

@dataclass
class ModelInfo:
    model: any
    last_week: date
    y_min: float
    y_max: float

def week_monday(d: date) -> date:
    return d - timedelta(days=d.weekday())

def load_models() -> dict[str, ModelInfo]:
    meta_path = MODELS_DIR / "meta.json"
    if not meta_path.exists(): return {}
    meta = json.loads(meta_path.read_text()).get("neighborhoods", {})
    out: dict[str, ModelInfo] = {}
    for p in MODELS_DIR.glob("*.json"):
        if p.name == "meta.json": continue
        nid = p.stem
        if nid not in meta: continue
        m = model_from_json(p.read_text())
        info = meta[nid]
        out[nid] = ModelInfo(
            model=m,
            last_week=datetime.fromisoformat(info["last_week"]).date(),
            y_min=float(info["y_min"]), y_max=float(info["y_max"]),
        )
    return out

def load_factors():
    if not FACTORS_PATH.exists(): return {}
    return json.loads(FACTORS_PATH.read_text()).get("neighborhoods", {})

MODELS = load_models()
FACTORS = load_factors()

def _get_factor(nid: str, wk: date, crime_type: str, time_of_day: str) -> float:
    if crime_type == "all" and time_of_day == "all":
        return 1.0
    f = FACTORS.get(nid, {})
    iw = pd.Timestamp(wk).isocalendar().week
    # prefer combo if both specified
    if crime_type != "all" and time_of_day != "all":
        key = f"combo:{crime_type}|{time_of_day}"
        entry = f.get(key)
        if entry:
            return float(entry["by_week"].get(str(int(iw)), entry["overall"]))
    # crime_type only
    if crime_type != "all":
        entry = f.get(f"ct:{crime_type}")
        if entry:
            return float(entry["by_week"].get(str(int(iw)), entry["overall"]))
    # time_of_day only
    if time_of_day != "all":
        entry = f.get(f"tod:{time_of_day}")
        if entry:
            return float(entry["by_week"].get(str(int(iw)), entry["overall"]))
    return 1.0

def _segment_minmax(nid: str, crime_type: str, time_of_day: str) -> tuple[float, float] | None:
    f = FACTORS.get(nid, {})
    # try combo, then ct, then tod
    if crime_type != "all" and time_of_day != "all":
        e = f.get(f"combo:{crime_type}|{time_of_day}")
        if e: return float(e["min"]), float(e["max"])
    if crime_type != "all":
        e = f.get(f"ct:{crime_type}")
        if e: return float(e["min"]), float(e["max"])
    if time_of_day != "all":
        e = f.get(f"tod:{time_of_day}")
        if e: return float(e["min"]), float(e["max"])
    return None

def _scale(yhat: float, lo: float, hi: float) -> float:
    if not math.isfinite(yhat) or hi <= lo: return 0.0
    return max(0.0, min(100.0, round(100.0 * (yhat - lo) / (hi - lo), 1)))

def predict_for_request(target_day: date, crime_type: str, time_of_day: str) -> list[dict]:
    if not MODELS: return []
    wk = week_monday(target_day)
    out = []

    for nid, info in MODELS.items():
        ahead = max(0, (wk - info.last_week).days // 7)
        future = info.model.make_future_dataframe(periods=ahead, freq="W-MON")
        fc = info.model.predict(future)
        row = fc.loc[fc["ds"] == pd.Timestamp(wk)]
        if row.empty: row = fc.iloc[[-1]]

        yhat  = float(row["yhat"].values[0])
        lower = float(row["yhat_lower"].values[0])
        upper = float(row["yhat_upper"].values[0])

        # apply segment factor
        factor = _get_factor(nid, wk, crime_type, time_of_day)
        yhat  = yhat  * factor
        lower = lower * factor
        upper = upper * factor

        # risk scaling: prefer segment min/max; else baseline
        seg_mm = _segment_minmax(nid, crime_type, time_of_day)
        if seg_mm:
            lo, hi = seg_mm
        else:
            lo, hi = info.y_min, info.y_max  # fallback

        # guard against weird values
        if not math.isfinite(yhat): yhat, lower, upper = 0.0, 0.0, 0.0

        out.append({
            "neighborhood_id": nid,
            "mean_incidents": round(yhat, 2),
            "lower": round(lower, 2),
            "upper": round(upper, 2),
            "risk": _scale(yhat, lo, hi),
        })
    return out

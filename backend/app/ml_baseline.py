#placeholder

# backend/app/ml_baseline.py
from __future__ import annotations
from datetime import date
import hashlib, json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

def _norm_id(f):
    # normalize a stable id for each feature (neighborhood)
    if "id" in f: return str(f["id"])
    props = f.get("properties", {})
    nid = props.get("neighborhood") or props.get("name") or props.get("nid")
    if nid: return str(nid)
    return hashlib.sha1(json.dumps(f["geometry"]).encode()).hexdigest()[:8]

def load_neighborhood_ids() -> list[str]:
    gj = json.loads((DATA_DIR / "neighborhoods.geojson").read_text())
    return [_norm_id(f) for f in gj.get("features", [])]

def simple_score(neigh_id: str, day: date) -> float:
    # Deterministic pseudo-score 0..100 (replace with real ML later)
    seed = f"{neigh_id}-{day.isoformat()}".encode()
    h = int(hashlib.sha1(seed).hexdigest()[:6], 16)
    return round((h % 10000) / 100.0, 1)

def predict_for_date(day: date) -> list[dict]:
    return [{"neighborhood_id": nid, "risk": simple_score(nid, day)}
            for nid in load_neighborhood_ids()]

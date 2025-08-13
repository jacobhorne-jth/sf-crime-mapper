# ml/train_prophet.py
from __future__ import annotations
from pathlib import Path
import json, argparse
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json

# your aggregated counts produced earlier
counts_csv = Path("backend/app/data/processed/counts_weekly.csv")

# where we'll save models for the backend to load
models_dir = Path("backend/app/models/prophet")
models_dir.mkdir(parents=True, exist_ok=True)

def weekify(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["ds"] = pd.to_datetime(x["week_start"])  # Prophet expects 'ds'
    x["y"]  = x["count"].astype(float)         # and 'y'
    return x[["ds","y"]].sort_values("ds")

def train_one(series: pd.DataFrame) -> Prophet:
    # We aggregated to weekly; weekly_seasonality should be OFF.
    m = Prophet(
        yearly_seasonality=True,   # keep annual pattern
        weekly_seasonality=False,  # <-- important for weekly data
        daily_seasonality=False,
        seasonality_mode="additive",
        interval_width=0.8,
        changepoint_prior_scale=0.05,   # a touch more regularization
        seasonality_prior_scale=5.0,    # reduce overfit that can blow up coeffs
    )
    # Holidays on weekly data can over-parameterize small series; skip for now.
    # m.add_country_holidays(country_name="US")
    m.fit(series)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default=str(counts_csv))
    ap.add_argument("--min_points", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.infile, parse_dates=["week_start"])

    # start simple: train on all crime + all time-of-day per neighborhood
    base = df[(df["crime_type"]=="all") & (df["time_of_day"]=="all")].copy()

    meta = {}
    trained = 0

    for nid, g in base.groupby("neighborhood_id"):
        series = weekify(g)
        if len(series) < args.min_points:
            continue  # skip short histories
        y_min, y_max = float(series["y"].min()), float(series["y"].max())
        last_week = series["ds"].max().date().isoformat()

        m = train_one(series)
        (models_dir / f"{nid}.json").write_text(model_to_json(m))
        meta[nid] = {"last_week": last_week, "y_min": y_min, "y_max": y_max}
        trained += 1

    (models_dir / "meta.json").write_text(json.dumps({"neighborhoods": meta}, indent=2))
    print(f"trained={trained}  models_dir={models_dir}  meta={models_dir/'meta.json'}")

if __name__ == "__main__":
    main()

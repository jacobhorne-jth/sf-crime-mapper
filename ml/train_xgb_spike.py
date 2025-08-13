# ml/train_xgb_spike.py
from __future__ import annotations
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

COUNTS = Path("backend/app/data/processed/counts_weekly.csv")
OUTDIR = Path("backend/app/models/xgb_spike")
OUTDIR.mkdir(parents=True, exist_ok=True)

N_LAGS = 8
SPIKE_K = 1.0  # label spikes if y_t > mean_{t-1,8} + K * std_{t-1,8}

def make_supervised(g: pd.DataFrame):
    g = g.sort_values("week_start").reset_index(drop=True).copy()
    g["y"] = g["count"].astype(float)

    # rolling baseline (past-only)
    r = g["y"].rolling(8, min_periods=8)
    g["mu8"]  = r.mean().shift(1)
    g["sd8"]  = r.std(ddof=0).shift(1)

    # label: spike if > mu + K*sd
    g["label"] = (g["y"] > (g["mu8"] + SPIKE_K * g["sd8"])).astype(int)

    # features (past-only)
    for k in range(1, N_LAGS + 1):
        g[f"lag{k}"] = g["y"].shift(k)
    g["ma4"] = g["y"].rolling(4).mean().shift(1)
    g["ma8"] = g["y"].rolling(8).mean().shift(1)
    wk = pd.to_datetime(g["week_start"]).dt.isocalendar().week.astype(int)
    g["sin52"] = np.sin(2 * np.pi * wk / 52.0)
    g["cos52"] = np.cos(2 * np.pi * wk / 52.0)
    g["trend"] = np.arange(len(g), dtype=float)

    feats = [f"lag{k}" for k in range(1, N_LAGS + 1)] + ["ma4","ma8","sin52","cos52","trend"]

    # keep rows where we have past info
    g = g.dropna(subset=feats + ["label"]).reset_index(drop=True)
    return g, feats

def train_one(nid: str, g_feat: pd.DataFrame, feats: list[str]):
    X = g_feat[feats].values
    y = g_feat["label"].values
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos < 5 or len(y) < 40:
        return None  # too little data to learn

    clf = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        reg_lambda=1.0,
        random_state=42,
        scale_pos_weight=(neg / pos if pos > 0 else 1.0),
        eval_metric="logloss",
    )
    clf.fit(X, y)
    last_week = pd.to_datetime(g_feat["week_start"]).max().date().isoformat()
    clf.save_model(str(OUTDIR / f"{nid}.json"))
    return {"last_week": last_week}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default=str(COUNTS))
    ap.add_argument("--outdir", default=str(OUTDIR))
    args = ap.parse_args()

    df = pd.read_csv(args.infile, parse_dates=["week_start"])
    base = df[(df["crime_type"]=="all") & (df["time_of_day"]=="all")][["neighborhood_id","week_start","count"]].copy()

    metas = {}
    feats_ref = None
    trained = 0
    for nid, g in base.groupby("neighborhood_id"):
        g_feat, feats = make_supervised(g)
        if feats_ref is None: feats_ref = feats
        meta = train_one(nid, g_feat, feats)
        if meta:
            metas[nid] = meta
            trained += 1

    (OUTDIR / "meta.json").write_text(json.dumps({
        "features": feats_ref,
        "n_lags": N_LAGS,
        "spike_k": SPIKE_K,
        "neighborhoods": metas,
    }, indent=2))
    print(f"trained={trained}, models_dir={OUTDIR}")

if __name__ == "__main__":
    main()

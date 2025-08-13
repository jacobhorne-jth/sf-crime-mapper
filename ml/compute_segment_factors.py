from __future__ import annotations
from pathlib import Path
import argparse, json
import pandas as pd
import numpy as np

counts_csv = Path("backend/app/data/processed/counts_weekly.csv")
out_path   = Path("backend/app/models/segment_factors.json")
out_path.parent.mkdir(parents=True, exist_ok=True)

def iso_week(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.isocalendar().week.astype(int)

def _ratio_df(df_seg: pd.DataFrame, df_base: pd.DataFrame) -> pd.DataFrame:
    df = (df_seg.merge(df_base, on=["neighborhood_id","week_start"], how="left", suffixes=("", "_base"))
                .rename(columns={"count_base":"base"}))
    df = df[df["base"] > 0].copy()
    df["ratio"] = df["count"] / df["base"]
    return df

def _pack(group: pd.DataFrame, key_prefix: str):
    out = {}
    for key, sub in group:
        nid, label = key
        ratios = sub[["iso_week","ratio"]]
        by_week = ratios.groupby("iso_week")["ratio"].mean().to_dict()
        overall = ratios["ratio"].mean()
        # clip to [0,1] because ratios are shares of total
        by_week = {str(int(k)): float(np.clip(v, 0.0, 1.0)) for k, v in by_week.items()}
        overall = float(np.clip(overall if pd.notna(overall) else 0.0, 0.0, 1.0))

        seg_min = int(sub["count"].min()) if len(sub) else 0
        seg_max = int(sub["count"].max()) if len(sub) else 0

        out.setdefault(nid, {})
        out[nid][f"{key_prefix}:{label}"] = {
            "by_week": by_week,
            "overall": overall,
            "min": seg_min,
            "max": seg_max,
        }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default=str(counts_csv))
    ap.add_argument("--outfile", default=str(out_path))
    args = ap.parse_args()

    df = pd.read_csv(args.infile, parse_dates=["week_start"])
    df["iso_week"] = iso_week(df["week_start"])

    base = df[(df["crime_type"]=="all") & (df["time_of_day"]=="all")][
        ["neighborhood_id","week_start","count"]
    ].rename(columns={"count":"count"}).copy()

    # crime_type factors (exclude 'all')
    ct = df[(df["time_of_day"]=="all") & (df["crime_type"]!="all")][
        ["neighborhood_id","week_start","iso_week","crime_type","count"]
    ].copy()
    ct_rat = _ratio_df(ct.rename(columns={"count":"count"}), base)
    ct_grouped = ct_rat.groupby(["neighborhood_id","crime_type"])
    out = _pack(ct_grouped, "ct")

    # time_of_day factors
    tod = df[(df["crime_type"]=="all") & (df["time_of_day"]!="all")][
        ["neighborhood_id","week_start","iso_week","time_of_day","count"]
    ].copy()
    tod_rat = _ratio_df(tod.rename(columns={"count":"count"}), base)
    tod_grouped = tod_rat.groupby(["neighborhood_id","time_of_day"])
    tmp = _pack(tod_grouped, "tod")
    for k, v in tmp.items(): out.setdefault(k, {}).update(v)

    # combo factors (ct & tod both not 'all')
    combo = df[(df["crime_type"]!="all") & (df["time_of_day"]!="all")][
        ["neighborhood_id","week_start","iso_week","crime_type","time_of_day","count"]
    ].copy()
    if not combo.empty:
        combo["label"] = combo["crime_type"] + "|" + combo["time_of_day"]
        combo_rat = _ratio_df(combo.rename(columns={"count":"count"}), base)
        combo_grouped = combo_rat.groupby(["neighborhood_id","label"])
        tmp = _pack(combo_grouped, "combo")
        for k, v in tmp.items(): out.setdefault(k, {}).update(v)

    out_json = {"neighborhoods": out}
    Path(args.outfile).write_text(json.dumps(out_json, indent=2))
    print(f"Wrote {args.outfile}")

if __name__ == "__main__":
    main()

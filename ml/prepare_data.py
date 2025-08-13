from __future__ import annotations
from pathlib import Path
import argparse, json
import pandas as pd

# default paths (tweak if you like)
counts_out = Path("backend/app/data/processed/counts_weekly.csv")
neigh_geojson = Path("backend/app/data/neighborhoods.geojson")

violent_words  = ["assault","robbery","homicide","murder","rape","sex","kidnapping","weapon"]
property_words = ["burglary","larceny","theft","shoplift","vehicle theft","motor vehicle theft","stolen property","arson","vandalism","fraud","embezzlement"]

def map_crime_type(cat: str) -> str:
    c = (cat or "").lower()
    if any(w in c for w in violent_words):  return "violent"
    if any(w in c for w in property_words): return "property"
    return "other"

def map_time_of_day(hour: int) -> str:
    return "day" if 6 <= int(hour) < 18 else "night"

def valid_neighborhood_ids() -> set[str]:
    gj = json.loads(neigh_geojson.read_text())
    ids = set()
    for f in gj["features"]:
        fid = f.get("id") or f["properties"].get("neighborhood") or f["properties"].get("name")
        if fid: ids.add(str(fid).strip())
    return ids

def build_weekly_counts(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    # columns straight from SFPD export/API
    dt  = pd.to_datetime(df["incident_datetime"], errors="coerce")
    cat = df["incident_category"].astype(str)
    # prefer the provided neighborhood name
    if "analysis_neighborhood" not in df.columns:
        raise RuntimeError("Expected 'analysis_neighborhood' in your CSV.")
    neigh = df["analysis_neighborhood"].astype(str).str.strip()

    good = pd.DataFrame({
        "incident_datetime": dt,
        "incident_category": cat,
        "neighborhood_id": neigh
    }).dropna(subset=["incident_datetime","neighborhood_id"])

    good = good[good["neighborhood_id"].isin(valid_neighborhood_ids())]

    good["crime_type"]  = good["incident_category"].map(map_crime_type)
    good["hour"]        = good["incident_datetime"].dt.hour
    good["time_of_day"] = good["hour"].map(map_time_of_day)
    good["week_start"]  = good["incident_datetime"].dt.to_period("W-MON").dt.start_time.dt.date

    grp = (good.groupby(["neighborhood_id","week_start","crime_type","time_of_day"], as_index=False)
                .size().rename(columns={"size":"count"}))

    # precompute simple "all" rollups so filters are easy
    all_rows = (grp.groupby(["neighborhood_id","week_start"], as_index=False)["count"].sum()
                   .assign(crime_type="all", time_of_day="all"))
    ct_rows  = (grp.groupby(["neighborhood_id","week_start","crime_type"], as_index=False)["count"].sum()
                   .assign(time_of_day="all"))
    tod_rows = (grp.groupby(["neighborhood_id","week_start","time_of_day"], as_index=False)["count"].sum()
                   .assign(crime_type="all"))

    out = pd.concat([grp, all_rows, ct_rows, tod_rows], ignore_index=True)
    return out.sort_values(["neighborhood_id","week_start","crime_type","time_of_day"]).reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to SFPD CSV (e.g., backend/app/data/raw/sfpd_incidents.csv)")
    ap.add_argument("--out", default=str(counts_out), help="where to write counts_weekly.csv")
    args = ap.parse_args()

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    result = build_weekly_counts(Path(args.input))
    result.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  rows={len(result):,}")

if __name__ == "__main__":
    main()

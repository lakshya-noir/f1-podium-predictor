"""
Feature engineering for F1 podium prediction.

CRITICAL DESIGN RULE: every rolling/form feature must only use races that
happened BEFORE the race being predicted. If you compute a driver's "average
finish over last 5 races" using .rolling() without .shift(1) first, you leak
the current race's result into its own feature. This is the single most
common way tabular ML "cheats" without you noticing — and the first thing
a sharp interviewer will ask about if you claim strong accuracy.

Expects these files in data/raw/ (from the Kaggle F1 dataset):
  races.csv, results.csv, driver_standings.csv, constructor_standings.csv

NOTE: column names below match the well-known Ergast-based Kaggle F1 dataset
schema as of early 2026. If your downloaded CSVs differ, print(df.columns)
first and adjust — don't assume, verify against your actual file.
"""
import pandas as pd
from pathlib import Path

RAW = Path(__file__).parent.parent / "data" / "raw"
PROCESSED = Path(__file__).parent.parent / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


def build_features(min_year: int = 2018) -> pd.DataFrame:
    races = pd.read_csv(RAW / "races.csv")[["raceId", "year", "round", "date", "circuitId"]]
    results = pd.read_csv(RAW / "results.csv")[
        ["raceId", "driverId", "constructorId", "grid", "positionOrder", "statusId"]
    ]
    driver_standings = pd.read_csv(RAW / "driver_standings.csv")[
        ["raceId", "driverId", "points", "position"]
    ].rename(columns={"points": "driver_points_std", "position": "driver_std_position"})
    constructor_standings = pd.read_csv(RAW / "constructor_standings.csv")[
        ["raceId", "constructorId", "points", "position"]
    ].rename(columns={"points": "constructor_points_std", "position": "constructor_std_position"})

    df = results.merge(races, on="raceId", how="left")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "raceId"]).reset_index(drop=True)

    # Target: podium finish. positionOrder is Ergast's full classification
    # order (includes DNFs ranked last), so this is a reasonable finishing-
    # rank proxy for an MVP. Real limitation, worth naming in your README:
    # it doesn't distinguish "finished 15th" from "crashed on lap 2" beyond
    # rank — a v2 could split on statusId to model reliability separately.
    df["podium"] = (df["positionOrder"] <= 3).astype(int)

    # --- Causal rolling form features ---
    # shift(1) BEFORE rolling: at the row for race N, this uses only races
    # 1..N-1. Verify this yourself: print df[df.driverId==X][['date','positionOrder','driver_form_last5']]
    # and confirm the value at race N does not include race N's own result.
    df["driver_form_last5"] = (
        df.groupby("driverId")["positionOrder"]
        .apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        .reset_index(drop=True)
    )
    df["constructor_form_last5"] = (
        df.groupby("constructorId")["positionOrder"]
        .apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        .reset_index(drop=True)
    )

    # --- Standings going INTO the race ---
    # driver_standings.csv rows are computed AFTER each race. To avoid leakage
    # we shift each driver's standings row forward by one race so "points
    # before this race" reflects the prior race, not this one.
    df = df.merge(driver_standings, on=["raceId", "driverId"], how="left")
    df = df.merge(constructor_standings, on=["raceId", "constructorId"], how="left")
    for col in ["driver_points_std", "driver_std_position",
                "constructor_points_std", "constructor_std_position"]:
        df[col] = df.groupby(
            "driverId" if "driver" in col else "constructorId"
        )[col].shift(1)

    # Drop rows with no history yet (each driver/constructor's first race in
    # the window) — cleaner than filling with 0, which would falsely imply
    # "zero points" instead of "unknown / not enough history".
    feature_cols = [
        "grid", "driver_form_last5", "constructor_form_last5",
        "driver_points_std", "driver_std_position",
        "constructor_points_std", "constructor_std_position",
    ]
    df = df[df["year"] >= min_year].dropna(subset=feature_cols).reset_index(drop=True)

    out_cols = ["raceId", "year", "round", "driverId", "constructorId", "podium"] + feature_cols
    result = df[out_cols]
    result.to_csv(PROCESSED / "features.csv", index=False)
    print(f"Wrote {len(result)} rows, {result['podium'].mean():.1%} podium rate, "
          f"seasons {result['year'].min()}-{result['year'].max()}")
    return result


if __name__ == "__main__":
    build_features()

"""
Generates FAKE data matching the real Kaggle F1 schema, purely so you can
run features.py and train.py end-to-end BEFORE downloading the real dataset,
to confirm nothing crashes.

DO NOT report any metric produced from this synthetic data anywhere — not in
your README, not on your resume, not in an interview. Random synthetic data
should produce ROC-AUC close to 0.5 (a coin flip). If you see a suspiciously
high number here, that would mean the pipeline has a bug, not a good model.
Delete data/raw/*.csv and replace with the real download before you train
anything you intend to report.
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path(__file__).parent.parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

N_DRIVERS, N_CONSTRUCTORS, SEASONS, ROUNDS = 20, 10, range(2018, 2025), 20

races, results, driver_standings, constructor_standings = [], [], [], []
race_id = 0
for year in SEASONS:
    for rnd in range(1, ROUNDS + 1):
        race_id += 1
        races.append({
            "raceId": race_id, "year": year, "round": rnd,
            "date": f"{year}-{(rnd % 12) + 1:02d}-01", "circuitId": rnd % 15,
        })
        grid_order = rng.permutation(range(1, N_DRIVERS + 1))
        finish_order = rng.permutation(range(1, N_DRIVERS + 1))
        for driver_id in range(1, N_DRIVERS + 1):
            constructor_id = ((driver_id - 1) % N_CONSTRUCTORS) + 1
            results.append({
                "raceId": race_id, "driverId": driver_id, "constructorId": constructor_id,
                "grid": int(grid_order[driver_id - 1]),
                "positionOrder": int(finish_order[driver_id - 1]),
                "statusId": 1,
            })
        for driver_id in range(1, N_DRIVERS + 1):
            driver_standings.append({
                "raceId": race_id, "driverId": driver_id,
                "points": int(rng.integers(0, 200)), "position": int(rng.integers(1, N_DRIVERS + 1)),
            })
        for constructor_id in range(1, N_CONSTRUCTORS + 1):
            constructor_standings.append({
                "raceId": race_id, "constructorId": constructor_id,
                "points": int(rng.integers(0, 400)), "position": int(rng.integers(1, N_CONSTRUCTORS + 1)),
            })

pd.DataFrame(races).to_csv(RAW / "races.csv", index=False)
pd.DataFrame(results).to_csv(RAW / "results.csv", index=False)
pd.DataFrame(driver_standings).to_csv(RAW / "driver_standings.csv", index=False)
pd.DataFrame(constructor_standings).to_csv(RAW / "constructor_standings.csv", index=False)
print(f"Wrote synthetic data: {race_id} races, {len(results)} result rows to {RAW}")

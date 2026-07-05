# F1 Podium Predictor

Predicts whether an F1 driver will finish on the podium (top 3), using grid
position, recent form, and championship standing going into the race.

**Status: MVP.** This is a working, evaluated model with a real live demo,
not a finished product. See "Roadmap" for what's not built yet.

## Why this exists

Most public F1 ML tutorials evaluate with a random train/test split, which
leaks future races into training and produces inflated accuracy. This
project uses a **season-based split**. Train on past seasons, test on a
season the model has never seen. This is the honest way to evaluate a
model meant to predict a future race.

## Data

[Kaggle: Formula 1 World Championship dataset](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
(Ergast-based). Download `races.csv`, `results.csv`, `driver_standings.csv`,
`constructor_standings.csv` into `data/raw/`.

## Features

| Feature | Description | Leakage-safe? |
|---|---|---|
| `grid` | Starting grid position | Yes. Known before race starts |
| `driver_form_last5` | Driver's average finish over the previous 5 races | Yes. Shifted before rolling |
| `constructor_form_last5` | Constructor's average finish over the previous 5 races | Yes. Shifted before rolling |
| `driver_points_std` / `driver_std_position` | Driver's championship standing before this race | Yes. Shifted to the previous race |
| `constructor_points_std` / `constructor_std_position` | Constructor's championship standing before this race | Yes. Shifted to the previous race |

## Methodology

- Train on seasons before `TEST_START_YEAR` (see `src/train.py`)
- Test on seasons from `TEST_START_YEAR` onward
- Compare Logistic Regression (baseline) against Random Forest
- Handle class imbalance (~15% podium rate) using `class_weight="balanced"`
- Evaluate using precision, recall, F1-score on the podium class, plus ROC-AUC. Accuracy alone is misleading on an imbalanced dataset.

## Results

| Model | Precision (podium) | Recall (podium) | F1 (podium) | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.43 | 0.88 | 0.58 | 0.915 |
| Random Forest | 0.61 | 0.72 | 0.66 | 0.926 |

The Random Forest model achieved the best overall performance, improving
precision and F1-score while maintaining strong recall. Both models
demonstrate good discriminative ability, with ROC-AUC above **0.91** on an
unseen test season.

## Running it

```bash
pip install -r requirements.txt
python src/generate_sample_data.py   # optional: smoke-test with synthetic data
python src/features.py               # or: drop real CSVs into data/raw/ first
python src/train.py
uvicorn src.api:app --reload --port 8000
python app.py
```

## Live demo

[link. deployed on Hugging Face Spaces]

## Roadmap

The following are planned improvements and are **not implemented yet**:

- Qualifying-session-level features (Q1/Q2/Q3 lap times)
- Weather and circuit-specific features
- Live data ingestion for upcoming race weekends
- Lap-time and pit-stop strategy modeling
- Predicting DNFs/retirements as a separate outcome instead of treating them as low finishing positions
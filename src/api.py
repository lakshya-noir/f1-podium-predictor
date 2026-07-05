"""
FastAPI serving layer. Run with: uvicorn src.api:app --reload --port 8000
Then POST to http://localhost:8000/predict
"""
import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODELS = Path(__file__).parent.parent / "models"
bundle = joblib.load(MODELS / "podium_model.pkl")
model, feature_cols = bundle["model"], bundle["feature_cols"]

app = FastAPI(title="F1 Podium Predictor")


class RaceInput(BaseModel):
    grid: int = Field(..., ge=1, le=24, description="Starting grid position")
    driver_form_last5: float = Field(..., description="Driver's avg finish position, last 5 races")
    constructor_form_last5: float = Field(..., description="Constructor's avg finish position, last 5 races")
    driver_points_std: float = Field(..., description="Driver's championship points before this race")
    driver_std_position: float = Field(..., description="Driver's championship standing before this race")
    constructor_points_std: float = Field(..., description="Constructor's championship points before this race")
    constructor_std_position: float = Field(..., description="Constructor's championship standing before this race")


@app.get("/")
def root():
    return {"status": "ok", "model": "F1 Podium Predictor"}


@app.post("/predict")
def predict(payload: RaceInput):
    row = pd.DataFrame([payload.dict()])[feature_cols]
    proba = float(model.predict_proba(row)[0, 1])
    return {"podium_probability": round(proba, 4), "predicted_podium": proba >= 0.5}

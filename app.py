"""
Gradio demo — deploy this to Hugging Face Spaces for a real, free, live
demo link in minutes (no server to manage, no cost).

Deploy steps (today):
  1. Create a free account at huggingface.co
  2. huggingface.co/new-space -> SDK: Gradio -> create
  3. git clone the Space repo it gives you
  4. Copy this app.py, requirements.txt, and models/podium_model.pkl into it
  5. git add -A && git commit -m "F1 podium predictor" && git push
  6. Space builds automatically, gives you a public URL — that's your live demo link

NOTE: free-tier Spaces sleep after ~48h of inactivity and take ~30-60s to
wake on the next visit. Don't be alarmed if it's slow the first time someone
(e.g. an interviewer) opens the link.
"""
import gradio as gr
import joblib
import pandas as pd
from pathlib import Path

bundle = joblib.load(Path(__file__).parent / "models" / "podium_model.pkl")
model, feature_cols = bundle["model"], bundle["feature_cols"]


def predict_podium(grid, driver_form, constructor_form,
                    driver_points, driver_std_pos,
                    constructor_points, constructor_std_pos):
    row = pd.DataFrame([{
        "grid": grid,
        "driver_form_last5": driver_form,
        "constructor_form_last5": constructor_form,
        "driver_points_std": driver_points,
        "driver_std_position": driver_std_pos,
        "constructor_points_std": constructor_points,
        "constructor_std_position": constructor_std_pos,
    }])[feature_cols]
    proba = model.predict_proba(row)[0, 1]
    label = "🏆 Podium" if proba >= 0.5 else "No podium"
    return f"{label} — probability: {proba:.1%}"


demo = gr.Interface(
    fn=predict_podium,
    inputs=[
        gr.Slider(1, 24, step=1, value=1, label="Starting grid position"),
        gr.Number(value=5.0, label="Driver's avg finish, last 5 races"),
        gr.Number(value=5.0, label="Constructor's avg finish, last 5 races"),
        gr.Number(value=100, label="Driver championship points (before this race)"),
        gr.Number(value=3, label="Driver championship standing (before this race)"),
        gr.Number(value=150, label="Constructor championship points (before this race)"),
        gr.Number(value=2, label="Constructor championship standing (before this race)"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="F1 Podium Predictor",
    description=(
        "Predicts podium (top-3) probability from grid position, recent form, "
        "and championship standing. Trained on historical F1 results with a "
        "season-based train/test split (see README for methodology and "
        "real evaluation metrics)."
    ),
)

if __name__ == "__main__":
    demo.launch()

"""
Train + evaluate the podium classifier.

TIME-BASED SPLIT, NOT RANDOM. This is the whole methodological point of the
project: train on seasons the model would have had access to, test on a
season it's never seen. Change TEST_START_YEAR to whatever makes sense for
the seasons your downloaded dataset actually covers — check features.py's
printed output for your min/max year first.
"""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

PROCESSED = Path(__file__).parent.parent / "data" / "processed"
MODELS = Path(__file__).parent.parent / "models"
MODELS.mkdir(exist_ok=True)

TEST_START_YEAR = 2023  # <-- adjust based on your actual data's year range

FEATURE_COLS = [
    "grid", "driver_form_last5", "constructor_form_last5",
    "driver_points_std", "driver_std_position",
    "constructor_points_std", "constructor_std_position",
]


def main():
    df = pd.read_csv(PROCESSED / "features.csv")

    train = df[df["year"] < TEST_START_YEAR]
    test = df[df["year"] >= TEST_START_YEAR]
    print(f"Train: {len(train)} rows (seasons < {TEST_START_YEAR})")
    print(f"Test:  {len(test)} rows (seasons >= {TEST_START_YEAR})")
    if len(test) == 0:
        raise ValueError(
            f"No test rows — your data doesn't reach {TEST_START_YEAR}. "
            "Lower TEST_START_YEAR to a year within your dataset's range."
        )

    X_train, y_train = train[FEATURE_COLS], train["podium"]
    X_test, y_test = test[FEATURE_COLS], test["podium"]

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    print("\n--- Baseline: Logistic Regression ---")
    logreg = LogisticRegression(class_weight="balanced", max_iter=1000)
    logreg.fit(X_train_s, y_train)
    pred = logreg.predict(X_test_s)
    proba = logreg.predict_proba(X_test_s)[:, 1]
    print(classification_report(y_test, pred, target_names=["no podium", "podium"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")

    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)  # tree models don't need scaling
    pred_rf = rf.predict(X_test)
    proba_rf = rf.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, pred_rf, target_names=["no podium", "podium"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, proba_rf):.3f}")

    print("\nFeature importances (Random Forest):")
    for name, imp in sorted(zip(FEATURE_COLS, rf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.3f}")

    # Save whichever model you decide is better based on the numbers above —
    # don't hardcode "rf wins", check your own printed metrics first.
    joblib.dump({"model": rf, "scaler": scaler, "feature_cols": FEATURE_COLS}, MODELS / "podium_model.pkl")
    print(f"\nSaved model to {MODELS / 'podium_model.pkl'}")
    print(
        "\nIMPORTANT: whatever numbers printed above are YOUR real numbers on "
        "YOUR real data split. Use those exact figures on your resume/README "
        "— not estimates, not rounded-up guesses."
    )


if __name__ == "__main__":
    main()

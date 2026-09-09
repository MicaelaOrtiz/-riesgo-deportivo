import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODELS = BASE / "models"
REPORTS = BASE / "reports"
for p in (DATA, MODELS, REPORTS):
    p.mkdir(exist_ok=True)

FEATURES = [
    "age", "weight_kg", "training_hours", "training_days",
    "sleep_hours", "rest_days", "intensity", "weekly_load",
    "pain", "competition_minutes"
]
TARGET = "risk"

def generate_dataset(n=1800, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 41, n)
    weight = np.clip(rng.normal(82, 13, n), 50, 125)
    hours = np.clip(rng.normal(7, 3, n), 1, 18)
    days = rng.integers(1, 7, n)
    sleep = np.clip(rng.normal(7.2, 1.2, n), 3.5, 10)
    rest = np.clip(7 - days, 0, 6)
    intensity = np.clip(np.rint(rng.normal(6.5, 1.8, n)), 1, 10)
    load = np.clip(hours * intensity * 100 + rng.normal(0, 100, n), 50, 1800)
    pain = np.clip(np.rint(rng.normal(2.2, 2.0, n)), 0, 10)
    competition = np.clip(rng.normal(100, 70, n), 0, 300)

    # Generador de etiquetas sintéticas para demo:
    # combina carga, intensidad, poco sueño/descanso y molestias.
    score = (
        0.018 * load
        + 1.6 * intensity
        + 2.4 * pain
        + 1.0 * hours
        + 0.9 * competition / 30
        - 3.2 * sleep
        - 2.4 * rest
        + rng.normal(0, 7, n)
    )
    risk = np.where(score < 25, "Bajo", np.where(score < 42, "Medio", "Alto"))

    df = pd.DataFrame({
        "age": age, "weight_kg": np.round(weight, 1),
        "training_hours": np.round(hours, 1), "training_days": days,
        "sleep_hours": np.round(sleep, 1), "rest_days": rest,
        "intensity": intensity.astype(int), "weekly_load": np.round(load, 0),
        "pain": pain.astype(int), "competition_minutes": np.round(competition, 0),
        "risk": risk
    })

    # Introducimos algunos problemas intencionalmente para demostrar limpieza.
    missing_idx = rng.choice(n, size=35, replace=False)
    df.loc[missing_idx[:15], "sleep_hours"] = np.nan
    df.loc[missing_idx[15:25], "weekly_load"] = np.nan
    df.loc[missing_idx[25:], "pain"] = np.nan

    bad_idx = rng.choice(n, size=10, replace=False)
    df.loc[bad_idx[:5], "sleep_hours"] = 15
    df.loc[bad_idx[5:], "intensity"] = 15

    df = pd.concat([df, df.iloc[:12]], ignore_index=True)
    return df

def clean_dataset(df):
    df = df.copy()
    df = df.drop_duplicates()

    numeric = FEATURES
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Valores imposibles pasan a NaN.
    ranges = {
        "age": (14, 80), "weight_kg": (35, 180),
        "training_hours": (0, 30), "training_days": (0, 7),
        "sleep_hours": (0, 14), "rest_days": (0, 7),
        "intensity": (1, 10), "weekly_load": (0, 3000),
        "pain": (0, 10), "competition_minutes": (0, 500)
    }
    for col, (lo, hi) in ranges.items():
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    for col in numeric:
        df[col] = df[col].fillna(df[col].median())

    df[TARGET] = df[TARGET].fillna("Medio")
    return df

def main():
    raw = generate_dataset()
    clean = clean_dataset(raw)

    raw.to_csv(DATA / "sports_risk_raw.csv", index=False)
    clean.to_csv(DATA / "sports_risk_clean.csv", index=False)

    X = clean[FEATURES]
    y = clean[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Red neuronal MLP:
    # entrada -> 32 neuronas -> 16 neuronas -> salida de 3 clases.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("neural_network", MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=0.0005,
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=False,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    probs = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    labels = list(model.named_steps["neural_network"].classes_)
    cm = confusion_matrix(y_test, pred, labels=labels)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "classes": labels,
        "test_samples": int(len(X_test)),
        "train_samples": int(len(X_train)),
        "classification_report": report
    }
    with open(REPORTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        REPORTS / "confusion_matrix.csv"
    )
    with open(REPORTS / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, pred, zero_division=0))

    joblib.dump({"model": model, "features": FEATURES}, MODELS / "risk_model.joblib")

    print("Dataset limpio:", len(clean), "filas")
    print("Accuracy:", round(accuracy, 4))
    print(classification_report(y_test, pred, zero_division=0))
    print("Modelo guardado en:", MODELS / "risk_model.joblib")

if __name__ == "__main__":
    main()

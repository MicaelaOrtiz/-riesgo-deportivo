"""
models/evaluator.py
----------------------
Calcula métricas sobre el set de test y las persiste como reportes.
Separado del entrenamiento: entrenar y medir son responsabilidades
distintas, y esto permite reevaluar un modelo ya entrenado sin
reentrenarlo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from src.riesgo_deportivo.models.trainer import TrainTestSplit


@dataclass
class EvaluationResult:
    accuracy: float
    classification_report: str
    confusion_matrix: pd.DataFrame


class ModelEvaluator:
    """Evalúa un modelo entrenado y guarda los reportes en disco."""

    def __init__(self, reports_dir: Path) -> None:
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, model: Pipeline, split: TrainTestSplit) -> EvaluationResult:
        y_pred = model.predict(split.X_test)
        labels = sorted(split.y_test.unique())

        accuracy = accuracy_score(split.y_test, y_pred)
        report = classification_report(split.y_test, y_pred, labels=labels)
        cm = confusion_matrix(split.y_test, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)

        return EvaluationResult(accuracy=accuracy, classification_report=report, confusion_matrix=cm_df)

    def save_reports(self, result: EvaluationResult, n_train: int, n_test: int) -> None:
        (self.reports_dir / "classification_report.txt").write_text(result.classification_report)
        result.confusion_matrix.to_csv(self.reports_dir / "confusion_matrix.csv")

        metrics = {
            "accuracy": round(float(result.accuracy), 4),
            "n_train": n_train,
            "n_test": n_test,
        }
        (self.reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

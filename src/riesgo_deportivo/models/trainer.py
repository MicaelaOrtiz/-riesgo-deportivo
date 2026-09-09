"""
models/trainer.py
--------------------
Orquesta el entrenamiento propiamente dicho: separa train/test y ajusta
el pipeline recibido. No sabe nada de generación de datos, limpieza,
métricas ni persistencia — cada una de esas responsabilidades vive en su
propio módulo (principio de responsabilidad única).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.riesgo_deportivo.domain.constants import FEATURES, TARGET


@dataclass
class TrainTestSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


class ModelTrainer:
    """Entrena un pipeline de sklearn ya construido (ver `ModelFactory`)."""

    def __init__(self, test_size: float = 0.20, seed: int = 42) -> None:
        self.test_size = test_size
        self.seed = seed

    def split(self, df: pd.DataFrame) -> TrainTestSplit:
        X = df[FEATURES]
        y = df[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed, stratify=y
        )
        return TrainTestSplit(X_train, X_test, y_train, y_test)

    def fit(self, model: Pipeline, split: TrainTestSplit) -> Pipeline:
        model.fit(split.X_train, split.y_train)
        return model

"""
api/services/prediction_service.py
-------------------------------------
Patrón **Facade**: la ruta HTTP (`routes/predict.py`) solo llama a
`PredictionService.predict(data)`. Toda la lógica de armar el
DataFrame, invocar el modelo, calcular probabilidades y resolver el
mensaje según el riesgo queda encapsulada acá adentro.
"""

from __future__ import annotations

import pandas as pd

from src.riesgo_deportivo.domain.constants import RISK_MESSAGES
from src.riesgo_deportivo.models.registry import ModelRegistry


class ModelNotLoadedError(RuntimeError):
    """El modelo todavía no fue entrenado / cargado."""


class PredictionService:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def predict(self, data: dict) -> dict:
        if not self.registry.is_loaded:
            raise ModelNotLoadedError(
                "Modelo no encontrado. Ejecutá primero: python train.py"
            )

        model = self.registry.model
        features = self.registry.features
        row = pd.DataFrame([data])[features]

        risk = model.predict(row)[0]
        probabilities = model.predict_proba(row)[0]
        classes = model.classes_

        probs = {str(c): round(float(p), 4) for c, p in zip(classes, probabilities)}
        confidence = float(max(probabilities))

        return {
            "risk": risk,
            "probability": round(confidence, 4),
            "probabilities": probs,
            "message": RISK_MESSAGES.get(risk, ""),
        }

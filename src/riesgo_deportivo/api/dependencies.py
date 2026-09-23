"""
api/dependencies.py
----------------------
Puntos de inyección de dependencias de FastAPI. Mantener esto separado
de las rutas permite testear los servicios sin levantar la API entera.
"""

from __future__ import annotations

from src.riesgo_deportivo.api.services.prediction_service import PredictionService
from src.riesgo_deportivo.models.registry import ModelRegistry


def get_model_registry() -> ModelRegistry:
    return ModelRegistry.instance()


def get_prediction_service() -> PredictionService:
    return PredictionService(registry=get_model_registry())

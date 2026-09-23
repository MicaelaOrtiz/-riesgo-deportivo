"""
api/schemas.py
-----------------
Modelos Pydantic para validar entrada/salida de la API. Los rangos
válidos vienen de `domain/constants.py` para no duplicar los límites
en dos lugares distintos.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.riesgo_deportivo.domain.constants import VALID_RANGES


class AthleteData(BaseModel):
    age: int = Field(..., ge=VALID_RANGES["age"][0], le=VALID_RANGES["age"][1])
    weight_kg: float = Field(..., ge=VALID_RANGES["weight_kg"][0], le=VALID_RANGES["weight_kg"][1])
    training_hours: float = Field(..., ge=VALID_RANGES["training_hours"][0], le=VALID_RANGES["training_hours"][1])
    training_days: int = Field(..., ge=VALID_RANGES["training_days"][0], le=VALID_RANGES["training_days"][1])
    sleep_hours: float = Field(..., ge=VALID_RANGES["sleep_hours"][0], le=VALID_RANGES["sleep_hours"][1])
    rest_days: int = Field(..., ge=VALID_RANGES["rest_days"][0], le=VALID_RANGES["rest_days"][1])
    intensity: int = Field(..., ge=VALID_RANGES["intensity"][0], le=VALID_RANGES["intensity"][1])
    weekly_load: float = Field(..., ge=VALID_RANGES["weekly_load"][0], le=VALID_RANGES["weekly_load"][1])
    pain: int = Field(..., ge=VALID_RANGES["pain"][0], le=VALID_RANGES["pain"][1])
    competition_minutes: float = Field(
        ..., ge=VALID_RANGES["competition_minutes"][0], le=VALID_RANGES["competition_minutes"][1]
    )


class PredictionResponse(BaseModel):
    risk: str
    probability: float
    probabilities: dict[str, float]
    message: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

"""
domain/constants.py
---------------------
Constantes y tipos del dominio del problema (independientes de cómo se
entrena o se sirve el modelo). Cualquier módulo que necesite saber "cuáles
son las variables de entrada" o "cuáles son los niveles de riesgo válidos"
importa desde acá, en vez de repetir listas sueltas por el código.
"""

from __future__ import annotations

from enum import Enum


class RiskLevel(str, Enum):
    """Niveles de riesgo posibles, en el mismo orden que su severidad."""

    BAJO = "Bajo"
    MEDIO = "Medio"
    ALTO = "Alto"


# Variables de entrada del modelo, en el orden en que se le pasan.
FEATURES: list[str] = [
    "age",
    "weight_kg",
    "training_hours",
    "training_days",
    "sleep_hours",
    "rest_days",
    "intensity",
    "weekly_load",
    "pain",
    "competition_minutes",
]

TARGET: str = "risk"

# Rangos fisiológicamente válidos por variable (usados para limpieza y
# validación). Fuera de este rango, un valor se considera dato erróneo.
VALID_RANGES: dict[str, tuple[float, float]] = {
    "age": (14, 80),
    "weight_kg": (35, 180),
    "training_hours": (0, 30),
    "training_days": (0, 7),
    "sleep_hours": (0, 14),
    "rest_days": (0, 7),
    "intensity": (1, 10),
    "weekly_load": (0, 3000),
    "pain": (0, 10),
    "competition_minutes": (0, 500),
}

# Mensajes explicativos por nivel de riesgo, usados por la API.
RISK_MESSAGES: dict[str, str] = {
    RiskLevel.BAJO.value: "El modelo estima un nivel bajo para los datos ingresados.",
    RiskLevel.MEDIO.value: (
        "El modelo estima un nivel intermedio; conviene prestar atención "
        "a recuperación y carga."
    ),
    RiskLevel.ALTO.value: (
        "El modelo estima un nivel elevado; revisar carga y recuperación "
        "con un profesional."
    ),
}

"""
data/cleaner.py
------------------
Limpieza del dataset usando el patrón **Chain of Responsibility**: cada
paso de limpieza es una clase independiente con un método `apply(df)`.
`DataCleaner` los encadena en orden. Esto permite agregar, quitar o
reordenar pasos sin tocar los demás (ej: agregar detección de outliers
más sofisticada el día de mañana).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from src.riesgo_deportivo.domain.constants import FEATURES, TARGET, VALID_RANGES


class CleaningStep(ABC):
    """Un paso individual de limpieza dentro de la cadena."""

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class RemoveDuplicatesStep(CleaningStep):
    """Elimina filas duplicadas exactas."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates().reset_index(drop=True)


class CoerceNumericStep(CleaningStep):
    """Convierte las columnas numéricas, transformando valores inválidos en NaN."""

    def __init__(self, columns: list[str] = FEATURES) -> None:
        self.columns = columns

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df


class ClipOutOfRangeStep(CleaningStep):
    """Convierte a NaN cualquier valor fisiológicamente imposible."""

    def __init__(self, ranges: dict[str, tuple[float, float]] = VALID_RANGES) -> None:
        self.ranges = ranges

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, (lo, hi) in self.ranges.items():
            df.loc[(df[col] < lo) | (df[col] > hi), col] = pd.NA
        return df


class ImputeMissingStep(CleaningStep):
    """Imputa NaN numéricos con la mediana, y el target faltante con 'Medio'."""

    def __init__(self, columns: list[str] = FEATURES, target: str = TARGET) -> None:
        self.columns = columns
        self.target = target

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.columns:
            df[col] = df[col].fillna(df[col].median())
        df[self.target] = df[self.target].fillna("Medio")
        return df


class DataCleaner:
    """Encadena una lista de `CleaningStep` y los aplica en orden.

    Uso:
        cleaner = DataCleaner()          # pasos por defecto
        clean_df = cleaner.clean(raw_df)
    """

    def __init__(self, steps: list[CleaningStep] | None = None) -> None:
        self.steps = steps or [
            RemoveDuplicatesStep(),
            CoerceNumericStep(),
            ClipOutOfRangeStep(),
            ImputeMissingStep(),
        ]

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        for step in self.steps:
            df = step.apply(df)
        return df

"""
data/generator.py
-------------------
Generación del dataset sintético de demo. Se aísla en su propia clase
(DatasetGenerator) para que, si en el futuro se quiere reemplazar por
datos reales, alcance con crear otra clase con la misma interfaz
(`generate() -> pd.DataFrame`) sin tocar el resto del pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from src.riesgo_deportivo.domain.constants import TARGET


class DatasetGenerator(ABC):
    """Interfaz común para cualquier fuente de datos del proyecto."""

    @abstractmethod
    def generate(self) -> pd.DataFrame:
        """Devuelve un DataFrame crudo, sin limpiar."""
        raise NotImplementedError


class SyntheticDatasetGenerator(DatasetGenerator):
    """Genera datos sintéticos reproducibles con fines educativos/demo.

    Incluye a propósito valores faltantes, fuera de rango y duplicados,
    para poder mostrar un pipeline de limpieza real en `data/cleaner.py`.
    """

    def __init__(self, n_samples: int = 1800, seed: int = 42) -> None:
        self.n_samples = n_samples
        self.seed = seed

    def generate(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        n = self.n_samples

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

        # Etiqueta sintética: combina carga, intensidad, sueño/descanso y dolor.
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

        df = pd.DataFrame(
            {
                "age": age,
                "weight_kg": np.round(weight, 1),
                "training_hours": np.round(hours, 1),
                "training_days": days,
                "sleep_hours": np.round(sleep, 1),
                "rest_days": rest,
                "intensity": intensity.astype(int),
                "weekly_load": np.round(load, 0),
                "pain": pain.astype(int),
                "competition_minutes": np.round(competition, 0),
                TARGET: risk,
            }
        )

        df = self._inject_missing_and_outliers(df, rng)
        df = self._inject_duplicates(df)
        return df

    @staticmethod
    def _inject_missing_and_outliers(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """Introduce valores faltantes y fuera de rango a propósito."""
        df = df.copy()
        n = len(df)
        missing_idx = rng.choice(n, size=35, replace=False)
        df.loc[missing_idx[:15], "sleep_hours"] = np.nan
        df.loc[missing_idx[15:25], "weekly_load"] = np.nan
        df.loc[missing_idx[25:], "pain"] = np.nan

        bad_idx = rng.choice(n, size=10, replace=False)
        df.loc[bad_idx[:5], "sleep_hours"] = 15
        df.loc[bad_idx[5:], "intensity"] = 15
        return df

    @staticmethod
    def _inject_duplicates(df: pd.DataFrame, n_duplicates: int = 12) -> pd.DataFrame:
        """Duplica las primeras `n_duplicates` filas (para demostrar la limpieza)."""
        return pd.concat([df, df.iloc[:n_duplicates]], ignore_index=True)

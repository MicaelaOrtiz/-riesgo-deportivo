"""
data/repository.py
---------------------
Patrón **Repository**: aísla el resto del código de *cómo y dónde* se
persisten los datasets. Hoy se guardan como CSV en disco; si mañana se
migra a una base de datos o a un bucket en la nube, solo se reemplaza
esta clase (o se crea una nueva que implemente `DatasetRepository`),
sin tocar generación, limpieza ni entrenamiento.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class DatasetRepository(ABC):
    """Interfaz para guardar y cargar datasets, sin importar el backend."""

    @abstractmethod
    def save_raw(self, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def save_clean(self, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def load_clean(self) -> pd.DataFrame: ...


class CsvDatasetRepository(DatasetRepository):
    """Implementación concreta que persiste los datasets como CSV."""

    def __init__(self, raw_path: Path, clean_path: Path) -> None:
        self.raw_path = raw_path
        self.clean_path = clean_path
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        self.clean_path.parent.mkdir(parents=True, exist_ok=True)

    def save_raw(self, df: pd.DataFrame) -> None:
        df.to_csv(self.raw_path, index=False)

    def save_clean(self, df: pd.DataFrame) -> None:
        df.to_csv(self.clean_path, index=False)

    def load_clean(self) -> pd.DataFrame:
        if not self.clean_path.exists():
            raise FileNotFoundError(
                f"No existe {self.clean_path}. Corré primero el entrenamiento "
                "(python train.py) para generarlo."
            )
        return pd.read_csv(self.clean_path)

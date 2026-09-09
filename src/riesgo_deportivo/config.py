"""
config.py
----------
Configuración centralizada del proyecto: rutas de carpetas y parámetros
generales. Al tener un único lugar con los paths, evitamos que cada
módulo arme sus propias rutas relativas (fuente típica de bugs).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Raíz del proyecto (3 niveles arriba de este archivo: src/riesgo_deportivo/config.py)
BASE_DIR = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class Settings:
    base_dir: Path = BASE_DIR
    data_raw_dir: Path = BASE_DIR / "data" / "raw"
    data_processed_dir: Path = BASE_DIR / "data" / "processed"
    models_dir: Path = BASE_DIR / "models"
    reports_dir: Path = BASE_DIR / "reports"
    templates_dir: Path = BASE_DIR / "app" / "templates"
    static_dir: Path = BASE_DIR / "app" / "static"

    raw_dataset_path: Path = BASE_DIR / "data" / "raw" / "sports_risk_raw.csv"
    clean_dataset_path: Path = BASE_DIR / "data" / "processed" / "sports_risk_clean.csv"
    model_path: Path = BASE_DIR / "models" / "risk_model.joblib"

    random_seed: int = 42
    test_size: float = 0.20

    def ensure_directories(self) -> None:
        for path in (
            self.data_raw_dir,
            self.data_processed_dir,
            self.models_dir,
            self.reports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


# Instancia única de configuración, importable desde cualquier módulo:
# from src.riesgo_deportivo.config import settings
settings = Settings()

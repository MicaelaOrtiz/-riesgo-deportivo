"""
models/registry.py
---------------------
Patrón **Singleton**: garantiza que el modelo entrenado se cargue una
sola vez en memoria (leer un .joblib de disco es costoso), y que toda
la API comparta esa misma instancia en vez de recargarla en cada
request.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import joblib
from sklearn.pipeline import Pipeline

from src.riesgo_deportivo.domain.constants import FEATURES


class ModelRegistry:
    """Carga y guarda el modelo entrenado (+ metadata) como singleton.

    Uso:
        registry = ModelRegistry.instance()
        registry.load(settings.model_path)
        model = registry.model
    """

    _instance: ClassVar["ModelRegistry | None"] = None

    @classmethod
    def instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            obj = cls.__new__(cls)
            obj.model = None
            obj.features = FEATURES
            cls._instance = obj
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def save(self, model: Pipeline, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "features": self.features}, path)
        self.model = model

    def load(self, path: Path) -> None:
        if not path.exists():
            self.model = None
            return
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.features = bundle.get("features", FEATURES)

    def reload(self, path: Path) -> None:
        """Fuerza una recarga desde disco (útil tras reentrenar)."""
        self.load(path)

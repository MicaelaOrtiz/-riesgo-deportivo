"""
models/factory.py
--------------------
Patrón **Factory Method**: centraliza la construcción de pipelines de
modelo (escalado + clasificador) a partir de un nombre. Agregar un nuevo
tipo de modelo (ej. "random_forest") no requiere tocar el training ni la
API: solo registrar un nuevo builder acá.
"""

from __future__ import annotations

from typing import Callable

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _build_mlp(seed: int) -> Pipeline:
    """Red neuronal MLP: entrada -> 32 -> 16 -> salida (3 clases)."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "neural_network",
                MLPClassifier(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    solver="adam",
                    alpha=0.0005,
                    learning_rate_init=0.001,
                    max_iter=1000,
                    early_stopping=False,
                    random_state=seed,
                ),
            ),
        ]
    )


def _build_random_forest(seed: int) -> Pipeline:
    """Alternativa de árbol, útil para comparar contra la red neuronal."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "random_forest",
                RandomForestClassifier(n_estimators=300, random_state=seed),
            ),
        ]
    )


def _build_logistic_regression(seed: int) -> Pipeline:
    """Baseline simple para comparar contra modelos más complejos."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(max_iter=1000, random_state=seed),
            ),
        ]
    )


class ModelFactory:
    """Crea pipelines de modelo (`sklearn.Pipeline`) a partir de un nombre.

    Uso:
        model = ModelFactory.create("mlp", seed=42)

    Para agregar un modelo nuevo, registrar un builder:
        ModelFactory.register("svm", _build_svm)
    """

    _builders: dict[str, Callable[[int], Pipeline]] = {
        "mlp": _build_mlp,
        "random_forest": _build_random_forest,
        "logistic_regression": _build_logistic_regression,
    }

    @classmethod
    def create(cls, model_type: str, seed: int = 42) -> Pipeline:
        try:
            builder = cls._builders[model_type]
        except KeyError as exc:
            available = ", ".join(cls._builders)
            raise ValueError(
                f"Tipo de modelo desconocido: '{model_type}'. Disponibles: {available}"
            ) from exc
        return builder(seed)

    @classmethod
    def register(cls, model_type: str, builder: Callable[[int], Pipeline]) -> None:
        """Permite extender la fábrica con nuevos tipos de modelo sin editar esta clase."""
        cls._builders[model_type] = builder

    @classmethod
    def available_models(cls) -> list[str]:
        return list(cls._builders)

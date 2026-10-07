"""
Tests de la API: /health, /predict y la página principal.

Usa un modelo entrenado "de verdad" (rápido, con pocos datos) para no
depender de que ya exista models/risk_model.joblib en disco.
"""

import pytest
from fastapi.testclient import TestClient

from src.riesgo_deportivo.api.app import create_app
from src.riesgo_deportivo.config import Settings
from src.riesgo_deportivo.data.cleaner import DataCleaner
from src.riesgo_deportivo.data.generator import SyntheticDatasetGenerator
from src.riesgo_deportivo.models.factory import ModelFactory
from src.riesgo_deportivo.models.registry import ModelRegistry
from src.riesgo_deportivo.models.trainer import ModelTrainer


@pytest.fixture
def client(tmp_path):
    # Entrenar un modelo chico y guardarlo donde la app lo va a buscar.
    raw = SyntheticDatasetGenerator(n_samples=300, seed=1).generate()
    clean = DataCleaner().clean(raw)
    trainer = ModelTrainer(test_size=0.2, seed=1)
    split = trainer.split(clean)
    model = trainer.fit(ModelFactory.create("mlp", seed=1), split)

    model_path = tmp_path / "modelo.joblib"
    ModelRegistry.instance().save(model, model_path)

    test_settings = Settings(model_path=model_path)
    app = create_app(test_settings)
    return TestClient(app)


def test_health_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_home_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_predict_returns_valid_risk(client):
    payload = {
        "age": 28, "weight_kg": 82, "training_hours": 15, "training_days": 6,
        "sleep_hours": 4, "rest_days": 0, "intensity": 10, "weekly_load": 1700,
        "pain": 9, "competition_minutes": 280,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert body["risk"] in {"Bajo", "Medio", "Alto"}
    assert 0.0 <= body["probability"] <= 1.0
    assert abs(sum(body["probabilities"].values()) - 1.0) < 0.01


def test_predict_rejects_invalid_input(client):
    payload = {
        "age": 999,  # fuera de rango a propósito
        "weight_kg": 82, "training_hours": 15, "training_days": 6,
        "sleep_hours": 4, "rest_days": 0, "intensity": 10, "weekly_load": 1700,
        "pain": 9, "competition_minutes": 280,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # error de validación de Pydantic

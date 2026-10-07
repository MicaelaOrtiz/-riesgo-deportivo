"""
Tests de la capa de modelos: factory, trainer, evaluator y registry.
"""

import pytest

from src.riesgo_deportivo.data.cleaner import DataCleaner
from src.riesgo_deportivo.data.generator import SyntheticDatasetGenerator
from src.riesgo_deportivo.models.evaluator import ModelEvaluator
from src.riesgo_deportivo.models.factory import ModelFactory
from src.riesgo_deportivo.models.registry import ModelRegistry
from src.riesgo_deportivo.models.trainer import ModelTrainer


@pytest.fixture
def clean_dataset():
    raw = SyntheticDatasetGenerator(n_samples=300, seed=1).generate()
    return DataCleaner().clean(raw)


def test_factory_creates_known_models():
    for model_type in ModelFactory.available_models():
        model = ModelFactory.create(model_type, seed=1)
        assert model is not None


def test_factory_raises_on_unknown_model():
    with pytest.raises(ValueError):
        ModelFactory.create("arquitectura_inexistente")


def test_trainer_split_respects_test_size(clean_dataset):
    trainer = ModelTrainer(test_size=0.2, seed=1)
    split = trainer.split(clean_dataset)
    total = len(split.X_train) + len(split.X_test)
    assert total == len(clean_dataset)
    assert abs(len(split.X_test) / total - 0.2) < 0.05


def test_trainer_fit_returns_fitted_model(clean_dataset):
    trainer = ModelTrainer(test_size=0.2, seed=1)
    split = trainer.split(clean_dataset)
    model = ModelFactory.create("mlp", seed=1)
    fitted = trainer.fit(model, split)

    predictions = fitted.predict(split.X_test)
    assert len(predictions) == len(split.X_test)


def test_evaluator_produces_valid_metrics(tmp_path, clean_dataset):
    trainer = ModelTrainer(test_size=0.2, seed=1)
    split = trainer.split(clean_dataset)
    model = trainer.fit(ModelFactory.create("mlp", seed=1), split)

    evaluator = ModelEvaluator(reports_dir=tmp_path)
    result = evaluator.evaluate(model, split)

    assert 0.0 <= result.accuracy <= 1.0
    assert not result.confusion_matrix.empty

    evaluator.save_reports(result, n_train=len(split.X_train), n_test=len(split.X_test))
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "classification_report.txt").exists()
    assert (tmp_path / "confusion_matrix.csv").exists()


def test_registry_save_and_load_roundtrip(tmp_path, clean_dataset):
    trainer = ModelTrainer(test_size=0.2, seed=1)
    split = trainer.split(clean_dataset)
    model = trainer.fit(ModelFactory.create("mlp", seed=1), split)

    model_path = tmp_path / "modelo.joblib"
    registry = ModelRegistry.instance()
    registry.save(model, model_path)
    assert registry.is_loaded

    # Simular un reinicio: limpiar el modelo en memoria y recargarlo del disco
    registry.model = None
    assert not registry.is_loaded

    registry.load(model_path)
    assert registry.is_loaded
    assert registry.model.predict(split.X_test) is not None

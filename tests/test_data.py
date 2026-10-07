"""
Tests de la capa de datos: generación, limpieza y repositorio.
"""

import pandas as pd
import pytest

from src.riesgo_deportivo.data.cleaner import DataCleaner
from src.riesgo_deportivo.data.generator import SyntheticDatasetGenerator
from src.riesgo_deportivo.data.repository import CsvDatasetRepository
from src.riesgo_deportivo.domain.constants import FEATURES, TARGET


def test_generator_produces_expected_columns():
    df = SyntheticDatasetGenerator(n_samples=200, seed=1).generate()
    assert set(FEATURES + [TARGET]).issubset(df.columns)
    assert len(df) >= 200  # puede tener más por los duplicados inyectados a propósito


def test_generator_injects_quality_issues_on_purpose():
    df = SyntheticDatasetGenerator(n_samples=500, seed=1).generate()
    assert df.isna().sum().sum() > 0, "debería tener nulos inyectados a propósito"
    assert df.duplicated().sum() > 0, "debería tener duplicados inyectados a propósito"


def test_cleaner_removes_all_issues():
    raw = SyntheticDatasetGenerator(n_samples=500, seed=1).generate()
    clean = DataCleaner().clean(raw)

    assert clean.isna().sum().sum() == 0
    assert clean.duplicated().sum() == 0
    assert set(clean[TARGET].unique()).issubset({"Bajo", "Medio", "Alto"})


def test_cleaner_keeps_values_within_valid_ranges():
    raw = SyntheticDatasetGenerator(n_samples=500, seed=1).generate()
    clean = DataCleaner().clean(raw)

    assert clean["age"].between(14, 80).all()
    assert clean["intensity"].between(1, 10).all()


def test_csv_repository_roundtrip(tmp_path):
    raw_path = tmp_path / "raw.csv"
    clean_path = tmp_path / "clean.csv"
    repo = CsvDatasetRepository(raw_path=raw_path, clean_path=clean_path)

    df = SyntheticDatasetGenerator(n_samples=50, seed=1).generate()
    repo.save_raw(df)
    repo.save_clean(df)

    assert raw_path.exists()
    assert clean_path.exists()

    loaded = repo.load_clean()
    assert isinstance(loaded, pd.DataFrame)
    assert len(loaded) == len(df)


def test_csv_repository_raises_if_missing(tmp_path):
    repo = CsvDatasetRepository(raw_path=tmp_path / "raw.csv", clean_path=tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError):
        repo.load_clean()

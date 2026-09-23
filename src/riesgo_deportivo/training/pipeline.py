"""
training/pipeline.py
-----------------------
Patrón **Facade**: expone un único método `run()` que por dentro
orquesta generación de datos, limpieza, persistencia, entrenamiento,
evaluación y guardado del modelo. El resto del proyecto (el script
`train.py` de la raíz) no necesita conocer estos detalles internos.
"""

from __future__ import annotations

from src.riesgo_deportivo.config import Settings, settings
from src.riesgo_deportivo.data.cleaner import DataCleaner
from src.riesgo_deportivo.data.generator import DatasetGenerator, SyntheticDatasetGenerator
from src.riesgo_deportivo.data.repository import CsvDatasetRepository
from src.riesgo_deportivo.models.evaluator import ModelEvaluator
from src.riesgo_deportivo.models.factory import ModelFactory
from src.riesgo_deportivo.models.registry import ModelRegistry
from src.riesgo_deportivo.models.trainer import ModelTrainer


class TrainingPipeline:
    """Orquesta el flujo completo de entrenamiento de punta a punta."""

    def __init__(
        self,
        settings_: Settings = settings,
        generator: DatasetGenerator | None = None,
        model_type: str = "mlp",
    ) -> None:
        self.settings = settings_
        self.generator = generator or SyntheticDatasetGenerator(seed=settings_.random_seed)
        self.model_type = model_type

        self.cleaner = DataCleaner()
        self.repository = CsvDatasetRepository(
            raw_path=self.settings.raw_dataset_path,
            clean_path=self.settings.clean_dataset_path,
        )
        self.trainer = ModelTrainer(test_size=self.settings.test_size, seed=self.settings.random_seed)
        self.evaluator = ModelEvaluator(reports_dir=self.settings.reports_dir)
        self.registry = ModelRegistry.instance()

    def run(self) -> dict:
        """Ejecuta el pipeline completo y devuelve un resumen de resultados."""
        self.settings.ensure_directories()

        print("1/6 · Generando dataset sintético...")
        raw_df = self.generator.generate()
        self.repository.save_raw(raw_df)

        print("2/6 · Limpiando datos...")
        clean_df = self.cleaner.clean(raw_df)
        self.repository.save_clean(clean_df)

        print("3/6 · Separando train/test...")
        split = self.trainer.split(clean_df)

        print(f"4/6 · Entrenando modelo ({self.model_type})...")
        model = ModelFactory.create(self.model_type, seed=self.settings.random_seed)
        model = self.trainer.fit(model, split)

        print("5/6 · Evaluando modelo...")
        result = self.evaluator.evaluate(model, split)
        self.evaluator.save_reports(result, n_train=len(split.X_train), n_test=len(split.X_test))

        print("6/6 · Guardando modelo entrenado...")
        self.registry.save(model, self.settings.model_path)

        summary = {
            "model_type": self.model_type,
            "accuracy": round(float(result.accuracy), 4),
            "n_train": len(split.X_train),
            "n_test": len(split.X_test),
            "model_path": str(self.settings.model_path),
        }
        print(f"\nListo. Accuracy en test: {summary['accuracy']:.2%}")
        print(f"Modelo guardado en: {summary['model_path']}")
        return summary

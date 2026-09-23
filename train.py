"""
train.py
---------
Punto de entrada para entrenar el modelo. La lógica real vive en
src/riesgo_deportivo/training/pipeline.py (patrón Facade); este script
es solo el "botón" que la dispara.

Uso:
    python train.py
"""

from src.riesgo_deportivo.training.pipeline import TrainingPipeline

if __name__ == "__main__":
    TrainingPipeline().run()

"""
api/app.py
------------
Patrón **Application Factory**: `create_app()` construye y devuelve la
app de FastAPI ya configurada (rutas, archivos estáticos, carga del
modelo). Tenerlo como función (en vez de un objeto global armado al
importar el módulo) facilita testear con distintas configuraciones.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.riesgo_deportivo.api.routes import health, pages, predict
from src.riesgo_deportivo.config import Settings, settings
from src.riesgo_deportivo.models.registry import ModelRegistry


def create_app(settings_: Settings = settings) -> FastAPI:
    app = FastAPI(
        title="Sports Risk Neural Network API",
        description="API educativa para estimar riesgo deportivo experimental.",
        version="1.0.0",
    )

    # Cargar el modelo entrenado (si existe) en el registry singleton.
    registry = ModelRegistry.instance()
    registry.load(settings_.model_path)

    app.mount("/static", StaticFiles(directory=str(settings_.static_dir)), name="static")

    app.include_router(pages.router)
    app.include_router(health.router)
    app.include_router(predict.router)

    return app


# Instancia por defecto, usada por uvicorn (ver app/main.py en la raíz).
app = create_app()

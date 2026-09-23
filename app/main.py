"""
app/main.py
-------------
Punto de entrada para levantar la API. La app real se construye en
src/riesgo_deportivo/api/app.py (Application Factory); este módulo
solo la importa para que el comando de siempre siga funcionando:

    uvicorn app.main:app --reload
"""

from src.riesgo_deportivo.api.app import app  # noqa: F401

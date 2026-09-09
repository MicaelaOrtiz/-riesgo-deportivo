# Predicción de riesgo deportivo con Red Neuronal

Proyecto educativo completo: dataset, limpieza, entrenamiento, evaluación, red neuronal, API FastAPI e interfaz web con predicción en tiempo real.

> **Importante:** el modelo es experimental y educativo. No diagnostica lesiones ni reemplaza una evaluación médica/deportiva profesional. El dataset incluido se genera de forma sintética para demostrar el pipeline completo.

## 1. Instalar

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Generar dataset, limpiar, entrenar y evaluar

```bash
python train.py
```

Esto crea:
- `data/sports_risk_raw.csv`: datos sintéticos sin limpiar.
- `data/sports_risk_clean.csv`: datos limpios.
- `models/risk_model.joblib`: red neuronal entrenada + escalador.
- `reports/metrics.json`: métricas de evaluación.
- `reports/confusion_matrix.csv`: matriz de confusión.
- `reports/classification_report.txt`: reporte por clase.

## 3. Ejecutar API + web

```bash
uvicorn app.main:app --reload
```

Abrir en el navegador:
`http://127.0.0.1:8000`

Documentación automática de API:
`http://127.0.0.1:8000/docs`

## 4. API

POST `/predict`

Ejemplo JSON:

```json
{
  "age": 25,
  "weight_kg": 84,
  "training_hours": 8,
  "training_days": 5,
  "sleep_hours": 6,
  "rest_days": 1,
  "intensity": 8,
  "weekly_load": 850,
  "pain": 3,
  "competition_minutes": 120
}
```

Respuesta:

```json
{
  "risk": "Medio",
  "probability": 0.63,
  "probabilities": {
    "Bajo": 0.08,
    "Medio": 0.63,
    "Alto": 0.29
  },
  "message": "..."
}
```

## Arquitectura

```text
Formulario web
     ↓
JavaScript fetch()
     ↓
FastAPI /predict
     ↓
Validación Pydantic
     ↓
Escalado de variables
     ↓
MLPClassifier (red neuronal)
     ↓
Probabilidades
     ↓
Resultado en pantalla
```

## Variables

- `age`: edad
- `weight_kg`: peso
- `training_hours`: horas de entrenamiento semanales
- `training_days`: días de entrenamiento semanales
- `sleep_hours`: horas promedio de sueño
- `rest_days`: días de descanso
- `intensity`: intensidad percibida (1-10)
- `weekly_load`: carga semanal aproximada
- `pain`: molestia actual percibida (0-10)
- `competition_minutes`: minutos de competición recientes

La variable objetivo es `risk`: Bajo, Medio o Alto.

## Cómo defenderlo en una exposición

1. Se parte de datos de deportistas.
2. Se limpian valores faltantes, duplicados y valores fuera de rango.
3. Se separan características (`X`) y etiqueta (`y`).
4. Se dividen los datos en entrenamiento y prueba.
5. Se estandarizan las variables.
6. Se entrena una red neuronal MLP.
7. Se evalúa con accuracy, precision, recall, F1 y matriz de confusión.
8. FastAPI expone el modelo mediante `/predict`.
9. La interfaz web manda los datos y muestra la predicción inmediatamente.

### Nota sobre el dataset

Para que el proyecto pueda ejecutarse sin depender de una descarga externa, `train.py` genera un dataset sintético reproducible. En un trabajo académico, una mejora importante sería reemplazarlo por datos reales/anónimos o un dataset público validado, manteniendo el mismo pipeline.

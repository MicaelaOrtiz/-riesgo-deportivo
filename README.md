# Predicción de Riesgo Deportivo — Red Neuronal

Proyecto educativo que entrena una red neuronal (MLP) para estimar el nivel
de riesgo (**Bajo / Medio / Alto**) de un atleta a partir de sus hábitos de
entrenamiento, sueño y recuperación. Incluye el pipeline completo: generación
de datos, limpieza, entrenamiento, evaluación, y una API + interfaz web para
predecir en tiempo real.

> **Importante:** el modelo es experimental y educativo. El dataset es
> **sintético** (generado por código, no son datos reales de atletas). No
> reemplaza una evaluación médica o deportiva profesional.

## Arquitectura del proyecto

El proyecto está organizado en capas, cada una con una responsabilidad
específica, aplicando patrones de diseño de software:

```
riesgo_deportivo_nn/
├── train.py                 # Entry point: entrena el modelo
├── app/
│   ├── main.py               # Entry point: levanta la API web
│   ├── static/                # CSS y JS de la interfaz
│   └── templates/              # HTML de la interfaz
├── src/riesgo_deportivo/
│   ├── config.py                # Configuración centralizada (rutas, seeds)
│   ├── domain/
│   │   └── constants.py          # FEATURES, RiskLevel, rangos válidos
│   ├── data/
│   │   ├── generator.py           # Generación del dataset sintético
│   │   ├── cleaner.py              # Limpieza de datos
│   │   └── repository.py           # Guardado/carga de datasets (CSV)
│   ├── models/
│   │   ├── factory.py               # Construcción de modelos (MLP, RF, etc.)
│   │   ├── trainer.py                # Split + entrenamiento
│   │   ├── evaluator.py               # Métricas y reportes
│   │   └── registry.py                # Carga del modelo entrenado
│   ├── training/
│   │   └── pipeline.py                 # Orquesta todo el entrenamiento
│   └── api/
│       ├── app.py                       # Arma la app FastAPI
│       ├── schemas.py                    # Validación de request/response
│       ├── dependencies.py                # Inyección de dependencias
│       ├── routes/                         # health.py, predict.py, pages.py
│       └── services/
│           └── prediction_service.py        # Lógica de predicción
├── tests/                    # Tests automáticos (pytest)
├── data/                      # Datasets generados (raw y procesados)
├── models/                     # Modelo entrenado (.joblib)
└── reports/                     # Métricas de la última evaluación
```

## Patrones de diseño aplicados

| Patrón | Dónde | Para qué |
|---|---|---|
| **Factory Method** | `models/factory.py` | Crear distintos tipos de modelo (MLP, Random Forest, Regresión Logística) sin acoplar el resto del código a uno solo. Agregar un modelo nuevo no requiere tocar el training ni la API. |
| **Chain of Responsibility** | `data/cleaner.py` | La limpieza de datos es una cadena de pasos independientes (quitar duplicados → validar rangos → imputar faltantes), fáciles de agregar/quitar/reordenar. |
| **Repository** | `data/repository.py` | Aísla cómo se guardan/leen los datasets (hoy CSV) del resto del pipeline. Si mañana se migra a una base de datos, solo cambia esta clase. |
| **Singleton** | `models/registry.py` | El modelo entrenado (`.joblib`) se carga una sola vez en memoria y lo comparte toda la API, en vez de releerlo del disco en cada request. |
| **Facade** | `training/pipeline.py`, `api/services/prediction_service.py` | Un único método (`TrainingPipeline.run()`, `PredictionService.predict()`) esconde por dentro varios pasos complejos coordinados. |
| **Application Factory** | `api/app.py` | `create_app()` arma la aplicación FastAPI de forma configurable, en vez de un objeto global fijo — más fácil de testear. |

## 1. Instalar

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Entrenar el modelo

```bash
python train.py
```

Esto ejecuta el pipeline completo (ver `src/riesgo_deportivo/training/pipeline.py`):

1. Genera el dataset sintético (`data/raw/sports_risk_raw.csv`)
2. Lo limpia (`data/processed/sports_risk_clean.csv`)
3. Separa train/test
4. Entrena la red neuronal (MLP, 2 capas ocultas de 32 y 16 neuronas)
5. Evalúa el modelo (`reports/metrics.json`, `classification_report.txt`, `confusion_matrix.csv`)
6. Guarda el modelo entrenado (`models/risk_model.joblib`)

## 3. Levantar la API + interfaz web

```bash
uvicorn app.main:app --reload
```

Abrí `http://127.0.0.1:8000` en el navegador para cargar datos de un atleta
y ver la predicción de riesgo. Endpoints disponibles:

- `GET /` — interfaz web
- `GET /health` — estado de la API y si el modelo está cargado
- `POST /predict` — predicción (recibe un JSON con las 10 variables del atleta)

## 4. Correr los tests

```bash
pytest -q
```

Valida que el pipeline de datos, el entrenamiento y la API funcionen
correctamente de punta a punta.

## Resultado actual del modelo

Con el dataset sintético balanceado (ver `domain/constants.py` y
`data/generator.py`):

| Clase de riesgo | Precisión | Recall | F1-score |
|---|---|---|---|
| Bajo | 0.75 | 0.70 | 0.73 |
| Medio | 0.52 | 0.53 | 0.52 |
| Alto | 0.64 | 0.68 | 0.66 |

**Accuracy general: ~63%**

Nota: la clase "Medio" es la más difícil de distinguir porque queda "en el
medio" entre las otras dos categorías. Se priorizó balancear las 3 clases
para que el modelo no ignore "Alto riesgo" (antes del balanceo, esa clase
tenía muy pocos ejemplos y el F1 era de solo 0.32).

## Notebook de Google Colab

Existe además una versión autocontenida en Jupyter Notebook
(`riesgo_deportivo_colab.ipynb`, fuera de este repo) que reproduce el mismo
pipeline en celdas independientes, pensada para correr en Google Colab y
mostrar el entrenamiento de forma visual (gráfico de la pérdida actualizándose
en vivo, época por época).

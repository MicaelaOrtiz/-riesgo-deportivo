from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE / "models" / "risk_model.joblib"

app = FastAPI(
    title="Sports Risk Neural Network API",
    description="API educativa para estimar riesgo deportivo experimental.",
    version="1.0.0"
)
templates = Jinja2Templates(directory=str(BASE / "app" / "templates"))

bundle = None
if MODEL_PATH.exists():
    bundle = joblib.load(MODEL_PATH)

class AthleteData(BaseModel):
    age: int = Field(..., ge=14, le=80)
    weight_kg: float = Field(..., ge=35, le=180)
    training_hours: float = Field(..., ge=0, le=30)
    training_days: int = Field(..., ge=0, le=7)
    sleep_hours: float = Field(..., ge=0, le=14)
    rest_days: int = Field(..., ge=0, le=7)
    intensity: int = Field(..., ge=1, le=10)
    weekly_load: float = Field(..., ge=0, le=3000)
    pain: int = Field(..., ge=0, le=10)
    competition_minutes: float = Field(..., ge=0, le=500)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bundle is not None}

@app.post("/predict")
def predict(data: AthleteData):
    if bundle is None:
        return {"error": "Modelo no encontrado. Ejecutá primero: python train.py"}

    features = bundle["features"]
    row = pd.DataFrame([data.model_dump()])[features]
    model = bundle["model"]

    risk = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]
    classes = model.named_steps["neural_network"].classes_

    probs = {str(c): round(float(p), 4) for c, p in zip(classes, probabilities)}
    confidence = float(max(probabilities))

    messages = {
        "Bajo": "El modelo estima un nivel bajo para los datos ingresados.",
        "Medio": "El modelo estima un nivel intermedio; conviene prestar atención a recuperación y carga.",
        "Alto": "El modelo estima un nivel elevado; revisar carga y recuperación con un profesional."
    }

    return {
        "risk": risk,
        "probability": round(confidence, 4),
        "probabilities": probs,
        "message": messages.get(risk, "")
    }

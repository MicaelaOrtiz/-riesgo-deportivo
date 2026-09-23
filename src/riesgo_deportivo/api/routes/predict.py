from fastapi import APIRouter, Depends

from src.riesgo_deportivo.api.dependencies import get_prediction_service
from src.riesgo_deportivo.api.schemas import AthleteData, PredictionResponse
from src.riesgo_deportivo.api.services.prediction_service import (
    ModelNotLoadedError,
    PredictionService,
)

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(
    data: AthleteData,
    service: PredictionService = Depends(get_prediction_service),
):
    try:
        result = service.predict(data.model_dump())
    except ModelNotLoadedError as exc:
        return {"error": str(exc)}
    return result

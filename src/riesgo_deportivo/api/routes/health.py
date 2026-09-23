from fastapi import APIRouter, Depends

from src.riesgo_deportivo.api.dependencies import get_model_registry
from src.riesgo_deportivo.api.schemas import HealthResponse
from src.riesgo_deportivo.models.registry import ModelRegistry

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(registry: ModelRegistry = Depends(get_model_registry)) -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=registry.is_loaded)

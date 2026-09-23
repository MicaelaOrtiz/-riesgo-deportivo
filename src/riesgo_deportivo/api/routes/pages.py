from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.riesgo_deportivo.config import settings

router = APIRouter()
templates = Jinja2Templates(directory=str(settings.templates_dir))


@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {})

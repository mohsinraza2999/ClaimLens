from fastapi import APIRouter
from datetime import datetime
from src.utils.schemas import HealthResponse

health_router = APIRouter(tags=['Health'])

@health_router.get("/health", response_model=HealthResponse, status_code=200)
async def check():

    response=HealthResponse(status= "ok",
                            service= "ClaimLens Backend",
                            timestamp=datetime.utcnow().isoformat() + "Z")
    return response
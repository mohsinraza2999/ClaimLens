from fastapi import FastAPI
from .routes.health_route import health_router
from .routes.prediction_route import prediction_router


def merge_routes()-> FastAPI:
    app = FastAPI(description="An end to end video classifer as opinion or claim base on description",
               title="ClaimLens Project", version="0.1.0")
    app.include_router(router=health_router, tag=["Health"])
    app.include_router(router=prediction_router, tags=["Classifier"])

    return app

app=merge_routes()

@app.on_event("startup")
async def startup_event():
    print("🚀 ClaimLens API starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    print("🛑 ClaimLens API shutting down...")



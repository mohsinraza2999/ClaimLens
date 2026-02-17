from fastapi import APIRouter, HTTPException
import time
from datetime import datetime
from src.utils.schemas import PredictionResponse, APIResponse
from src.inference.core.prediction import Prediction
prediction_router= APIRouter(tags=['Classifier'])

@prediction_router.post("/classify", response_class=PredictionResponse,status_code=200)
async def predict(data:APIResponse)-> PredictionResponse:

    Prediction_object = Prediction()
    start_time = time.perf_counter()
    try:
        result=Prediction_object.predict_class()
        latency = (time.perf_counter() - start_time) * 1000
        response=PredictionResponse(timestamp=datetime.utcnow().isoformat(),
                                prediction= result,
                                latency_ms= round(latency, 3))
        return response
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        #logging.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail="Internal inference failure") 
    

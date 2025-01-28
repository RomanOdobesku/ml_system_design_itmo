import os
from datetime import datetime

import mlflow.pyfunc
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.logger import LOGGER

LOGGER.info("FastAPI serice start")

# Создаем приложение FastAPI
app = FastAPI(title="CatBoost Prediction Service")

load_dotenv(override=True)

LOGGER.info("Connecting to mlflow")
remote_server_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(remote_server_uri)

# Загружаем модель при старте приложения
MODEL_NAME = "tuned_catboost"
MODEL_VERSION = "latest"

try:
    LOGGER.info("Loading model")
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_VERSION}")
    LOGGER.info("Model loaded successfully")
except Exception as e:
    LOGGER.info(f"Error loading model: {e}")
    model = None  # pylint: disable=C0103


# Модель для входных данных (Pydantic)
class PredictionRequest(BaseModel):
    data: list[dict]


# Модель для ответа
class PredictionResponse(BaseModel):
    predictions: list[float]


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    LOGGER.info("Processing request")
    # Проверяем, что модель загружена
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    try:
        # Получаем входные данные в виде списка словарей
        input_data = request.data

        # Преобразуем типы данных для соответствия сигнатуре модели
        for record in input_data:
            for key, value in record.items():
                # Преобразуем целые числа в int32
                if key in ["is_main", "is_campaign_early"] and isinstance(value, int):
                    record[key] = bool(value)
                elif isinstance(value, int):
                    record[key] = np.int32(value)
                # Преобразуем числа с плавающей точкой в float32
                elif isinstance(value, float):
                    record[key] = np.float32(value)
                # Преобразуем строки с датами в datetime, если ключ предполагает дату
                elif key in ["event_date", "end_date"] and isinstance(value, str):
                    record[key] = datetime.fromisoformat(value)

        LOGGER.info("Calculating prediction")
        predictions = model.predict(input_data)
        LOGGER.info("Calculation complete")

        # Возвращаем результат
        return PredictionResponse(predictions=predictions.tolist())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

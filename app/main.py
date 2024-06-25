import logging
import os
import tempfile
from typing import List, Dict

import mlflow.pyfunc
import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from prometheus_client import Counter, Histogram, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def check_env_variables():
    required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "MLFLOW_S3_ENDPOINT_URL"]
    for var in required_vars:
        if os.getenv(var) is None:
            logger.error(f"Missing environment variable: {var}")
            exit(1)
        os.environ[var] = os.getenv(var)


check_env_variables()

app = FastAPI()

# Prometheus metrics
PREDICTION_TIME = Histogram('model_prediction_seconds', 'Time spent on model prediction')
FILE_PROCESSING_TIME = Histogram('file_processing_seconds', 'Time spent processing input file')
PREDICTION_ERRORS = Counter('prediction_errors_total', 'Total number of prediction errors')
FILE_PROCESSING_ERRORS = Counter('file_processing_errors_total', 'Total number of file processing errors')
PREDICTIONS_TOTAL = Counter('predictions_total', 'Total number of predictions made')
MODEL_LOADED = Gauge('model_loaded', 'Indicates if the model is loaded successfully')

Instrumentator().instrument(app).expose(app)


class ModelConfig:
    MODEL_NAME = "clf_catboost"
    CAT_FEATURES = ['HomePlanet', 'Destination', 'Deck', 'Side']


class Model:
    def __init__(self, model_name: str):
        try:
            self.model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")
            MODEL_LOADED.set(1)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            MODEL_LOADED.set(0)
            raise

    async def predict(self, data: pd.DataFrame) -> List[int]:
        try:
            with PREDICTION_TIME.time():
                predictions = self.model.predict(data)
            PREDICTIONS_TOTAL.inc(len(predictions))
            return predictions
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            PREDICTION_ERRORS.inc()
            raise


model = Model(ModelConfig.MODEL_NAME)


@app.get('/health')
def health():
    return {'status': 'healthy'}


@app.post("/predict")
async def create_upload_file(file: UploadFile = File(...)) -> List[Dict[str, str]]:
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV Files accepted.")

    try:
        with FILE_PROCESSING_TIME.time():
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                content = await file.read()
                temp_file.write(content)

            data = pd.read_csv(temp_file.name)
            os.unlink(temp_file.name)

            data[ModelConfig.CAT_FEATURES] = data[ModelConfig.CAT_FEATURES].apply(lambda x: x.astype('category'))

        ids = data['PassengerId']
        preds = await model.predict(data.drop(['PassengerId'], axis=1))

        res_df = pd.DataFrame({'PassengerId': ids, 'Transported': preds})
        return res_df.to_dict(orient="records")

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        FILE_PROCESSING_ERRORS.inc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

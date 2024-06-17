import os

import mlflow.pyfunc
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException

load_dotenv()

os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")

app = FastAPI()


class Model:
    def __init__(self, model_name):
        """
        To initialize the model
        model_name: Name of the model in registry
        model_stage: Stage of the model
        """
        # Load the model from Registry
        self.model = mlflow.pyfunc.load_model(f"models:/{model_name}/latest")

    def predict(self, data):
        """
        To use the loaded model to make predictions on the data
        data: Pandas DataFrame to perform predictions
        """
        predictions = self.model.predict(data)
        return predictions


model = Model("clf_catboost")


@app.get('/health')
def health():
    return 'health endpoint'


@app.post("/predict/")
async def predict(data: dict):
    prediction = model.predict(data)
    return {"prediction": prediction.tolist()}


@app.post("/invocations")
async def create_upload_file(file: UploadFile = File(...)):
    if file.filename.endswith(".csv"):
        with open(file.filename, "wb") as f:
            f.write(file.file.read())
        data = pd.read_csv(file.filename)

        cat_features = ['HomePlanet', 'Destination', 'Deck', 'Side']
        data[cat_features] = data[cat_features].apply(lambda x: x.astype('category'))

        os.remove(file.filename)
        ids = data['PassengerId']

        preds = model.predict(data.drop(['PassengerId'], axis=1))
        res_df = pd.concat([ids, pd.Series(preds)], axis=1)
        res_df.columns = ['PassengerId', 'Transported']

        return res_df.to_dict(orient="records")

    else:
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV Files accepted.")


if os.getenv("AWS_ACCESS_KEY_ID") is None or os.getenv("AWS_SECRET_ACCESS_KEY") is None:
    exit(1)

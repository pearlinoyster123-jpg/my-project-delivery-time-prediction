import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Delivery Time Prediction API", redirect_slashes=False)

model = None
preprocessor = None

def load_artifacts():
    global model, preprocessor
    if model is None or preprocessor is None:
        model_path = os.path.join(BASE_DIR, "model.pkl")
        prep_path = os.path.join(BASE_DIR, "preprocessor.pkl")

        if not os.path.exists(model_path) or not os.path.exists(prep_path):
            raise FileNotFoundError("Model or Preprocessor files missing.")

        model = joblib.load(model_path)
        preprocessor = joblib.load(prep_path)

class PredictionInput(BaseModel):
    Order_Hour: int
    Is_Weekend: int
    Weather: str
    Vehicle_Type: str
    Rider_Experience_Years: float
    Restaurant_Load: str
    Preparation_Time_Min: float
    Road_Distance_km: float
    Traffic_Level: str
    Average_Speed_kmph: float

@app.get("/")
def home():
    return {"message": "Delivery Time Prediction API is running"}

@app.post("/predict")
def predict(payload: PredictionInput):
    try:
        load_artifacts()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load error: {str(e)}")

    data = pd.DataFrame([payload.model_dump()])

    try:
        data_processed = preprocessor.transform(data)
        prediction = model.predict(data_processed)
        return {"predicted_delivery_time_min": round(float(prediction[0]), 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction processing error: {str(e)}")

def load_artifacts():
  global model, preprocessor
  # ... function contents ...

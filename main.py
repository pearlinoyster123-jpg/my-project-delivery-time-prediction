import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Delivery Time Prediction API")

# Dynamically target directory paths on Vercel
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor.pkl")

# Load saved models
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


class DeliveryInput(BaseModel):
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
def predict(input_data: DeliveryInput):
    try:
        data = pd.DataFrame([input_data.model_dump()])
        data_processed = preprocessor.transform(data)
        prediction = model.predict(data_processed)

        return {
            "predicted_delivery_time_min": round(float(prediction[0]), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

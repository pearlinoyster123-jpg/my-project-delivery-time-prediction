import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Delivery Time Prediction API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor.pkl")

# Define schema
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

# Helper function to get data dictionary compatible with Pydantic v1 and v2
def get_dict(data_obj):
    if hasattr(data_obj, "model_dump"):
        return data_obj.model_dump()
    return data_obj.dict()

@app.get("/")
def home():
    return {"message": "Delivery Time Prediction API is running"}

@app.post("/predict")
def predict(input_data: DeliveryInput):
    try:
        # Verify model files exist before loading
        if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
            raise FileNotFoundError("Model or Preprocessor .pkl file missing in deployment.")

        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)

        # Convert input payload to DataFrame
        data_dict = get_dict(input_data)
        data = pd.DataFrame([data_dict])

        # Transform and Predict
        data_processed = preprocessor.transform(data)
        prediction = model.predict(data_processed)

        return {
            "predicted_delivery_time_min": round(float(prediction[0]), 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

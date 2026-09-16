import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Delivery Time Prediction API")

# Global variables for model lazy-loading
model = None
preprocessor = None


def load_artifacts():
  global model, preprocessor
  if model is None or preprocessor is None:
    model_path = os.path.join(BASE_DIR, "model.pkl")
    prep_path = os.path.join(BASE_DIR, "preprocessor.pkl")

    if not os.path.exists(model_path) or not os.path.exists(prep_path):
      raise FileNotFoundError("Model or Preprocessor files missing from build.")

    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)


@app.get("/")
def home():
  return {"message": "Delivery Time Prediction API is running"}


@app.post("/predict")
def predict(
    Order_Hour: int,
    Is_Weekend: int,
    Weather: str,
    Vehicle_Type: str,
    Rider_Experience_Years: float,
    Restaurant_Load: str,
    Preparation_Time_Min: float,
    Road_Distance_km: float,
    Traffic_Level: str,
    Average_Speed_kmph: float,
):
  try:
    load_artifacts()
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Model load error: {str(e)}")

  data = pd.DataFrame([{
      "Order_Hour": Order_Hour,
      "Is_Weekend": Is_Weekend,
      "Weather": Weather,
      "Vehicle_Type": Vehicle_Type,
      "Rider_Experience_Years": Rider_Experience_Years,
      "Restaurant_Load": Restaurant_Load,
      "Preparation_Time_Min": Preparation_Time_Min,
      "Road_Distance_km": Road_Distance_km,
      "Traffic_Level": Traffic_Level,
      "Average_Speed_kmph": Average_Speed_kmph,
  }])

  data_processed = preprocessor.transform(data)
  prediction = model.predict(data_processed)

  return {"predicted_delivery_time_min": round(float(prediction[0]), 2)}

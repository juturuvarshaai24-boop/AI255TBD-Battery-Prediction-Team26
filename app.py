
import joblib
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel, Field

# Create the API
app = FastAPI(
    title="Smart Doorbell Battery Prediction API",
    description="Predicts remaining battery life using sensor readings.",
    version="1.0"
)

# Load trained model
model = joblib.load("battery_model.pkl")


# Define expected input data and validation limits
class BatteryInput(BaseModel):
    battery_v: float = Field(ge=3.35, le=4.20)
    temp_c: float = Field(ge=8, le=46)
    rings_per_day: int = Field(ge=0, le=37)
    wifi_rssi_dbm: int = Field(ge=-88, le=-45)
    age_months: int = Field(ge=0, le=36)


# API homepage
@app.get("/")
def home():
    return {
        "message": "Smart Doorbell Battery Prediction API is running!"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: BatteryInput):

    input_data = [[
        data.battery_v,
        data.temp_c,
        data.rings_per_day,
        data.wifi_rssi_dbm,
        data.age_months
    ]]

    prediction = model.predict(input_data)

    days = max(float(prediction[0]), 0)

    return {
        "estimated_battery_life_days": round(days, 2),
        "unit": "days"
    }


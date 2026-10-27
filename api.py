
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Load the trained model
MODEL_PATH = Path(__file__).parent / "random_forest_model.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="House Price Prediction API")

FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


class HouseFeatures(BaseModel):
    MedInc: float = Field(gt=0, description="Median income in tens of thousands of dollars")
    HouseAge: float = Field(ge=0)
    AveRooms: float = Field(gt=0)
    AveBedrms: float = Field(gt=0)
    Population: float = Field(gt=0)
    AveOccup: float = Field(gt=0)
    Latitude: float = Field(ge=32, le=42)
    Longitude: float = Field(ge=-125, le=-114)


@app.get("/")
def home():
    return {"message": "House Price Prediction API is running!"}


@app.post("/predict")
def predict_price(house: HouseFeatures):
    input_data = pd.DataFrame([house.model_dump()], columns=FEATURES)

    # Model output is in units of $100,000
    prediction = float(model.predict(input_data)[0])
    predicted_price = prediction * 100_000

    return {
        "predicted_price_units": round(prediction, 4),
        "predicted_price_usd": round(predicted_price, 2),
        "currency": "USD",
    }


import joblib
import pandas as pd
from pathlib import Path

# Load the saved model
model_path = Path(__file__).parent / "random_forest_model.pkl"
model = joblib.load(model_path)

print("===== HOUSE PRICE PREDICTION =====")
print("Enter the house details below:\n")

# Collect user inputs
house_data = {
    "MedInc": float(input("Median income (in tens of thousands of dollars): ")),
    "HouseAge": float(input("House age (years): ")),
    "AveRooms": float(input("Average rooms per household: ")),
    "AveBedrms": float(input("Average bedrooms per household: ")),
    "Population": float(input("Block population: ")),
    "AveOccup": float(input("Average occupants per household: ")),
    "Latitude": float(input("Latitude: ")),
    "Longitude": float(input("Longitude: ")),
}

# Keep features in the same order as training
house = pd.DataFrame([house_data])

# Predict
predicted_value = model.predict(house)[0]
predicted_dollars = predicted_value * 100000

print("\n===== PREDICTION RESULT =====")
print(f"Predicted target value: {predicted_value:.4f}")
print(f"Approximate predicted house value: ${predicted_dollars:,.0f}")


from sklearn.datasets import fetch_california_housing
import pandas as pd


def load_data():
    """Load the California Housing dataset and return a DataFrame."""
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    return df


if __name__ == "__main__":
    df = load_data()

    print("Dataset loaded successfully!")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst five rows:")
    print(df.head())

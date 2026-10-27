
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from data_loader import load_data
from evaluate import evaluate_model


def train_and_compare_models(df):
    # Separate features and target
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    # Use the same split for every model
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluate
        metrics = evaluate_model(y_test, y_pred)

        results.append({
            "Model": name,
            **metrics,
        })

        trained_models[name] = model

        print(f"{name} completed.")

    # Display comparison
    results_df = pd.DataFrame(results)

    print("\n===== MODEL COMPARISON =====")
    print(results_df.round(4).to_string(index=False))

    return trained_models, results_df


if __name__ == "__main__":
    df = load_data()
    train_and_compare_models(df)

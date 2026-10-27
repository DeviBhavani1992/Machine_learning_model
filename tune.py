import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

from data_loader import load_data
from evaluate import evaluate_model


def tune_random_forest(df):
    # 1. Separate features (X) and target (y)
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    # 2. Split data: keep test data separate from tuning
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # 3. Define the hyperparameters to explore
    param_distributions = {
        "n_estimators": [100, 150],
        "max_depth": [None, 15, 25],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [1.0, "sqrt"],
    }

    # 4. Search for a better configuration using cross-validation
    base_model = RandomForestRegressor(
        random_state=42,
        n_jobs=1,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=6,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    print("\n===== STARTING HYPERPARAMETER TUNING =====")
    search.fit(X_train, y_train)

    # 5. Display the best configuration
    print("\n===== BEST HYPERPARAMETERS =====")
    print(search.best_params_)

    print(f"\nBest Cross-Validation RMSE: {-search.best_score_:.4f}")
    print(
        f"Approximate CV error in dollars: "
        f"${-search.best_score_ * 100000:,.0f}"
    )

    # 6. Evaluate the selected model on untouched test data
    best_model = search.best_estimator_
    model_path = Path("random_forest_model.pkl")
    joblib.dump(best_model, model_path)

    print(f"Model saved at: {model_path.resolve()}")
    y_pred = best_model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)

    print("\n===== FINAL TEST RESULTS =====")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print(
        f"\nApproximate test MAE in dollars: "
        f"${metrics['MAE'] * 100000:,.0f}"
    )

    return best_model, search, metrics


if __name__ == "__main__":
    df = load_data()
    best_model, search, metrics = tune_random_forest(df)

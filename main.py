
from data_loader import load_data
from train import train_model


def main():
    print("===== HOUSE PRICE PREDICTION =====")

    # Step 1: Load dataset
    df = load_data()
    print(f"Dataset loaded: {df.shape}")

    # Step 2: Train and evaluate model
    model, metrics = train_model(df)

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()

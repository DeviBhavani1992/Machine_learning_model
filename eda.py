
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_data


def plot_target_distribution(df):
    """Visualize the distribution of house values."""
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="MedHouseVal", bins=40, kde=True)
    plt.title("Distribution of Median House Values")
    plt.xlabel("Median House Value ($100,000 units)")
    plt.ylabel("Number of Districts")
    plt.tight_layout()
    plt.show()


def plot_income_vs_house_value(df):
    """Visualize the relationship between income and house value."""
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="MedInc",
        y="MedHouseVal",
        alpha=0.3
    )
    plt.title("Median Income vs Median House Value")
    plt.xlabel("Median Income (in $10,000 units)")
    plt.ylabel("Median House Value ($100,000 units)")
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df):
    """Display correlations between numerical features."""
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def run_eda(df):
    """Run all exploratory data analysis visualizations."""
    plot_target_distribution(df)
    plot_income_vs_house_value(df)
    plot_correlation_heatmap(df)


if __name__ == "__main__":
    df = load_data()
    run_eda(df)

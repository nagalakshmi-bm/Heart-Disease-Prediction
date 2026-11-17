import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load model comparison results
df = pd.read_csv("report/model_comparison.csv")

# Create visualization folder
os.makedirs("visualizations", exist_ok=True)

sns.set_theme(style="whitegrid")

# Metrics to visualize
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

for metric in metrics:

    plt.figure(figsize=(9, 5))

    sns.barplot(
        data=df,
        x="Model",
        y=metric
    )

    plt.title(f"{metric} Comparison of Machine Learning Models")
    plt.xlabel("Machine Learning Models")
    plt.ylabel(metric)

    plt.ylim(0, 1)
    plt.xticks(rotation=15)

    plt.tight_layout()

    filename = metric.lower().replace(" ", "_") + "_comparison.png"

    plt.savefig(f"visualizations/{filename}")
    plt.close()

print("ALL MODEL PERFORMANCE GRAPHS CREATED SUCCESSFULLY!")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_heart.csv")

# Load trained Random Forest model
model = joblib.load("models/random_forest.pkl")

# Extract feature importance
feature_importance = model.named_steps["model"].feature_importances_

# Create feature importance dataframe
importance_df = pd.DataFrame({
    "Feature": df.drop("target", axis=1).columns,
    "Importance": feature_importance
})

# Sort features
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Display results
print("FEATURE IMPORTANCE ANALYSIS")
print(importance_df)

# Save results
os.makedirs("report", exist_ok=True)

importance_df.to_csv(
    "report/feature_importance.csv",
    index=False
)

# Create visualization
plt.figure(figsize=(10, 6))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance - Random Forest Model")
plt.xlabel("Importance Score")
plt.ylabel("Health Features")

plt.tight_layout()

plt.savefig(
    "visualizations/feature_importance.png"
)

plt.close()

print("\nFeature importance analysis completed!")
print("Results saved successfully!")
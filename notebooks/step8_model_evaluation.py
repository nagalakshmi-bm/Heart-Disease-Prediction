import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_heart.csv")

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load trained models
models = {
    "Logistic Regression": joblib.load(
        "models/logistic_regression.pkl"
    ),
    "Decision Tree": joblib.load(
        "models/decision_tree.pkl"
    ),
    "Random Forest": joblib.load(
        "models/random_forest.pkl"
    )
}

results = []

print("MODEL EVALUATION RESULTS\n")

for name, model in models.items():

    # Predict test data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("=" * 45)
    print(name)

    print("\nAccuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

# Create comparison table
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df)

# Save comparison results
results_df.to_csv(
    "report/model_comparison.csv",
    index=False
)

print("\nModel comparison saved successfully!")
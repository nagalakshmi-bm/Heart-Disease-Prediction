import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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

# Define machine learning models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

# Create models folder
os.makedirs("models", exist_ok=True)

print("MACHINE LEARNING MODEL TRAINING\n")

# Train and save models
for name, classifier in models.items():

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", classifier)
    ])

    pipeline.fit(X_train, y_train)

    # Save trained model
    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(pipeline, "models/" + filename)

    print(name, "trained successfully!")
    print("Model saved:", filename)
    print("-" * 40)

print("\nALL MODELS TRAINED SUCCESSFULLY!")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_heart.csv")

# Separate input features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale numerical features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display dataset information
print("DATA PREPROCESSING COMPLETED")

print("\nOriginal Dataset Shape:", df.shape)

print("\nTraining Data Shape:", X_train_scaled.shape)

print("\nTesting Data Shape:", X_test_scaled.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())

print("\nFEATURE SCALING COMPLETED SUCCESSFULLY!")
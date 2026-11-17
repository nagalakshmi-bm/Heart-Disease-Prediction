import pandas as pd

# Load the dataset
df = pd.read_csv("dataset/heart.csv")

# Display first 5 rows
print("FIRST 5 ROWS OF HEART DISEASE DATASET")
print(df.head())

# Display dataset shape
print("\nDATASET SHAPE")
print(df.shape)

# Display column names
print("\nDATASET COLUMNS")
print(df.columns.tolist())
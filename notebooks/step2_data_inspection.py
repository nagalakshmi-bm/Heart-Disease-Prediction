import pandas as pd

# Load dataset
df = pd.read_csv("dataset/heart.csv")

# Dataset information
print("DATASET INFORMATION")
print(df.info())

# Statistical summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())

# Missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Duplicate records
print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())

# Target distribution
print("\nHEART DISEASE TARGET DISTRIBUTION")
print(df["target"].value_counts())

print("\nDATA INSPECTION COMPLETED")
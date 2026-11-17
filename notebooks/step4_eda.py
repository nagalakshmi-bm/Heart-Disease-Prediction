import pandas as pd

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_heart.csv")

# Basic information
print("EXPLORATORY DATA ANALYSIS")

print("\n1. DATASET SHAPE")
print(df.shape)

# Average age
print("\n2. AVERAGE AGE")
print(df["age"].mean())

# Minimum and maximum age
print("\n3. AGE RANGE")
print("Minimum Age:", df["age"].min())
print("Maximum Age:", df["age"].max())

# Average cholesterol
print("\n4. AVERAGE CHOLESTEROL")
print(df["chol"].mean())

# Average maximum heart rate
print("\n5. AVERAGE MAXIMUM HEART RATE")
print(df["thalach"].mean())

# Heart disease distribution
print("\n6. HEART DISEASE DISTRIBUTION")
print(df["target"].value_counts())

# Statistical summary
print("\n7. STATISTICAL SUMMARY")
print(df.describe())

print("\nEDA COMPLETED SUCCESSFULLY!")
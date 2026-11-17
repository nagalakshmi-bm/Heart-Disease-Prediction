import pandas as pd

# Load dataset
df = pd.read_csv("dataset/heart.csv")

print("BEFORE DATA CLEANING")
print("Dataset Shape:", df.shape)

# Check missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Check duplicates
print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())

# Remove duplicate records
df = df.drop_duplicates()

# Check data types
print("\nDATA TYPES")
print(df.dtypes)

# Final dataset information
print("\nAFTER DATA CLEANING")
print("Dataset Shape:", df.shape)

# Save cleaned dataset
df.to_csv("dataset/cleaned_heart.csv", index=False)

print("\nCleaned dataset saved successfully!")
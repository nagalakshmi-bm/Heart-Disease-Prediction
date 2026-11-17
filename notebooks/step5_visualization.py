import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_heart.csv")

# Create visualization folder if not exists
os.makedirs("visualizations", exist_ok=True)

sns.set_theme(style="whitegrid")

# 1. Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["age"], bins=20, kde=True)
plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig("visualizations/age_distribution.png")
plt.close()

# 2. Gender Distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="sex")
plt.title("Gender Distribution")
plt.xlabel("Gender (0 = Female, 1 = Male)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("visualizations/gender_distribution.png")
plt.close()

# 3. Heart Disease Distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="target")
plt.title("Heart Disease Target Distribution")
plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("visualizations/heart_disease_distribution.png")
plt.close()

# 4. Cholesterol Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["chol"], bins=20, kde=True)
plt.title("Cholesterol Distribution")
plt.xlabel("Cholesterol")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("visualizations/cholesterol_distribution.png")
plt.close()

# 5. Maximum Heart Rate Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["thalach"], bins=20, kde=True)
plt.title("Maximum Heart Rate Distribution")
plt.xlabel("Maximum Heart Rate")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("visualizations/maximum_heart_rate_distribution.png")
plt.close()

print("ALL 5 VISUALIZATIONS CREATED AND SAVED SUCCESSFULLY!")
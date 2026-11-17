import pandas as pd
import os

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# Load results
df = pd.read_csv("dataset/cleaned_heart.csv")
results = pd.read_csv("report/model_comparison.csv")
importance = pd.read_csv("report/feature_importance.csv")

# Create report folder
os.makedirs("report", exist_ok=True)

# PDF document
pdf = SimpleDocTemplate(
    "report/Heart_Disease_Project_Report.pdf",
    pagesize=A4
)

styles = getSampleStyleSheet()
content = []

# Title
content.append(Paragraph(
    "Heart Disease Prediction and Healthcare Data Analysis",
    styles["Title"]
))
content.append(Spacer(1, 20))

# Introduction
content.append(Paragraph("1. Introduction", styles["Heading1"]))
content.append(Paragraph(
    "This project analyzes a healthcare dataset using Python "
    "and Machine Learning techniques. The objective is to explore "
    "health-related attributes and develop educational classification "
    "models using the available dataset.",
    styles["BodyText"]
))
content.append(Spacer(1, 12))

# Dataset
content.append(Paragraph("2. Dataset Overview", styles["Heading1"]))
content.append(Paragraph(
    f"Total Records: {df.shape[0]}<br/>"
    f"Total Columns: {df.shape[1]}<br/>"
    f"Target Column: target",
    styles["BodyText"]
))
content.append(Spacer(1, 12))

# Methodology
content.append(Paragraph("3. Methodology", styles["Heading1"]))
content.append(Paragraph(
    "The project followed these stages:<br/>"
    "1. Dataset Collection<br/>"
    "2. Data Inspection<br/>"
    "3. Data Cleaning<br/>"
    "4. Exploratory Data Analysis<br/>"
    "5. Data Visualization<br/>"
    "6. Data Preprocessing<br/>"
    "7. Machine Learning Model Training<br/>"
    "8. Model Evaluation<br/>"
    "9. Feature Importance Analysis",
    styles["BodyText"]
))
content.append(Spacer(1, 12))

# Model comparison
content.append(Paragraph(
    "4. Machine Learning Model Comparison",
    styles["Heading1"]
))

table_data = [
    ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
]

for _, row in results.iterrows():
    table_data.append([
        row["Model"],
        f'{row["Accuracy"]:.3f}',
        f'{row["Precision"]:.3f}',
        f'{row["Recall"]:.3f}',
        f'{row["F1 Score"]:.3f}'
    ])

table = Table(table_data, repeatRows=1)

table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("PADDING", (0, 0), (-1, -1), 7)
]))

content.append(table)
content.append(Spacer(1, 15))

# Feature importance
content.append(Paragraph(
    "5. Feature Importance Analysis",
    styles["Heading1"]
))

top_features = importance.head(5)

for _, row in top_features.iterrows():
    content.append(Paragraph(
        f'{row["Feature"]}: {row["Importance"]:.4f}',
        styles["BodyText"]
    ))

content.append(Spacer(1, 12))

# Conclusion
content.append(Paragraph("6. Conclusion", styles["Heading1"]))
content.append(Paragraph(
    "This project demonstrates the application of data science "
    "and machine learning techniques to a healthcare dataset. "
    "Different classification models were trained and evaluated. "
    "Their performance was compared using standard evaluation metrics. "
    "The results are intended for educational purposes only and "
    "must not be used for medical diagnosis.",
    styles["BodyText"]
))

# Build PDF
pdf.build(content)

print("FINAL PROJECT REPORT GENERATED SUCCESSFULLY!")
print("Saved: report/Heart_Disease_Project_Report.pdf")
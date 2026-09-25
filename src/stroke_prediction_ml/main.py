# Stroke Prediction ML (Sci-kit Learn Logistic Regression Model)
import pandas as pd
import matplotlib.pyplot as plt

# Import Data
df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
df = df.drop(columns=['id'])
print(df.head())

# Data Visualization
plt.scatter(df["age"], df["stroke"])

plt.xlabel("Age")
plt.yticks([0, 1], ["No Stroke", "Stroke"])
plt.title("Age Distribution by Stroke Status")

plt.show()

# Data Preprocessing

# Model Training

# Model Evaluation
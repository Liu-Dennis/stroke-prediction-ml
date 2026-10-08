import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class StrokePredictionNet(nn.Module):
    def __init__(self, dropout=0.4):
        super(StrokePredictionNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

def main():
    # 1. Train model directly in memory
    model = StrokePredictionNet()
    model.load_state_dict(torch.load('predictionNN.pth', weights_only=True))
    model.eval()

    # 2. Collect user inputs
    gender = input("Gender (Male/Female/Other): ")
    age = float(input("Age: "))
    hypertension = int(input("Hypertension (0/1): "))
    heart_disease = int(input("Heart Disease (0/1): "))
    ever_married = input("Ever Married (Yes/No): ")
    work_type = input("Work Type (Private/Self-employed/Govt_job/children/Never_worked): ")
    residence_type = input("Residence Type (urban/rural): ")
    rtn_avg_glucose_level = float(input("Rtn Avg Glucose Level: "))
    bmi = float(input("BMI: "))
    smoking_status = input("Smoking Status (formerly_smoked/never_smoked/smokes/unknown): ")

    # 3. Format inputs to match model features
    user_dict = {
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'avg_glucose_level': rtn_avg_glucose_level,
        'bmi': bmi,
        'gender_' + gender.lower(): 1.0,
        'ever_married_' + ever_married.lower(): 1.0 if ever_married.lower() == 'yes' else 0.0,
        'work_type_' + work_type.lower(): 1.0,
        'residence_type_' + residence_type.lower(): 1.0,
        'smoking_status_' + smoking_status.lower(): 1.0
    }

    feature_template = {
        'age': 0.0, 
        'hypertension': 0.0, 
        'heart_disease': 0.0,
        'avg_glucose_level': 0.0, 
        'bmi': 0.0,
        'gender_male': 0.0, 
        'gender_other': 0.0,
        'ever_married_yes': 0.0,
        'work_type_never_worked': 0.0,
        'work_type_private': 0.0,
        'work_type_self_employed': 0.0,
        'work_type_children': 0.0,
        'residence_type_urban': 0.0,
        'smoking_status_formerly_smoked': 0.0,
        'smoking_status_never_smoked': 0.0,
        'smoking_status_smokes': 0.0
    }

    for k, v in user_dict.items():
        if k in feature_template:
            feature_template[k] = float(v)

    # 4. Predict
    input_values = list(feature_template.values())
    input_tensor = torch.tensor([input_values], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.sigmoid(output).item()
        prediction = 1 if probability >= 0.5 else 0

    print(f"\nPredicted Class: {prediction} (Probability: {probability:.4f})")
    return prediction

if __name__ == "__main__":
    result = main()
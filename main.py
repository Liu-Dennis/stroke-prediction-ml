import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PredictionNet(nn.Module):
    def __init__(self):
        super(PredictionNet, self).__init__()
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(1)

def train_model():
    print("Loading dataset and training model...")
    df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    X_dataframe = df.iloc[:, :-1].copy()
    Y_dataframe = df.iloc[:, -1].copy()

    X_dataframe = pd.get_dummies(X_dataframe, drop_first=True)
    X_dataframe = X_dataframe.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    Y_dataframe = pd.to_numeric(Y_dataframe, errors='coerce').fillna(0.0).astype(np.int64)

    X_tensor = torch.from_numpy(X_dataframe.to_numpy(dtype=np.float32))
    Y_tensor = torch.from_numpy(np.asarray(Y_dataframe, dtype=np.float32))

    model = PredictionNet()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)

    model.train()
    epochs = 10
    batch_size = 64

    for epoch in range(epochs):
        for i in range(0, len(X_tensor), batch_size):
            batch_x = X_tensor[i:i + batch_size]
            batch_y = Y_tensor[i:i + batch_size]

            optimizer.zero_grad()
            output = model(batch_x)
            loss = F.binary_cross_entropy_with_logits(output, batch_y)
            loss.backward()
            optimizer.step()

    print("Training complete!\n")
    return model

def main():
    # 1. Train model directly in memory
    model = train_model()

    # 2. Collect user inputs
    gender = input("Gender (Male/Female/Other): ")
    age = float(input("Age: "))
    hypertension = int(input("Hypertension (0/1): "))
    heart_disease = int(input("Heart Disease (0/1): "))
    ever_married = input("Ever Married (Yes/No): ")
    work_type = input("Work Type (Private/Self-employed/Govt_job/children/Never_worked): ")
    Residence_type = input("Residence Type (Urban/Rural): ")
    rtn_avg_glucose_level = float(input("Rtn Avg Glucose Level: "))
    bmi = float(input("BMI: "))
    smoking_status = input("Smoking Status (formerly smoked/never smoked/smokes/Unknown): ")

    # 3. Format inputs to match model features
    user_dict = {
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'avg_glucose_level': rtn_avg_glucose_level,
        'bmi': bmi,
        'gender_' + gender: 1.0,
        'ever_married_Yes': 1.0 if ever_married.lower() == 'yes' else 0.0,
        'work_type_' + work_type: 1.0,
        'Residence_type_' + Residence_type: 1.0,
        'smoking_status_' + smoking_status: 1.0
    }

    feature_template = {
        'age': 0.0, 'hypertension': 0.0, 'heart_disease': 0.0,
        'avg_glucose_level': 0.0, 'bmi': 0.0,
        'gender_Male': 0.0, 'gender_Other': 0.0,
        'ever_married_Yes': 0.0,
        'work_type_Never_worked': 0.0, 'work_type_Private': 0.0,
        'work_type_Self-employed': 0.0, 'work_type_children': 0.0,
        'Residence_type_Urban': 0.0,
        'smoking_status_formerly smoked': 0.0,
        'smoking_status_never smoked': 0.0,
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
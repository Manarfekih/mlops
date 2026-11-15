
# ===== 1. Import the libraries =====

import os
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# ===== 2. Create the application =====

app = FastAPI(title="Churn API")


# ===== 3. Read the configuration from environment variables =====

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_URI = os.getenv("MODEL_URI", "models:/churn-model@champion")


# ===== 4. Load the champion ONLY ONCE, at startup =====

mlflow.set_tracking_uri(TRACKING_URI)

model = mlflow.sklearn.load_model(MODEL_URI)

print(f"Model loaded from: {MODEL_URI}")


# ===== 5. Describe the expected data =====

class Customer(BaseModel):
    age: int
    salaire: float
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float
    consommation: float


# ===== 6. Route /health: check that the API is alive =====

@app.get("/health")
def health():
    return {"status": "ok"}


# ===== 7. Route /predict: receive a customer, return the prediction =====

@app.post("/predict")
def predict(customer: Customer):

    # a) Turn the received customer into a one-row table
    row = pd.DataFrame([customer.model_dump()])

    # b) Turn the text "abonnement" into 0/1 columns
    row = pd.get_dummies(row)

    # c) Keep the model's columns, in the right order
    row = row.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # d) Predict: 1 = leaves, 0 = stays
    result = int(model.predict(row)[0])

    return {
        "churn": result,
        "message": "The customer will leave"
        if result == 1
        else "The customer stays"
    }

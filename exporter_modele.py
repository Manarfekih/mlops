
# ===== 1. Import the libraries =====

import mlflow
import mlflow.sklearn


# ===== 2. Connect to the Registry database (the one of Lab 4) =====

mlflow.set_tracking_uri("sqlite:///mlflow.db")


# ===== 3. Load the champion from the Registry =====

model = mlflow.sklearn.load_model(
    "models:/churn-model@champion"
)


# ===== 4. Write it to an ordinary folder, "model" =====

# serialization_format="cloudpickle": same format as in Labs 3 and 4
# This folder will be copied into the Docker image.

mlflow.sklearn.save_model(
    model,
    "model",
    serialization_format="cloudpickle"
)

print("Champion exported to the model/ folder")

# src/predict.py
"""
Reusable prediction module using MLflow
"""

import mlflow
import pandas as pd

def predict_model(model_name: str, input_data: pd.DataFrame):
    """
    Load the latest version of a model from MLflow and make predictions.

    Parameters:
    - model_name (str): Registered MLflow model name
    - input_data (pd.DataFrame): Feature matrix (X)

    Returns:
    - predictions (np.array)
    """
    # Load latest production model
    model_uri = f"models:/{model_name}/latest"
    model = mlflow.pyfunc.load_model(model_uri)

    # Predict
    predictions = model.predict(input_data)

    return predictions


# -----------------------------
# 2. Run standalone
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict using MLflow registered model")
    parser.add_argument("--model_name", type=str, required=True, help="Registered MLflow model name")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--save_path", type=str, default=None, help="Optional path to save predictions CSV")
    args = parser.parse_args()

    predictions = predict_model(args.model_name, args.data_path, args.save_path)
    print("🎉 Prediction complete!")
    print(predictions.head())
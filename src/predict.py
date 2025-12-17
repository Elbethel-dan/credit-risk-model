import pandas as pd
import mlflow
import mlflow.pyfunc

MODEL_NAME = "CreditRiskModel"
MODEL_STAGE = "Production"   # or "Staging" / "None"
INPUT_DATA = "data/processed.csv"
OUTPUT_DATA = "data/predictions.csv"

ID_COL = "CustomerId"

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
df = pd.read_csv(INPUT_DATA)

assert ID_COL in df.columns, "CustomerId column is missing!"

ids = df[ID_COL]
X = df.drop(columns=[ID_COL])

# --------------------------------------------------
# 2. Load model from MLflow Model Registry
# --------------------------------------------------
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
model = mlflow.pyfunc.load_model(model_uri)

print(f"✅ Loaded model: {MODEL_NAME} [{MODEL_STAGE}]")

# --------------------------------------------------
# 3. Generate predictions
# --------------------------------------------------
# Predict probabilities
y_proba = model.predict(X)

# Handle models returning 2D probabilities
if y_proba.ndim > 1:
    y_proba = y_proba[:, 1]

# Binary prediction (default threshold = 0.5)
y_pred = (y_proba >= 0.5).astype(int)

# --------------------------------------------------
# 4. Save predictions
# --------------------------------------------------
predictions_df = pd.DataFrame({
    ID_COL: ids,
    "risk_probability": y_proba,
    "risk_prediction": y_pred
})

predictions_df.to_csv(OUTPUT_DATA, index=False)

print("📁 Predictions saved to:", OUTPUT_DATA)
print(predictions_df.head())

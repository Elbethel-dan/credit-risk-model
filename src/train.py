import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import mlflow
import mlflow.sklearn

RANDOM_STATE = 42
TARGET_COL = "target"
ID_COL = "CustomerId"

# --------------------------------------------------
# 1. Load Data
# --------------------------------------------------
df = pd.read_csv("data/final_dataset.csv")

# Separate features and target
X = df.drop(columns=[TARGET_COL, ID_COL])
y = df[TARGET_COL]

# --------------------------------------------------
# 2. Train-Test Split (Reproducibility)
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# --------------------------------------------------
# 3. Evaluation Function
# --------------------------------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

# --------------------------------------------------
# 4. MLflow Setup
# --------------------------------------------------
mlflow.set_experiment("Credit_Risk_Modeling")

# --------------------------------------------------
# 5. Logistic Regression + Grid Search
# --------------------------------------------------
with mlflow.start_run(run_name="LogisticRegression_GridSearch"):

    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    lr_param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs"]
    }

    grid_lr = GridSearchCV(
        lr,
        param_grid=lr_param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1
    )

    grid_lr.fit(X_train, y_train)
    best_lr = grid_lr.best_estimator_

    metrics = evaluate_model(best_lr, X_test, y_test)

    # Log parameters & metrics
    mlflow.log_params(grid_lr.best_params_)
    mlflow.log_metrics(metrics)

    # Log model
    mlflow.sklearn.log_model(
        best_lr,
        artifact_path="model",
        registered_model_name="CreditRiskModel"
    )

    print("✅ Logistic Regression logged to MLflow")

# --------------------------------------------------
# 6. Random Forest + Random Search
# --------------------------------------------------
with mlflow.start_run(run_name="RandomForest_RandomSearch"):

    rf = RandomForestClassifier(random_state=RANDOM_STATE)

    rf_param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    random_rf = RandomizedSearchCV(
        rf,
        param_distributions=rf_param_dist,
        n_iter=20,
        scoring="roc_auc",
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    random_rf.fit(X_train, y_train)
    best_rf = random_rf.best_estimator_

    metrics = evaluate_model(best_rf, X_test, y_test)

    # Log parameters & metrics
    mlflow.log_params(random_rf.best_params_)
    mlflow.log_metrics(metrics)

    # Log model
    mlflow.sklearn.log_model(
        best_rf,
        artifact_path="model",
        registered_model_name="CreditRiskModel"
    )

    print("✅ Random Forest logged to MLflow")

# --------------------------------------------------
# 7. Final Message
# --------------------------------------------------
print("🎉 Training complete. Compare runs in MLflow UI.")
print("Run: mlflow ui")

# train_model.py
"""
Reusable module to train multiple ML models with MLflow tracking
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import re


# Optional: if using XGBoost or LightGBM
from xgboost import XGBClassifier

RANDOM_STATE = 42

#-----------------------------
# Removing special characters from feature names
#-----------------------------
def clean_feature_names(X: pd.DataFrame) -> pd.DataFrame:
    """
    Make feature names safe for XGBoost by removing special characters.
    """
    X = X.copy()
    X.columns = [
        re.sub(r"[^A-Za-z0-9_]+", "_", col)
        for col in X.columns
    ]
    return X

# --------------------------
# Evaluation function
# --------------------------
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

# --------------------------
# Train function
# --------------------------
def train_model(data_path, target_col, id_col):
    """
    Trains multiple ML models and logs to MLflow.

    Parameters:
    - data_path (str): Path to CSV dataset
    - target_col (str): Name of target column
    - id_col (str): Name of ID column

    Returns:
    - metrics_dict: Dictionary with evaluation metrics for each model
    """
    # Load data
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_col, id_col])
    y = df[target_col]

# 🔹 Clean feature names for XGBoost compatibility
    X = clean_feature_names(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    metrics_dict = {}
    mlflow.set_experiment("Credit_Risk_Modeling")

    # --------------------------
    # Logistic Regression
    # --------------------------
    with mlflow.start_run(run_name="LogisticRegression_GridSearch"):
        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        lr_param_grid = {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs"]}
        grid_lr = GridSearchCV(lr, param_grid=lr_param_grid, scoring="roc_auc", cv=5, n_jobs=-1)
        grid_lr.fit(X_train, y_train)
        best_lr = grid_lr.best_estimator_
        metrics_dict["LogisticRegression"] = evaluate_model(best_lr, X_test, y_test)
        mlflow.log_params(grid_lr.best_params_)
        mlflow.log_metrics(metrics_dict["LogisticRegression"])
        mlflow.sklearn.log_model(best_lr, artifact_path="model", registered_model_name="CreditRiskModel")

    # --------------------------
    # Decision Tree
    # --------------------------
    with mlflow.start_run(run_name="DecisionTree_GridSearch"):
        dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
        dt_param_grid = {"max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]}
        grid_dt = GridSearchCV(dt, param_grid=dt_param_grid, scoring="roc_auc", cv=5, n_jobs=-1)
        grid_dt.fit(X_train, y_train)
        best_dt = grid_dt.best_estimator_
        metrics_dict["DecisionTree"] = evaluate_model(best_dt, X_test, y_test)
        mlflow.log_params(grid_dt.best_params_)
        mlflow.log_metrics(metrics_dict["DecisionTree"])
        mlflow.sklearn.log_model(best_dt, artifact_path="model", registered_model_name="CreditRiskModel")

    # --------------------------
    # Random Forest
    # --------------------------
    with mlflow.start_run(run_name="RandomForest_RandomSearch"):
        rf = RandomForestClassifier(random_state=RANDOM_STATE)
        rf_param_dist = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
        random_rf = RandomizedSearchCV(rf, param_distributions=rf_param_dist, n_iter=20,
                                       scoring="roc_auc", cv=5, random_state=RANDOM_STATE, n_jobs=-1)
        random_rf.fit(X_train, y_train)
        best_rf = random_rf.best_estimator_
        metrics_dict["RandomForest"] = evaluate_model(best_rf, X_test, y_test)
        mlflow.log_params(random_rf.best_params_)
        mlflow.log_metrics(metrics_dict["RandomForest"])
        mlflow.sklearn.log_model(best_rf, artifact_path="model", registered_model_name="CreditRiskModel")

    # --------------------------
    # Gradient Boosting (XGBoost)
    # --------------------------
    with mlflow.start_run(run_name="XGBoost_RandomSearch"):
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
        xgb_param_dist = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 10],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 1.0]
        }
        random_xgb = RandomizedSearchCV(xgb, param_distributions=xgb_param_dist, n_iter=20,
                                        scoring="roc_auc", cv=5, random_state=RANDOM_STATE, n_jobs=-1)
        random_xgb.fit(X_train, y_train)
        best_xgb = random_xgb.best_estimator_
        metrics_dict["XGBoost"] = evaluate_model(best_xgb, X_test, y_test)
        mlflow.log_params(random_xgb.best_params_)
        mlflow.log_metrics(metrics_dict["XGBoost"])
        mlflow.sklearn.log_model(best_xgb, artifact_path="model", registered_model_name="CreditRiskModel")

    
    return metrics_dict

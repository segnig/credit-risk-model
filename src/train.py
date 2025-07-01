import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

def evaluate_model(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba)
    }

def train_model(model, param_grid, X_train, y_train, X_test, y_test, model_name):
    with mlflow.start_run(run_name=model_name):
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(y_test, y_pred, y_proba)

        # Log everything
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, model_name)
        print(f"✅ {model_name} logged to MLflow")

        return best_model, metrics


if __name__ == "__main__":
    df = pd.read_csv("data/processed/data_with_target.csv")

    X = df.drop(columns=["CustomerId", "is_high_risk", "TransactionStartTime"])
    y = df["is_high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment("credit-risk-model")

    # Logistic Regression
    lr_model = LogisticRegression(solver='liblinear')
    lr_params = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2']
    }
    train_model(lr_model, lr_params, X_train, y_train, X_test, y_test, "LogisticRegression")

    # Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_params = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    train_model(rf_model, rf_params, X_train, y_train, X_test, y_test, "RandomForest")

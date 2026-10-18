"""Train the churn model, log it to MLflow, register it and set the `champion` alias."""
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
MODEL_NAME = "churn-model"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL = ["Contract", "InternetService", "PaymentMethod"]

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("churn")

df = pd.read_csv(DATA_URL)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
X = df[NUMERIC + CATEGORICAL]
y = (df["Churn"] == "Yes").astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing + model in ONE pipeline -> no training/serving skew
pipeline = Pipeline([
    ("prep", ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])),
    ("model", LogisticRegression(max_iter=1000, C=1.0)),
])

with mlflow.start_run() as run:
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, proba > 0.5),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    mlflow.log_params({"C": 1.0, "max_iter": 1000})
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(pipeline, "model", input_example=X_test.head(2))
    print("Metrics:", metrics)

version = mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME)
MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", version.version)
print(f"Registered {MODEL_NAME} v{version.version} as @champion")

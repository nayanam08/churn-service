"""FastAPI service that serves the `champion` churn model."""
import os
from contextlib import asynccontextmanager
from typing import Literal

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_URI = "models:/churn-model@champion"
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = mlflow.sklearn.load_model(MODEL_URI)  # load once at startup
    yield


app = FastAPI(title="Churn Prediction API", lifespan=lifespan)


class Customer(BaseModel):
    tenure: int = Field(ge=0, le=100, examples=[12])
    MonthlyCharges: float = Field(ge=0, examples=[70.5])
    TotalCharges: float = Field(ge=0, examples=[846.0])
    SeniorCitizen: Literal[0, 1] = Field(examples=[0])
    Contract: Literal["Month-to-month", "One year", "Two year"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict")
def predict(customer: Customer):
    if "model" not in state:
        raise HTTPException(503, "Model not loaded")
    X = pd.DataFrame([customer.model_dump()])
    proba = float(state["model"].predict_proba(X)[0, 1])
    return {"churn_probability": round(proba, 4), "will_churn": proba > 0.5}

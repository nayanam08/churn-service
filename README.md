---
title: Churn Prediction API
emoji: 📉
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
---

# Churn Prediction Service

An MLOps learning project: a scikit-learn churn model served with FastAPI, packaged with Docker and deployed to Render.

## Run locally
```bash
pip install -r requirements.txt
python train.py
uvicorn app:app --port 8000      # open http://localhost:8000/docs
pytest                           # run tests (after train.py)
```

## Run with Docker
```bash
docker build -t churn-api .
docker run -p 7860:7860 churn-api
```

## Endpoints
- `GET /health` – liveness check
- `POST /predict` – returns churn probability for one customer

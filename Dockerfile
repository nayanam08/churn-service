FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train at build time so the image ships with a registered model (replaced in Stage 5)
RUN python train.py

EXPOSE 7860
# Use the port the host gives us ($PORT on Render), default 7860
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}

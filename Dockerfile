FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY mlruns ./mlruns
COPY data ./data

RUN pip install --no-cache-dir fastapi uvicorn pandas scikit-learn mlflow joblib pyarrow prometheus-client

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
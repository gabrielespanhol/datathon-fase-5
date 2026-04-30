FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

COPY pyproject.toml ./
COPY src ./src
COPY mlruns ./mlruns
COPY data ./data
COPY docs/ ./docs/
COPY mlflow.db /app/mlflow.db

# 1. Instala o Torch com o índice correto (essencial para o +cu121 ser encontrado)
RUN pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install -e ".[awq,config,async,utils,serialization]"

EXPOSE 8000

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
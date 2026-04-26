.PHONY: data process train repro retrain test lint api mlflow dvc-status \
        docker-up docker-down docker-build docker-logs docker-restart

PYTORCH_INDEX=https://download.pytorch.org/whl/cu121

data:
	python -m src.scripts.generate_fraud_data

process:
	python -m src.scripts.process_data

train:
	dvc repro

retrain:
	dvc repro -f

dvc-status:
	dvc status

test:
	python -m pytest  

lint:
	python -m ruff check . --fix  

api:
	uvicorn src.serving.app:app --reload

freeze:
	pip freeze > requirements.txt

install-all:
	pip install --upgrade pip setuptools wheel
	pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --index-url $(PYTORCH_INDEX)
	pip install -e ".[dev,ml,awq,celery,dvc,config,cli,async,utils,serialization,graph]"

mlflow:
	mlflow ui

build:
	docker-compose build

up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-restart:
	docker-compose down && docker-compose up -d

docker-logs:
	docker-compose logs -f
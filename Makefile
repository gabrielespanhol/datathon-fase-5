.PHONY: data process train repro retrain test lint api mlflow dvc-status \
        docker-up docker-down docker-build docker-logs docker-restart

data:
	python -m src.scripts.generate_fraud_data

process:
	python -m src.scripts.process_data

train:
	python -m src.models.train

repro:
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

mlflow:
	mlflow ui

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-restart:
	docker-compose down && docker-compose up -d

docker-logs:
	docker-compose logs -f
.PHONY: data process train repro retrain test lint api mlflow dvc-status

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
	pytest

lint:
	ruff check .

api:
	uvicorn src.serving.app:app --reload

mlflow:
	mlflow ui
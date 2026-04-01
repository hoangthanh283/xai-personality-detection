.PHONY: setup install test lint clean data kb baseline rag-xpr evaluate

# Setup
setup:
	uv venv --python 3.12 .venv
	uv pip install --python .venv/bin/python -r requirements.txt
	.venv/bin/python -m spacy download en_core_web_sm

install:
	uv pip install --python .venv/bin/python -r requirements.txt

# Testing
test:
	pytest tests/ -v

lint:
	ruff check src/ scripts/ tests/
	ruff format --check src/ scripts/ tests/

format:
	ruff format src/ scripts/ tests/

# Data pipeline
data-preprocess:
	python scripts/preprocess_data.py --all

data-mbti:
	python scripts/preprocess_data.py --dataset mbti

data-essays:
	python scripts/preprocess_data.py --dataset essays

data-pandora:
	python scripts/preprocess_data.py --dataset pandora

data-verify:
	python scripts/preprocess_data.py --verify

# Knowledge base
kb-build:
	docker compose up -d qdrant
	python scripts/build_kb.py --step all --config configs/kb_config.yaml

kb-verify:
	python scripts/build_kb.py --step verify

# Baseline training
baseline-ml:
	python scripts/train_baseline.py --model all_ml --dataset mbti --task 16class

baseline-distilbert:
	python scripts/train_baseline.py --model distilbert --dataset mbti --task 16class

baseline-roberta:
	python scripts/train_baseline.py --model roberta --dataset mbti --task 16class

# RAG-XPR
rag-xpr-run:
	python scripts/run_rag_xpr.py --config configs/rag_xpr_config.yaml --dataset mbti --split test

rag-xpr-dry:
	python scripts/run_rag_xpr.py --config configs/rag_xpr_config.yaml --dataset mbti --dry_run 10

# Evaluation
evaluate:
	python scripts/evaluate.py --mode full --predictions_dir outputs/predictions/ --output outputs/reports/

# Docker
docker-up:
	docker compose up -d

docker-down:
	docker compose down

# Clean outputs
clean-outputs:
	rm -rf outputs/predictions/* outputs/models/* outputs/reports/*

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.egg-info" -exec rm -rf {} +

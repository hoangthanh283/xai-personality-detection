.PHONY: setup install test lint clean data kb baseline rag-xpr evaluate data-download data-convert-evd data-preprocess

UV_RUN=uv run --no-project --python 3.12 --with-requirements requirements.txt

# Setup
setup:
	$(UV_RUN) python -m spacy download en_core_web_sm

install:
	@echo "No separate install step needed. Use 'uv run --python 3.12 --with-requirements requirements.txt ...'"

# Testing
test:
	$(UV_RUN) pytest tests/ -v

lint:
	$(UV_RUN) ruff check src/ scripts/ tests/
	$(UV_RUN) ruff format --check src/ scripts/ tests/

format:
	$(UV_RUN) ruff format src/ scripts/ tests/

# Data pipeline
data-download:
	$(UV_RUN) python scripts/download_data.py --all

data-convert-evd:
	$(UV_RUN) python scripts/convert_personality_evd.py --input_dir data/raw/personality_evd --output_dir data/raw/personality_evd

data-preprocess:
	$(UV_RUN) python scripts/preprocess_data.py --all

data-mbti:
	$(UV_RUN) python scripts/preprocess_data.py --dataset mbti

data-essays:
	$(UV_RUN) python scripts/preprocess_data.py --dataset essays

data-pandora:
	$(UV_RUN) python scripts/preprocess_data.py --dataset pandora

data-verify:
	$(UV_RUN) python scripts/preprocess_data.py --verify

# Knowledge base
kb-build:
	docker compose up -d qdrant
	$(UV_RUN) python scripts/build_kb.py --step all --config configs/kb_config.yaml

kb-verify:
	$(UV_RUN) python scripts/build_kb.py --step verify

# Baseline training
baseline-ml:
	$(UV_RUN) python scripts/train_baseline.py --model all_ml --dataset mbti --task 16class

baseline-distilbert:
	$(UV_RUN) python scripts/train_baseline.py --model distilbert --dataset mbti --task 16class

baseline-roberta:
	$(UV_RUN) python scripts/train_baseline.py --model roberta --dataset mbti --task 16class

# RAG-XPR
rag-xpr-run:
	$(UV_RUN) python scripts/run_rag_xpr.py --config configs/rag_xpr_config.yaml --dataset mbti --split test

rag-xpr-dry:
	$(UV_RUN) python scripts/run_rag_xpr.py --config configs/rag_xpr_config.yaml --dataset mbti --dry_run 10

# Evaluation
evaluate:
	$(UV_RUN) python scripts/evaluate.py --mode full --predictions_dir outputs/predictions/ --output outputs/reports/

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

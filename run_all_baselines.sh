#!/bin/bash
set -e
cd /mnt/DataDrive/Workspace/Master-HUST/NLP/xai-personality-detection

echo "=== Starting ML 16class grid search ==="
uv run --no-project --python 3.12 --with-requirements requirements.txt python scripts/train_baseline.py --wandb_project XAI-RAG --model all_ml --dataset mbti --task 16class --grid_search 2>&1 | tee outputs/reports/ml_16class_grid.log

echo "=== Starting DistilBERT 16class ==="
uv run --no-project --python 3.12 --with-requirements requirements.txt python scripts/train_baseline.py --wandb_project XAI-RAG --model distilbert --dataset mbti --task 16class 2>&1 | tee outputs/reports/distilbert_v3.log

echo "=== All baselines complete ==="

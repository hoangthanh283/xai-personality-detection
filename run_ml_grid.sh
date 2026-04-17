#!/bin/bash
cd /mnt/DataDrive/Workspace/Master-HUST/NLP/xai-personality-detection
uv run --no-project --python 3.12 --with-requirements requirements.txt python scripts/train_baseline.py --wandb_project XAI-RAG --model all_ml --dataset mbti --task 16class --grid_search 2>&1 | tee outputs/reports/ml_baselines_grid_search.log

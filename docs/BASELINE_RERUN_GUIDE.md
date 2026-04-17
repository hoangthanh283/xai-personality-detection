# Full Baseline Rerun Guide

This guide covers the full baseline matrix across all datasets and all baseline model families currently supported by the codebase.

Baseline model families:

- Classical ML: `logistic_regression`, `svm`, `naive_bayes`, `xgboost`, `random_forest`
- Ensemble: `ensemble`
- Transformers: `distilbert`, `roberta`

Dataset/task coverage:

- `mbti`
  - `16class`
  - `4dim`
- `essays`
  - `ocean_binary`
- `pandora_big5`
  - `ocean_binary`
- `personality_evd`
  - `ocean_binary`

Important note on `personality_evd`:

- In this workspace, the converted raw data does not contain MBTI labels.
- The conversion script writes `personality: {speaker: None}` and real labels exist under `personality_ocean`.
- So `personality_evd` should be run for OCEAN baselines, not MBTI `16class`.

Important note on runtime:

- `pandora_big5` is very large.
- Full classical and transformer runs on `pandora_big5` will be much slower than the other datasets.
- The transformer config already uses dataset-specific overrides to keep those runs more realistic.

## 1. Go To Repo Root

```bash
cd /mnt/DataDrive/Workspace/Master-HUST/NLP/xai-personality-detection
```

## 2. Refresh Personality-Evd Preprocessing

Run this once before starting the full rerun. It rebuilds `personality_evd` with the fixed custom split logic.

```bash
uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/preprocess_data.py --dataset personality_evd
```

## 3. Stop Old Runs And Reset Logs

```bash
pkill -f 'scripts/train_baseline.py' || true
pkill -f 'run_cpu_classical_baselines.sh' || true
pkill -f 'run_gpu_transformer_baselines.sh' || true

mkdir -p outputs/reports
truncate -s 0 outputs/reports/cpu_classical_baselines.log
truncate -s 0 outputs/reports/gpu_transformer_baselines.log
```

## 4. Single Command For Both Queues

If you want one launcher for everything, use:

```bash
cd /mnt/DataDrive/Workspace/Master-HUST/NLP/xai-personality-detection
bash scripts/run_all_baselines.sh
```

This wrapper:

- starts the CPU and GPU queues together
- writes logs to `outputs/reports/cpu_classical_baselines.log` and `outputs/reports/gpu_transformer_baselines.log`
- stops the sibling queue if one side fails
- stops both queues if you press `Ctrl+C`

The underlying scripts are:

- `scripts/run_all_baselines.sh`
- `scripts/run_cpu_classical_baselines.sh`
- `scripts/run_gpu_transformer_baselines.sh`

## 5. One Command Per Queue

If you prefer to control each queue manually, use these two commands.

Classical baselines:

```bash
cd /mnt/DataDrive/Workspace/Master-HUST/NLP/xai-personality-detection
bash -lc 'set -o pipefail; bash scripts/run_cpu_classical_baselines.sh 2>&1 | tee -a outputs/reports/cpu_classical_baselines.log'
```

Transformer baselines:

```bash
cd /mnt/DataDrive/Workspace/Master-HUST/NLP/xai-personality-detection
bash -lc 'set -o pipefail; CUDA_VISIBLE_DEVICES=0 bash scripts/run_gpu_transformer_baselines.sh 2>&1 | tee -a outputs/reports/gpu_transformer_baselines.log'
```

## 6. CPU Queue Contents

Run this in its own terminal:

```bash
cd /mnt/DataDrive/Workspace/Master-HUST/NLP/xai-personality-detection

bash -lc '
set -euo pipefail
set -a
source .env
set +a

uv_base=(uv run --no-project --python 3.12 --with-requirements requirements.txt python scripts/train_baseline.py --wandb_project "$WANDB_PROJECT")

"${uv_base[@]}" --model all_ml   --dataset mbti            --task 16class
"${uv_base[@]}" --model ensemble --dataset mbti            --task 16class
"${uv_base[@]}" --model all_ml   --dataset mbti            --task 4dim
"${uv_base[@]}" --model ensemble --dataset mbti            --task 4dim

"${uv_base[@]}" --model all_ml   --dataset essays          --task ocean_binary
"${uv_base[@]}" --model ensemble --dataset essays          --task ocean_binary

"${uv_base[@]}" --model all_ml   --dataset pandora_big5    --task ocean_binary
"${uv_base[@]}" --model ensemble --dataset pandora_big5    --task ocean_binary

"${uv_base[@]}" --model all_ml   --dataset personality_evd --task ocean_binary
"${uv_base[@]}" --model ensemble --dataset personality_evd --task ocean_binary
' 2>&1 | tee -a outputs/reports/cpu_classical_baselines.log
```

## 7. GPU Queue Contents

Run this in a second terminal:

```bash
cd /mnt/DataDrive/Workspace/Master-HUST/NLP/xai-personality-detection

CUDA_VISIBLE_DEVICES=0 bash -lc '
set -euo pipefail
set -a
source .env
set +a

uv_base=(uv run --no-project --python 3.12 --with-requirements requirements.txt python scripts/train_baseline.py --wandb_project "$WANDB_PROJECT")

"${uv_base[@]}" --model distilbert --dataset mbti            --task 16class
"${uv_base[@]}" --model roberta    --dataset mbti            --task 16class
"${uv_base[@]}" --model distilbert --dataset mbti            --task 4dim
"${uv_base[@]}" --model roberta    --dataset mbti            --task 4dim

"${uv_base[@]}" --model distilbert --dataset essays          --task ocean_binary
"${uv_base[@]}" --model roberta    --dataset essays          --task ocean_binary

"${uv_base[@]}" --model distilbert --dataset pandora_big5    --task ocean_binary
"${uv_base[@]}" --model roberta    --dataset pandora_big5    --task ocean_binary

"${uv_base[@]}" --model distilbert --dataset personality_evd --task ocean_binary
"${uv_base[@]}" --model roberta    --dataset personality_evd --task ocean_binary
' 2>&1 | tee -a outputs/reports/gpu_transformer_baselines.log
```

## 8. Monitor Progress

CPU log:

```bash
tail -f outputs/reports/cpu_classical_baselines.log
```

GPU log:

```bash
tail -f outputs/reports/gpu_transformer_baselines.log
```

GPU utilization:

```bash
watch -n 2 nvidia-smi
```

Running processes:

```bash
ps -o pid,etimes,%cpu,%mem,cmd -C python -C python3
```

## 9. Optional Single-Run Commands

Example: RoBERTa on MBTI 4-dim

```bash
cd /mnt/DataDrive/Workspace/Master-HUST/NLP/xai-personality-detection
set -a
source .env
set +a

uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/train_baseline.py \
  --wandb_project "$WANDB_PROJECT" \
  --model roberta \
  --dataset mbti \
  --task 4dim
```

Example: all classical models on Pandora Big5

```bash
cd /mnt/DataDrive/Workspace/Master-HUST/NLP/xai-personality-detection
set -a
source .env
set +a

uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/train_baseline.py \
  --wandb_project "$WANDB_PROJECT" \
  --model all_ml \
  --dataset pandora_big5 \
  --task ocean_binary
```

Example: DistilBERT on Personality-Evd

```bash
cd /mnt/DataDrive/Workspace/Master-HUST/NLP/xai-personality-detection
set -a
source .env
set +a

uv run --no-project --python 3.12 --with-requirements requirements.txt \
  python scripts/train_baseline.py \
  --wandb_project "$WANDB_PROJECT" \
  --model distilbert \
  --dataset personality_evd \
  --task ocean_binary
```

## 10. Notes

- All baseline families now log consistent `train_*`, `eval_*`, and `test_*` metrics to W&B.
- Accuracy, F1, precision, and recall are included.
- Transformer baselines explicitly load pretrained checkpoints via Hugging Face `from_pretrained`.
- `personality_evd` is now re-split from the converted raw files with `UNKNOWN` OCEAN labels removed.
- If you want classical hyperparameter search, add `--grid_search` to the classical commands.

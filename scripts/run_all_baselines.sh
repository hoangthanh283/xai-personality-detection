#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p outputs/reports

CPU_LOG="outputs/reports/cpu_classical_baselines.log"
GPU_LOG="outputs/reports/gpu_transformer_baselines.log"
GPU_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"

: > "$CPU_LOG"
: > "$GPU_LOG"

CPU_PID=""
GPU_PID=""

terminate_children() {
  local pid
  for pid in "$CPU_PID" "$GPU_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}

on_interrupt() {
  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stopping both baseline queues..."
  terminate_children
  wait "${CPU_PID:-}" 2>/dev/null || true
  wait "${GPU_PID:-}" 2>/dev/null || true
  exit 130
}

trap on_interrupt INT TERM

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting classical baseline queue..."
bash scripts/run_cpu_classical_baselines.sh > >(tee -a "$CPU_LOG") 2>&1 &
CPU_PID=$!

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting transformer baseline queue on CUDA_VISIBLE_DEVICES=$GPU_DEVICE..."
CUDA_VISIBLE_DEVICES="$GPU_DEVICE" bash scripts/run_gpu_transformer_baselines.sh \
  > >(tee -a "$GPU_LOG") 2>&1 &
GPU_PID=$!

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Queues started."
echo "  CPU log: $CPU_LOG"
echo "  GPU log: $GPU_LOG"
echo "  CPU PID: $CPU_PID"
echo "  GPU PID: $GPU_PID"

set +e
wait -n "$CPU_PID" "$GPU_PID"
FIRST_STATUS=$?
set -e

if [[ $FIRST_STATUS -ne 0 ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] One queue failed. Stopping the other queue..."
  terminate_children
fi

set +e
wait "$CPU_PID"
CPU_STATUS=$?
wait "$GPU_PID"
GPU_STATUS=$?
set -e

if [[ $CPU_STATUS -ne 0 || $GPU_STATUS -ne 0 ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Baseline run finished with failures."
  echo "  CPU exit code: $CPU_STATUS"
  echo "  GPU exit code: $GPU_STATUS"
  exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All baseline queues completed successfully."

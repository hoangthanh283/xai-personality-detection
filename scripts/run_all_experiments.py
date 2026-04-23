#!/usr/bin/env python3
"""Orchestrate the full experiment matrix.

Usage:
    python scripts/run_all_experiments.py --group baselines
    python scripts/run_all_experiments.py --group rag_xpr
    python scripts/run_all_experiments.py --group ablations
    python scripts/run_all_experiments.py --group personality_evd
    python scripts/run_all_experiments.py --all --wandb_project rag-xpr

Baseline experiments run CPU (classical ML) and GPU (transformer) queues in
parallel via run_cpu_classical_baselines.sh and run_gpu_transformer_baselines.sh.
"""
import argparse
import os
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_logging  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_command(cmd: list[str]) -> int:
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error(f"Command failed with exit code {result.returncode}")
    return result.returncode


def _stream(src, sinks: list) -> None:
    for line in iter(src.readline, b""):
        for sink in sinks:
            sink.write(line)
            sink.flush()
    src.close()


def _launch_queue(script: Path, log_path: Path, env: dict) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "wb")
    proc = subprocess.Popen(
        ["bash", str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    threading.Thread(
        target=_stream,
        args=(proc.stdout, [sys.stdout.buffer, log_file]),
        daemon=True,
    ).start()
    return proc


def run_baselines(args, seed: int) -> None:
    """Run all baseline experiments in parallel CPU+GPU queues."""
    cpu_script = REPO_ROOT / "scripts" / "run_cpu_classical_baselines.sh"
    gpu_script = REPO_ROOT / "scripts" / "run_gpu_transformer_baselines.sh"
    cpu_log = REPO_ROOT / "outputs" / "reports" / "cpu_classical_baselines.log"
    gpu_log = REPO_ROOT / "outputs" / "reports" / "gpu_transformer_baselines.log"

    base_env = {**os.environ, "REPO_ROOT": str(REPO_ROOT), "SEED": str(seed)}
    gpu_device = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_env = {**base_env, "CUDA_VISIBLE_DEVICES": gpu_device}

    procs: list[subprocess.Popen] = []

    def _terminate_all(signum=None, frame=None) -> None:
        logger.warning("Stopping all baseline queues...")
        for p in procs:
            try:
                p.terminate()
            except OSError:
                pass
        for p in procs:
            p.wait()
        sys.exit(130)

    signal.signal(signal.SIGINT, _terminate_all)
    signal.signal(signal.SIGTERM, _terminate_all)

    logger.info(f"[{_ts()}] Starting classical baseline queue → {cpu_log}")
    procs.append(_launch_queue(cpu_script, cpu_log, base_env))

    logger.info(f"[{_ts()}] Starting transformer baseline queue (CUDA={gpu_device}) → {gpu_log}")
    procs.append(_launch_queue(gpu_script, gpu_log, gpu_env))

    cpu_status = procs[0].wait()
    gpu_status = procs[1].wait()

    if cpu_status != 0 or gpu_status != 0:
        logger.error(f"Baseline queues finished with failures — CPU:{cpu_status} GPU:{gpu_status}")
        sys.exit(1)

    logger.info(f"[{_ts()}] All baseline queues completed successfully.")


def run_rag_xpr(args, seed: int) -> None:
    """Run RAG-XPR experiments (R1-R6)."""
    base_cmd = [sys.executable, "scripts/run_rag_xpr.py", "--config", "configs/rag_xpr_config.yaml", "--seed", str(seed)]

    experiments = [
        [
            "--dataset", "mbti",
            "--llm_provider", "openrouter",
            "--llm_model", "qwen/qwen3.6-plus-preview:free",
        ],
        ["--dataset", "mbti"],
        ["--dataset", "mbti", "--llm_provider", "vllm", "--llm_model", "meta-llama/Llama-3.1-8B-Instruct"],
        ["--dataset", "essays", "--framework", "ocean"],
        ["--dataset", "pandora", "--framework", "ocean"],
        ["--dataset", "personality_evd", "--framework", "ocean"],
    ]

    for exp_args in experiments:
        run_command(base_cmd + exp_args)


def run_ablations(args, seed: int) -> None:
    """Run ablation studies (A1-A8)."""
    base_cmd = [sys.executable, "scripts/run_rag_xpr.py", "--config", "configs/rag_xpr_config.yaml",
                "--dataset", "mbti", "--sample", "500", "--seed", str(seed)]
    ablations = ["no_kb", "no_evidence_filter", "no_cope", "no_step_2", "semantic_only", "keyword_only", "small_kb", "large_kb"]
    for ablation in ablations:
        run_command(base_cmd + ["--ablation", ablation, "--output", f"outputs/predictions/ablation_{ablation}.jsonl"])


def run_llm_direct(args, seed: int) -> None:
    """Run LLM Direct experiments (L1-L6)."""
    base_cmd = [sys.executable, "scripts/run_rag_xpr.py", "--config", "configs/rag_xpr_config.yaml",
                "--mode", "llm_direct", "--dataset", "mbti", "--seed", str(seed)]

    experiments = [
        ["--prompt", "zero_shot", "--sample", "500", "--output", "outputs/predictions/llm_direct_L1.jsonl"],
        ["--prompt", "few_shot", "--sample", "500", "--output", "outputs/predictions/llm_direct_L2.jsonl"],
        ["--prompt", "cot_basic", "--sample", "500", "--output", "outputs/predictions/llm_direct_L3.jsonl"],
        ["--prompt", "zero_shot", "--sample", "200", "--llm_provider", "openai", "--llm_model", "gpt-4o", "--output", "outputs/predictions/llm_direct_L4.jsonl"],
        ["--prompt", "zero_shot", "--sample", "500", "--llm_provider", "vllm", "--llm_model", "meta-llama/Llama-3.1-8B-Instruct", "--output", "outputs/predictions/llm_direct_L5.jsonl"],
        ["--prompt", "few_shot", "--sample", "500", "--llm_provider", "vllm", "--llm_model", "meta-llama/Llama-3.1-8B-Instruct", "--output", "outputs/predictions/llm_direct_L6.jsonl"],
    ]
    for exp_args in experiments:
        run_command(base_cmd + exp_args)


def run_personality_evd(args, seed: int) -> None:
    """Run Personality Evd experiments (E1-E4)."""
    # E1: RAG-XPR (full)
    run_command([sys.executable, "scripts/run_rag_xpr.py", "--config", "configs/rag_xpr_config.yaml",
                 "--dataset", "personality_evd", "--framework", "ocean", "--seed", str(seed),
                 "--output", "outputs/predictions/evd_E1_rag_xpr.jsonl"])

    # E2: LLM + CoPE (no RAG)
    run_command([sys.executable, "scripts/run_rag_xpr.py", "--config", "configs/rag_xpr_config.yaml",
                 "--dataset", "personality_evd", "--framework", "ocean", "--seed", str(seed), "--ablation", "no_kb",
                 "--output", "outputs/predictions/evd_E2_no_rag.jsonl"])

    # E3: LLM zero-shot
    run_command([sys.executable, "scripts/run_rag_xpr.py", "--config", "configs/rag_xpr_config.yaml",
                 "--mode", "llm_direct", "--prompt", "zero_shot",
                 "--dataset", "personality_evd", "--framework", "ocean", "--seed", str(seed),
                 "--output", "outputs/predictions/evd_E3_llm_direct.jsonl"])

    # E4: DistilBERT OCEAN baseline
    run_command([sys.executable, "scripts/train_baseline.py", "--model", "distilbert",
                 "--dataset", "personality_evd", "--task", "ocean_binary", "--seed", str(seed)])


def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--group", choices=["baselines", "rag_xpr", "ablations", "personality_evd", "llm_direct"])
    parser.add_argument("--all", action="store_true", help="Run all experiment groups")
    parser.add_argument("--wandb_project", default="rag-xpr")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds")
    parser.add_argument("--output_dir", default="outputs/")
    args = parser.parse_args()

    setup_logging()

    seeds = [int(s) for s in args.seeds.split(",")]
    logger.info(f"Running with seeds: {seeds}")

    groups_to_run = []
    if args.all:
        groups_to_run = ["baselines", "llm_direct", "rag_xpr", "ablations", "personality_evd"]
    elif args.group:
        groups_to_run = [args.group]

    for seed in seeds:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# RUNNING EXPERIMENTS WITH SEED: {seed}")
        logger.info(f"{'#'*60}")
        for group in groups_to_run:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running experiment group: {group} (Seed: {seed})")
            logger.info(f"{'='*60}")
            if group == "baselines":
                run_baselines(args, seed)
            elif group == "rag_xpr":
                run_rag_xpr(args, seed)
            elif group == "ablations":
                run_ablations(args, seed)
            elif group == "llm_direct":
                run_llm_direct(args, seed)
            elif group == "personality_evd":
                run_personality_evd(args, seed)


if __name__ == "__main__":
    main()

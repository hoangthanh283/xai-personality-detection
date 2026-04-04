#!/usr/bin/env python
"""Run RAG-XPR inference pipeline.

Usage:
    python scripts/run_rag_xpr.py --config configs/rag_xpr_config.yaml --dataset mbti --split test
    python scripts/run_rag_xpr.py --config configs/rag_xpr_config.yaml --dry_run 10
    python scripts/run_rag_xpr.py --config configs/rag_xpr_config.yaml --mode llm_direct --prompt zero_shot --sample 500
"""
import argparse
import json
import sys
from pathlib import Path

import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_config import setup_logging  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Handle _base inheritance
    if "_base" in cfg:
        base_path = Path(config_path).parent / cfg.pop("_base")
        if base_path.exists():
            with open(base_path) as f:
                base_cfg = yaml.safe_load(f)
            base_cfg.update(cfg)
            cfg = base_cfg
    return cfg


def run_full_pipeline(args, config: dict) -> None:
    """Run the full RAG-XPR pipeline on a dataset split."""
    from src.data.loader import DataLoader
    from src.rag_pipeline.pipeline import RAGXPRPipeline

    loader = DataLoader("data/processed")
    records = loader.load_split(args.dataset, args.split)

    # Optionally sample for dry run or cost saving
    if args.dry_run:
        records = records[:args.dry_run]
        logger.info(f"DRY RUN: processing only {args.dry_run} samples")
    elif args.sample:
        import random
        records = random.sample(records, min(args.sample, len(records)))
        logger.info(f"Sampling {len(records)} records")

    logger.info("Initializing RAG-XPR pipeline...")
    pipeline = RAGXPRPipeline(config)

    # Determine output path
    output_path = args.output or config.get("output", {}).get("output_dir", "outputs/predictions/")
    output_file = (
        Path(output_path) / f"rag_xpr_{args.dataset}_{args.split}.jsonl"
        if not args.output or Path(args.output).is_dir()
        else Path(args.output)
    )
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running inference on {len(records)} records...")
    results = []
    with open(output_file, "w", encoding="utf-8") as f_out:
        for i, record in enumerate(records):
            try:
                text = record.get("text", "")
                gold_label = record.get("label_mbti") or str(record.get("label_ocean", ""))
                result = pipeline.predict(text)
                output_record = {
                    "id": record.get("id", f"sample_{i}"),
                    "text": text,
                    "gold_label": gold_label,
                    **result,
                }
                f_out.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                results.append(output_record)
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(records)}")
            except Exception as e:
                logger.error(f"Failed on record {record.get('id', i)}: {e}")

    logger.info(f"Saved {len(results)} predictions to {output_file}")

    # Quick accuracy estimate
    correct = sum(
        1 for r in results
        if r.get("predicted_label", "").upper() == r.get("gold_label", "").upper()
    )
    if results:
        logger.info(f"Quick accuracy: {correct}/{len(results)} = {correct/len(results):.2%}")


def run_llm_direct(args, config: dict) -> None:
    """Run LLM direct baseline (no RAG, no CoPE)."""
    from src.data.loader import DataLoader
    from src.rag_pipeline.llm_client import build_llm_client

    loader = DataLoader("data/processed")
    records = loader.load_split(args.dataset, args.split or "test")

    if args.sample:
        import random
        records = random.sample(records, min(args.sample, len(records)))

    llm = build_llm_client(config["llm"])

    prompt_templates = {
        "zero_shot": (
            "You are an expert psychologist. Based on the following text, predict the person's MBTI personality type. "
            "Return JSON: {\"mbti\": \"XXXX\"}\n\nText: {text}"
        ),
        "few_shot": (
            "You are an expert psychologist. Examples:\n"
            "- 'I love spending time alone reading' → INTJ\n"
            "- 'I get energy from social events and love meeting people' → ENFP\n\n"
            "Now predict the MBTI type for:\nText: {text}\n\nReturn JSON: {\"mbti\": \"XXXX\"}"
        ),
        "cot_basic": (
            "You are an expert psychologist. Analyze this text for personality traits, "
            "reason step by step, then predict the MBTI type.\n\nText: {text}\n\n"
            "Return JSON: {\"reasoning\": \"...\", \"mbti\": \"XXXX\"}"
        ),
    }

    prompt_template = prompt_templates.get(args.prompt or "zero_shot", prompt_templates["zero_shot"])
    output_path = args.output or f"outputs/predictions/llm_direct_{args.prompt or 'zero_shot'}_{args.dataset}.jsonl"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    results = []
    with open(output_path, "w", encoding="utf-8") as f_out:
        for i, record in enumerate(records):
            text = record.get("text", "")[:2000]  # Truncate for cost
            gold = record.get("label_mbti", "")
            prompt = prompt_template.format(text=text)
            try:
                response = llm.generate([{"role": "user", "content": prompt}])
                data = json.loads(response)
                pred = data.get("mbti", "UNKNOWN").strip().upper()
                output = {"id": record.get("id", i), "text": text, "gold_label": gold, "predicted_label": pred, "raw_response": response}
                f_out.write(json.dumps(output, ensure_ascii=False) + "\n")
                results.append(output)
            except Exception as e:
                logger.error(f"LLM call failed at {i}: {e}")

    correct = sum(1 for r in results if r.get("predicted_label", "") == r.get("gold_label", ""))
    logger.info(f"LLM Direct accuracy: {correct}/{len(results)} = {correct/max(len(results), 1):.2%}")
    logger.info(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run RAG-XPR inference pipeline")
    parser.add_argument("--config", default="configs/rag_xpr_config.yaml")
    parser.add_argument("--dataset", default="mbti", choices=["mbti", "essays", "pandora", "personality_evd"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--llm_provider", choices=["openrouter", "openai", "vllm", "ollama"])
    parser.add_argument("--llm_model", help="Override LLM model name")
    parser.add_argument("--framework", choices=["mbti", "ocean"])
    parser.add_argument("--dry_run", type=int, help="Process only N samples")
    parser.add_argument("--sample", type=int, help="Random sample N records")
    parser.add_argument("--mode", choices=["full", "llm_direct"], default="full")
    parser.add_argument("--prompt", choices=["zero_shot", "few_shot", "cot_basic"])
    parser.add_argument("--ablation", help="Ablation config override name")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)
    config = load_config(args.config)

    # Apply global config overrides
    if args.llm_provider:
        config["llm"]["provider"] = args.llm_provider
    if args.llm_model:
        config["llm"]["model"] = args.llm_model
    if args.framework:
        config["cope"]["framework"] = args.framework

    # Apply ablation overrides
    if args.ablation == "no_kb":
        config["retrieval"]["skip_kb"] = True
    elif args.ablation == "no_cope":
        config["cope"]["skip_steps"] = [2, 3]
    elif args.ablation == "no_evidence_filter":
        config["evidence_retrieval"]["pre_filter"] = False
    elif args.ablation == "no_step_2":
        config["cope"]["skip_steps"] = [2]
    elif args.ablation == "semantic_only":
        config["evidence_retrieval"]["method"] = "semantic"
    elif args.ablation == "keyword_only":
        config["evidence_retrieval"]["method"] = "keyword"
    elif args.ablation == "small_kb":
        config["retrieval"]["collection"] = config["retrieval"].get("collection", "psych_kb") + "_small"
    elif args.ablation == "large_kb":
        config["retrieval"]["collection"] = config["retrieval"].get("collection", "psych_kb") + "_large"

    if args.mode == "llm_direct":
        run_llm_direct(args, config)
    else:
        run_full_pipeline(args, config)


if __name__ == "__main__":
    main()

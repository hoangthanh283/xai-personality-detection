#!/usr/bin/env python
"""Run RAG-XPR inference pipeline.

Usage:
    python scripts/run_rag_xpr.py --config configs/rag_xpr_config.yaml --dataset mbti --split test
    python scripts/run_rag_xpr.py --config configs/rag_xpr_config.yaml --dry_run 10
    python scripts/run_rag_xpr.py --config configs/rag_xpr_config.yaml --mode llm_direct --prompt zero_shot --sample 500
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv  # noqa: E402
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on shell env vars

from src.utils.logging_config import setup_logging  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.utils.observability import MultiBackendLogger  # noqa: E402
from src.utils.wandb_inference import InferenceLogger  # noqa: E402


def _parse_label_dict(label_str: str | dict | None) -> dict | None:
    """Parse comma-joined `O:HIGH,C:LOW,...` into {trait: HIGH/LOW} dict."""
    if not label_str:
        return None
    if isinstance(label_str, dict):
        return {k: str(v).upper() for k, v in label_str.items()}
    if not isinstance(label_str, str):
        return None
    parsed: dict[str, str] = {}
    for chunk in label_str.split(","):
        if ":" in chunk:
            k, _, v = chunk.partition(":")
            parsed[k.strip().upper()] = v.strip().upper()
    return parsed or None


def _setup_wandb_logger(args, config: dict, run_name_suffix: str) -> MultiBackendLogger | None:
    """Initialize a MultiBackendLogger for an inference run if wandb_project provided."""
    wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT")
    if not wandb_project:
        return None
    tier_id = config.get("tier", "tier_unknown")
    run_name = f"{tier_id}_{run_name_suffix}_{args.dataset}"
    if args.ablation:
        run_name = f"{tier_id}_{args.ablation}_{args.dataset}"
    tags = list((config.get("wandb", {}) or {}).get("tags", []))
    for t in [tier_id, args.dataset, args.ablation or "default"]:
        if t and t not in tags:
            tags.append(t)
    group = (config.get("wandb", {}) or {}).get("group", tier_id)
    tb_dir = None
    if (config.get("tensorboard", {}) or {}).get("enabled", True):
        tb_dir = f"outputs/tensorboard/{tier_id}/{run_name}"
    return MultiBackendLogger.init_run(
        project=wandb_project,
        name=run_name,
        tags=tags,
        group=group,
        config={k: v for k, v in config.items() if k not in {"_base"}} | {
            "dataset": args.dataset,
            "split": args.split,
            "seed": args.seed,
            "ablation": args.ablation or "default",
            "framework": (config.get("cope", {}) or {}).get("framework"),
        },
        tensorboard_dir=tb_dir,
    )


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


def _format_gold_ocean(ocean_dict: dict | None) -> str:
    if not ocean_dict:
        return ""
    return ",".join(f"{t}:{ocean_dict[t]}" for t in ["O", "C", "E", "A", "N"] if t in ocean_dict)


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

    # Determine output path
    output_path = args.output or config.get("output", {}).get("output_dir", "outputs/predictions/")
    output_file = (
        Path(output_path) / f"rag_xpr_{args.dataset}_{args.split}.jsonl"
        if not args.output or Path(args.output).is_dir()
        else Path(args.output)
    )
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    completed_ids = set()
    results = []
    if args.resume and output_file.exists():
        with open(output_file, encoding="utf-8") as f_in:
            for line in f_in:
                try:
                    row = json.loads(line)
                    results.append(row)
                    if row.get("id") is not None:
                        completed_ids.add(row["id"])
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed existing output line during resume")
        before = len(records)
        records = [r for r in records if r.get("id") not in completed_ids]
        logger.info(
            f"Resuming from existing output: {len(completed_ids)} completed rows, "
            f"{len(records)} remaining out of {before}"
        )

    # W&B + tensorboard observability (Tier 4/5 — full RAG pipeline)
    mb_logger = _setup_wandb_logger(args, config, run_name_suffix="full")
    inf_logger: InferenceLogger | None = None
    if mb_logger is not None:
        log_every = (config.get("inference", {}) or {}).get("log_every_n", 20)
        inf_logger = InferenceLogger(mb_logger, log_every_n=log_every)

    logger.info("Initializing RAG-XPR pipeline...")
    pipeline = RAGXPRPipeline(config)

    framework = (config.get("cope", {}) or {}).get("framework", "mbti")
    ablation = args.ablation or (config.get("inference", {}) or {}).get("ablation", "full")
    kb_eligible = ablation not in {"no_kb"}

    logger.info(f"Running inference on {len(records)} records (ablation={ablation}, framework={framework})...")
    with open(output_file, "a" if args.resume else "w", encoding="utf-8") as f_out:
        for i, record in enumerate(records):
            sample_start = time.perf_counter()
            try:
                text = record.get("text", "")
                if framework == "ocean":
                    gold_label = _format_gold_ocean(record.get("label_ocean"))
                else:
                    gold_label = record.get("label_mbti")
                if not gold_label:
                    continue
                result = pipeline.predict(text)
                latency = time.perf_counter() - sample_start
                output_record = {
                    "id": record.get("id", f"sample_{i}"),
                    "text": text,
                    "gold_label": gold_label,
                    "latency_seconds": round(latency, 3),
                    **result,
                }
                f_out.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                f_out.flush()
                results.append(output_record)

                # Stream running metrics to W&B + tensorboard
                if inf_logger is not None and framework == "ocean":
                    pred_dict = _parse_label_dict(result.get("predicted_label"))
                    gold_dict = _parse_label_dict(gold_label)
                    intermediate = result.get("intermediate", {}) or {}
                    quotes = [
                        ev.get("quote", "") for ev in intermediate.get("step1_evidence", [])
                        if isinstance(ev, dict) and ev.get("quote")
                    ]
                    kb_chunks = [
                        c.get("chunk_id") for c in intermediate.get("kb_chunks_used", [])
                        if isinstance(c, dict)
                    ]
                    inf_logger.log_sample(
                        pred=pred_dict,
                        gold=gold_dict,
                        latency=latency,
                        json_parsed=pred_dict is not None,
                        evidence_quotes=quotes,
                        source_text=text,
                        kb_chunks_cited=kb_chunks if kb_eligible else None,
                        kb_eligible=kb_eligible,
                    )

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(records)} remaining records")
            except Exception as e:
                logger.error(f"Failed on record {record.get('id', i)}: {e}")

    logger.info(f"Saved {len(results)} predictions to {output_file}")

    # Quick accuracy estimate (legacy log)
    correct = sum(
        1 for r in results
        if str(r.get("predicted_label", "")).upper() == str(r.get("gold_label", "")).upper()
    )
    if results:
        logger.info(f"Quick exact-match accuracy: {correct}/{len(results)} = {correct/len(results):.2%}")

    # Finalize W&B + tensorboard
    if inf_logger is not None:
        inf_logger.finalize()
    if mb_logger is not None:
        mb_logger.finish()


_MBTI_PROMPTS = {
    "zero_shot": (
        "You are an expert psychologist. Based on the following text, predict the person's MBTI personality type. "
        "Return strict JSON: {{\"mbti\": \"XXXX\"}}\n\nText: {text}"
    ),
    "few_shot": (
        "You are an expert psychologist. Examples:\n"
        "- 'I love spending time alone reading' → INTJ\n"
        "- 'I get energy from social events and love meeting people' → ENFP\n\n"
        "Now predict the MBTI type for:\nText: {text}\n\nReturn strict JSON: {{\"mbti\": \"XXXX\"}}"
    ),
    "cot_basic": (
        "You are an expert psychologist. Analyze this text for personality traits, "
        "reason step by step, then predict the MBTI type.\n\nText: {text}\n\n"
        "Return strict JSON: {{\"reasoning\": \"...\", \"mbti\": \"XXXX\"}}"
    ),
}

_OCEAN_PROMPTS = {
    "zero_shot": (
        "You are an expert psychologist. Based on the following text, predict the person's Big Five (OCEAN) "
        "personality traits. For each trait output HIGH or LOW.\n\n"
        "Text: {text}\n\n"
        "Return strict JSON only: {{\"O\": \"HIGH/LOW\", \"C\": \"HIGH/LOW\", \"E\": \"HIGH/LOW\", \"A\": \"HIGH/LOW\", \"N\": \"HIGH/LOW\"}}"
    ),
    "few_shot": (
        "You are an expert psychologist. Predict Big Five traits as HIGH or LOW. Examples:\n"
        "- 'I love trying new ideas and exploring' → {{\"O\":\"HIGH\",\"C\":\"LOW\",\"E\":\"HIGH\",\"A\":\"HIGH\",\"N\":\"LOW\"}}\n"
        "- 'I worry constantly and avoid social events' → {{\"O\":\"LOW\",\"C\":\"LOW\",\"E\":\"LOW\",\"A\":\"LOW\",\"N\":\"HIGH\"}}\n\n"
        "Text: {text}\n\n"
        "Return strict JSON only: {{\"O\":\"HIGH/LOW\",\"C\":\"HIGH/LOW\",\"E\":\"HIGH/LOW\",\"A\":\"HIGH/LOW\",\"N\":\"HIGH/LOW\"}}"
    ),
    "cot_basic": (
        "You are an expert psychologist. Reason step by step about the Big Five traits in this text, "
        "then output predictions.\n\nText: {text}\n\n"
        "Return strict JSON: {{\"reasoning\": \"...\", \"O\":\"HIGH/LOW\",\"C\":\"HIGH/LOW\",\"E\":\"HIGH/LOW\",\"A\":\"HIGH/LOW\",\"N\":\"HIGH/LOW\"}}"
    ),
}


def _extract_json(response: str) -> dict | None:
    """Best-effort extraction of a JSON object from an LLM response."""
    if not response:
        return None
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    # Find the first {...} block
    import re
    match = re.search(r"\{[\s\S]*\}", response)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


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

    framework = (config.get("cope", {}) or {}).get("framework", "mbti")
    if args.framework:
        framework = args.framework
    prompts = _OCEAN_PROMPTS if framework == "ocean" else _MBTI_PROMPTS
    prompt_template = prompts.get(args.prompt or "zero_shot", prompts["zero_shot"])

    dataset_name = (args.dataset or "").lower()
    if framework == "ocean" and dataset_name in {"mbti"}:
        raise ValueError(
            f"framework='ocean' is incompatible with dataset='{dataset_name}' "
            "(MBTI records have no label_ocean). Pass --framework mbti."
        )
    if framework == "mbti" and dataset_name and dataset_name != "mbti":
        raise ValueError(
            f"framework='mbti' is incompatible with dataset='{dataset_name}'. "
            "Pass --framework ocean."
        )
    logger.info(f"run_llm_direct: dataset={dataset_name} framework={framework} prompt={args.prompt or 'zero_shot'}")

    output_path = args.output or f"outputs/predictions/llm_direct_{args.prompt or 'zero_shot'}_{args.dataset}.jsonl"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # W&B + tensorboard observability
    mb_logger = _setup_wandb_logger(args, config, run_name_suffix=f"zeroshot_{args.prompt or 'zero_shot'}")
    inf_logger: InferenceLogger | None = None
    if mb_logger is not None:
        log_every = (config.get("inference", {}) or {}).get("log_every_n", 20)
        inf_logger = InferenceLogger(mb_logger, log_every_n=log_every)

    results = []
    skipped_no_gold = 0
    with open(output_path, "w", encoding="utf-8") as f_out:
        for i, record in enumerate(records):
            text = record.get("text", "")[:2000]  # Truncate for cost
            if framework == "ocean":
                gold = _format_gold_ocean(record.get("label_ocean"))
            else:
                gold = record.get("label_mbti", "")
            if not gold:
                skipped_no_gold += 1
                continue
            prompt = prompt_template.format(text=text)
            sample_start = time.perf_counter()
            try:
                response = llm.generate([{"role": "user", "content": prompt}])
                latency = time.perf_counter() - sample_start
                data = _extract_json(response)
                parse_ok = data is not None
                if framework == "ocean":
                    if parse_ok:
                        pred_dict = {t: str(data.get(t, "")).strip().upper() for t in ["O", "C", "E", "A", "N"]}
                        pred_label = ",".join(f"{t}:{v}" for t, v in pred_dict.items() if v in {"HIGH", "LOW"})
                    else:
                        pred_dict = None
                        pred_label = "UNKNOWN"
                else:
                    pred_label = (data.get("mbti", "UNKNOWN") if data else "UNKNOWN").strip().upper()
                    pred_dict = None
                output = {
                    "id": record.get("id", i),
                    "text": text,
                    "gold_label": gold,
                    "predicted_label": pred_label,
                    "raw_response": response,
                    "latency_seconds": round(latency, 3),
                }
                f_out.write(json.dumps(output, ensure_ascii=False) + "\n")
                results.append(output)
                if inf_logger is not None and framework == "ocean":
                    inf_logger.log_sample(
                        pred=pred_dict,
                        gold=_parse_label_dict(gold),
                        latency=latency,
                        json_parsed=parse_ok,
                        evidence_quotes=None,        # zero-shot has no evidence
                        source_text=None,
                        kb_chunks_cited=None,
                        kb_eligible=False,
                    )
            except Exception as e:
                latency = time.perf_counter() - sample_start
                logger.error(f"LLM call failed at {i}: {e}")
                if inf_logger is not None:
                    inf_logger.log_sample(
                        pred=None, gold=_parse_label_dict(gold), latency=latency,
                        json_parsed=False, kb_eligible=False,
                    )

    correct = sum(1 for r in results if r.get("predicted_label", "") == r.get("gold_label", ""))
    logger.info(f"LLM Direct accuracy: {correct}/{len(results)} = {correct/max(len(results), 1):.2%}")
    logger.info(f"Saved to {output_path}")
    if skipped_no_gold:
        logger.warning(
            f"Skipped {skipped_no_gold}/{len(records)} records due to missing gold "
            f"(framework={framework}, dataset={dataset_name}). "
            "If 100%, the framework likely does not match the dataset."
        )
        if skipped_no_gold == len(records):
            raise RuntimeError(
                f"All {len(records)} records skipped (no usable gold). "
                f"Check framework={framework} vs dataset={dataset_name}."
            )
    if inf_logger is not None:
        inf_logger.finalize()
    if mb_logger is not None:
        mb_logger.finish()


def main():
    parser = argparse.ArgumentParser(description="Run RAG-XPR inference pipeline")
    parser.add_argument("--config", default="configs/rag_xpr_config.yaml")
    parser.add_argument(
        "--dataset", default="mbti", choices=["mbti", "essays", "pandora", "personality_evd"]
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--llm_provider", choices=["openrouter", "openai", "vllm", "ollama"])
    parser.add_argument("--llm_model", help="Override LLM model name")
    parser.add_argument("--framework", choices=["mbti", "ocean"])
    parser.add_argument("--dry_run", type=int, help="Process only N samples")
    parser.add_argument("--sample", type=int, help="Random sample N records")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing JSONL output file by skipping completed ids")
    parser.add_argument("--mode", choices=["full", "llm_direct"], default="full")
    parser.add_argument("--prompt", choices=["zero_shot", "few_shot", "cot_basic"])
    parser.add_argument("--ablation", help="Ablation config override name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", help="W&B project name (default: $WANDB_PROJECT env var)")
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

    # Wire dataset name into evidence_retrieval so frozen-SVM loads correct OCEAN checkpoints
    if args.dataset and args.dataset != "mbti":
        config.setdefault("evidence_retrieval", {})["roberta_dataset"] = args.dataset

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
    # RoBERTa scorer / prior ablations
    elif args.ablation == "keyword-only":
        config["evidence_retrieval"]["scorer"] = "keyword"
        config["evidence_retrieval"]["use_roberta_prior"] = False
    elif args.ablation == "roberta-scorer":
        config["evidence_retrieval"]["scorer"] = "roberta"
        config["evidence_retrieval"]["use_roberta_prior"] = False
    elif args.ablation == "roberta-prior":
        config["evidence_retrieval"]["scorer"] = "keyword"
        config["evidence_retrieval"]["use_roberta_prior"] = True
    elif args.ablation == "roberta-both":
        config["evidence_retrieval"]["backbone"] = "roberta"
        config["evidence_retrieval"]["scorer"] = "roberta"
        config["evidence_retrieval"]["use_roberta_prior"] = True
    # Frozen-BERT+SVM backbone ablations
    elif args.ablation == "frozen-svm-only":
        config["evidence_retrieval"]["backbone"] = "frozen_svm"
        config["evidence_retrieval"]["scorer"] = "roberta"
        config["evidence_retrieval"]["use_roberta_prior"] = False
    elif args.ablation == "frozen-svm-prior":
        config["evidence_retrieval"]["backbone"] = "frozen_svm"
        config["evidence_retrieval"]["scorer"] = "keyword"
        config["evidence_retrieval"]["use_roberta_prior"] = True
    elif args.ablation == "frozen-svm-both":
        config["evidence_retrieval"]["backbone"] = "frozen_svm"
        config["evidence_retrieval"]["scorer"] = "roberta"
        config["evidence_retrieval"]["use_roberta_prior"] = True

    if args.mode == "llm_direct":
        run_llm_direct(args, config)
    else:
        run_full_pipeline(args, config)


if __name__ == "__main__":
    main()

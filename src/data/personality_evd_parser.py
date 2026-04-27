"""Personality Evd dataset → JSONL parser.

Pipeline:
1. Parse dialogue structure
2. For each speaker: concatenate their utterances as the "text"
3. Preserve evidence annotations (used for XAI evaluation ground truth)
4. Split per original paper's train/val/test if provided, else 70/15/15

Output:
  data/processed/personality_evd/{train,val,test}.jsonl
  data/processed/personality_evd/evidence_gold.jsonl  # ground truth evidence
"""

import hashlib
import json
from pathlib import Path

from loguru import logger
from sklearn.model_selection import train_test_split


class PersonalityEvdParser:
    """Parses Personality Evd dialogue dataset into unified JSONL format."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.use_original_split = self.config.get("use_original_split", True)
        self.split_ratio = self.config.get("split_ratio", [0.70, 0.15, 0.15])
        self.seed = self.config.get("seed", 42)
        self.drop_unknown_ocean = self.config.get("drop_unknown_ocean", True)

    def parse_dialogue(self, dialogue_data: dict, split_name: str = "train") -> list[dict]:
        """Parse a single dialogue into per-speaker records."""
        dialogue = dialogue_data.get("dialogue", [])
        personality = dialogue_data.get("personality", {})
        personality_ocean = dialogue_data.get("personality_ocean", {})
        evidence_list = dialogue_data.get("evidence", [])

        records = []
        dialogue_id = hashlib.md5(json.dumps(dialogue).encode()).hexdigest()[:8]

        # Group utterances by speaker
        speaker_utterances: dict[str, list[str]] = {}
        for turn in dialogue:
            speaker = turn.get("speaker", "")
            utterance = turn.get("utterance", "").strip()
            if speaker and utterance:
                speaker_utterances.setdefault(speaker, []).append(utterance)

        # Build gold evidence index by speaker
        speaker_evidence: dict[str, list[dict]] = {}
        for ev in evidence_list:
            speaker = ev.get("speaker", "")
            speaker_evidence.setdefault(speaker, []).append(ev)

        for speaker, utterances in speaker_utterances.items():
            mbti_type = personality.get(speaker)
            text = " ".join(utterances)

            gold_evidence = speaker_evidence.get(speaker, [])
            record_id = f"evd_{dialogue_id}_{speaker}"

            record = {
                "id": record_id,
                "text": text,
                "label_mbti": mbti_type,
                "label_mbti_dimensions": self._parse_dimensions(mbti_type) if mbti_type else None,
                "label_ocean": personality_ocean.get(speaker),
                "source": "personality_evd",
                "split": split_name,
                "metadata": {
                    "speaker": speaker,
                    "dialogue_id": dialogue_id,
                    "num_utterances": len(utterances),
                },
                "evidence_gold": gold_evidence if gold_evidence else None,
            }
            records.append(record)

        return records

    @staticmethod
    def _parse_dimensions(mbti_type: str) -> dict[str, str] | None:
        if not mbti_type or len(mbti_type) != 4:
            return None
        t = mbti_type.upper()
        return {"IE": t[0], "SN": t[1], "TF": t[2], "JP": t[3]}

    def parse_file(self, file_path: Path, split_name: str = "train") -> list[dict]:
        records = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.extend(self.parse_dialogue(data, split_name))
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"Skipping malformed record: {e}")
        return records

    @staticmethod
    def _ocean_signature(record: dict) -> str | None:
        ocean = record.get("label_ocean") or {}
        values = [ocean.get(trait) for trait in ["O", "C", "E", "A", "N"]]
        if any(v is None for v in values):
            return None
        return "|".join(str(v).upper() for v in values)

    def _filter_records_for_ocean(self, records: list[dict]) -> list[dict]:
        if not self.drop_unknown_ocean:
            return records

        filtered = []
        dropped = 0
        for rec in records:
            ocean = rec.get("label_ocean") or {}
            values = [str(ocean.get(trait, "")).upper() for trait in ["O", "C", "E", "A", "N"]]
            if all(v in {"HIGH", "LOW"} for v in values):
                filtered.append(rec)
            else:
                dropped += 1

        if dropped:
            logger.info(f"Dropped {dropped} personality_evd records with UNKNOWN OCEAN labels")
        return filtered

    def _build_custom_splits(self, combined: list[dict]) -> dict[str, list[dict]]:
        combined = self._filter_records_for_ocean(combined)
        if not combined:
            return {"train": [], "val": [], "test": []}

        signatures = [self._ocean_signature(rec) for rec in combined]
        if any(sig is None for sig in signatures):
            raise ValueError("personality_evd custom split requires complete OCEAN signatures")

        val_size = self.split_ratio[1]
        test_size = self.split_ratio[2]
        train_idx, valtest_idx = train_test_split(
            range(len(combined)),
            test_size=(val_size + test_size),
            stratify=signatures,
            random_state=self.seed,
        )
        valtest_labels = [signatures[i] for i in valtest_idx]
        val_idx, test_idx = train_test_split(
            list(valtest_idx),
            test_size=test_size / (val_size + test_size),
            stratify=valtest_labels,
            random_state=self.seed,
        )

        all_records = {"train": [], "val": [], "test": []}
        for split_name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            for i in indices:
                rec = dict(combined[i])
                rec["split"] = split_name
                all_records[split_name].append(rec)
        return all_records

    def run(self, data_dir: str, output_dir: str) -> None:
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_records: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

        if self.use_original_split:
            for split_name in ["train", "val", "test"]:
                for possible_name in [split_name, f"{split_name}_data"]:
                    split_file = data_path / f"{possible_name}.json"
                    if not split_file.exists():
                        split_file = data_path / f"{possible_name}.jsonl"
                    if split_file.exists():
                        records = self.parse_file(split_file, split_name)
                        all_records[split_name].extend(records)
                        logger.info(f"Loaded {len(records)} records for {split_name} split")
                        break

        if not any(all_records.values()):
            logger.warning("Creating custom splits for personality_evd")
            all_files = list(data_path.glob("*.json")) + list(data_path.glob("*.jsonl"))
            combined = []
            for f in all_files:
                if f.name in {"train.json", "train.jsonl", "val.json", "val.jsonl", "test.json", "test.jsonl"}:
                    combined.extend(self.parse_file(f, "all"))

            if combined:
                all_records = self._build_custom_splits(combined)

        # Save to JSONL
        for split_name, records in all_records.items():
            if records:
                out_file = output_path / f"{split_name}.jsonl"
                with open(out_file, "w", encoding="utf-8") as f:
                    for rec in records:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                logger.info(f"Saved {len(records)} {split_name} records to {out_file}")

        # Save gold evidence separately
        all_records_flat = [r for recs in all_records.values() for r in recs]
        evidence_records = [r for r in all_records_flat if r.get("evidence_gold")]
        if evidence_records:
            evd_file = output_path / "evidence_gold.jsonl"
            with open(evd_file, "w", encoding="utf-8") as f:
                for rec in evidence_records:
                    gold = {"id": rec["id"], "evidence": rec["evidence_gold"]}
                    f.write(json.dumps(gold, ensure_ascii=False) + "\n")
            logger.info(f"Saved {len(evidence_records)} gold evidence records to {evd_file}")

        logger.info("Personality Evd parsing complete.")

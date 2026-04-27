"""Translate Personality-Evd raw Chinese files to English using Ollama.

Uses qwen2.5:7b (excellent for zh-en, includes Pinyin transliteration of names).

Pipeline:
  1. Load raw Chinese files from data/raw/personality_evd/Dataset/
  2. Extract all unique Chinese strings (utterances + reasoning + speaker names)
  3. Translate via Ollama with concurrent requests (default 4 workers)
  4. Persist cache to disk after every N translations (resumable)
  5. Reconstruct output structure mirroring raw, save to data/raw/personality_evd_en/Dataset/

Output structure (identical schema, English content):
  data/raw/personality_evd_en/Dataset/
    dialogue.json
    EPR-State Task/{train,valid,test}_annotation.json
    EPR-Trait Task/trait_annotation.json

Speaker names → Pinyin (e.g., 乔英子 → Qiao Yingzi), consistent across all files.
Turn line format `第N句X说：text` → `Utterance N X said: text`.

Usage:
  python scripts/translate_personality_evd.py --workers 4
  python scripts/translate_personality_evd.py --resume     # use existing cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:7b"

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "raw" / "personality_evd" / "Dataset"
OUTPUT_DIR = ROOT / "data" / "raw" / "personality_evd_en" / "Dataset"
CACHE_PATH = ROOT / "data" / "raw" / "personality_evd_en" / "_translation_cache.json"
NAME_MAP_PATH = ROOT / "data" / "raw" / "personality_evd_en" / "_speaker_name_map.json"

TURN_RE_ZH = re.compile(r"^第(?P<utt_id>\d+)句(?P<speaker>.+?)说[：:](?P<utterance>.*)$")

PROMPT_TPL = """Translate the following Chinese text to natural fluent English.
Preserve emotional tone, conversational nuance, and personality cues.
Transliterate Chinese personal names using Pinyin (e.g., 乔英子 → Qiao Yingzi).
Return ONLY the English translation, no quotes, no explanations.

Chinese: {text}
English:"""

# Lock for cache writes
_cache_lock = threading.Lock()


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def load_cache() -> dict[str, str]:
    if CACHE_PATH.exists():
        with CACHE_PATH.open() as f:
            return json.load(f)
    return {}


def persist_cache(cache: dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(cache, f, ensure_ascii=False)
    tmp.replace(CACHE_PATH)


def translate_one(text: str, model: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT_TPL.format(text=text)}],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 1024},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
            return data["message"]["content"].strip()
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt == 2:
                raise RuntimeError(f"Ollama failed after 3 retries: {e}") from e
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def build_name_map(speakers_zh: list[str], cache: dict[str, str], model: str) -> dict[str, str]:
    """Translate speaker names individually (small set, deserve high-quality individual calls)."""
    name_map = {}
    for name in speakers_zh:
        h = "name:" + sha(name)
        if h in cache:
            name_map[name] = cache[h]
            continue
        en = translate_one(name, model)
        # Sanitize: take first line, max 4 tokens (names should be short)
        en = en.splitlines()[0].strip().rstrip(".")
        if len(en.split()) > 4 or any(c in en for c in [",", ":", ";"]):
            # fallback: simple Pinyin conversion would be better; for now keep raw
            print(f"  WARN: suspicious name translation: {name} -> {en!r}", file=sys.stderr)
        cache[h] = en
        name_map[name] = en
    persist_cache(cache)
    NAME_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    NAME_MAP_PATH.write_text(json.dumps(name_map, ensure_ascii=False, indent=2))
    return name_map


def collect_strings(
    dialogues: dict, state_data: dict[str, dict], trait_data: dict
) -> tuple[set[str], set[str], set[str]]:
    """Walk all 4 raw files, collect unique utterances, reasoning, names."""
    utterances: set[str] = set()
    reasonings: set[str] = set()
    names: set[str] = set(dialogues.keys())

    # 1. Dialogue.json: turn lines
    for speaker, sdata in dialogues.items():
        for dlg_id, turns in sdata.get("dialogue", {}).items():
            for line in turns:
                m = TURN_RE_ZH.match(line.strip())
                if m:
                    utterances.add(m.group("utterance").strip())
                    names.add(m.group("speaker").strip())
                else:
                    utterances.add(line.strip())

    # 2. EPR-State annotations: nat_lang reasoning
    for split_name, split_data in state_data.items():
        for speaker, sdata in split_data.items():
            names.add(speaker)
            for dlg_id, traits in sdata.get("annotation", {}).items():
                for trait, ann in traits.items():
                    nl = (ann.get("nat_lang") or "").strip()
                    if nl:
                        reasonings.add(nl)

    # 3. EPR-Trait annotations: nat_lang
    for speaker, sdata in trait_data.items():
        names.add(speaker)
        for trait, ann in sdata.items():
            nl = (ann.get("nat_lang") or "").strip()
            if nl:
                reasonings.add(nl)

    return utterances, reasonings, names


def parallel_translate(items: list[str], cache: dict[str, str], model: str, workers: int, label: str) -> None:
    """Translate items concurrently, mutating cache. Persist every 100."""
    todo = [(i, s) for i, s in enumerate(items) if sha(s) not in cache]
    print(f"[{label}] {len(items)} unique items, {len(items) - len(todo)} cached, {len(todo)} to translate")
    if not todo:
        return

    completed = 0
    t0 = time.time()
    persist_every = 100

    def worker(item: tuple[int, str]) -> tuple[str, str]:
        idx, text = item
        try:
            en = translate_one(text, model)
            return text, en
        except Exception as e:
            print(f"  [{label}] FAIL on item {idx}: {e}", file=sys.stderr)
            return text, ""

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, item): item for item in todo}
        for fut in as_completed(futures):
            text, en = fut.result()
            if en:
                with _cache_lock:
                    cache[sha(text)] = en
            completed += 1
            if completed % 50 == 0 or completed == len(todo):
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 0.001)
                eta = (len(todo) - completed) / max(rate, 0.001)
                print(
                    f"  [{label}] {completed}/{len(todo)} ({rate:.1f}/s, ETA {eta / 60:.1f}min)",
                    flush=True,
                )
            if completed % persist_every == 0:
                with _cache_lock:
                    persist_cache(cache)
    with _cache_lock:
        persist_cache(cache)


def get(cache: dict[str, str], text: str, fallback: str = "") -> str:
    return cache.get(sha(text), fallback or text)


def rewrite_dialogues(dialogues: dict, cache: dict[str, str], name_map: dict[str, str]) -> dict:
    out = {}
    for speaker_zh, sdata in dialogues.items():
        speaker_en = name_map.get(speaker_zh, speaker_zh)
        new_dialogue = {}
        for dlg_id, turns in sdata.get("dialogue", {}).items():
            new_turns = []
            for line in turns:
                m = TURN_RE_ZH.match(line.strip())
                if m:
                    utt_id = m.group("utt_id")
                    sp_zh = m.group("speaker").strip()
                    utt_zh = m.group("utterance").strip()
                    sp_en = name_map.get(sp_zh, sp_zh)
                    utt_en = get(cache, utt_zh, utt_zh)
                    new_turns.append(f"Utterance {utt_id} {sp_en} said: {utt_en}")
                else:
                    new_turns.append(get(cache, line.strip(), line.strip()))
            new_dialogue[dlg_id] = new_turns
        out[speaker_en] = {
            "dlg_num": sdata.get("dlg_num"),
            "dialogue": new_dialogue,
        }
    return out


def rewrite_state(state_data: dict, cache: dict[str, str], name_map: dict[str, str]) -> dict:
    out = {}
    for speaker_zh, sdata in state_data.items():
        speaker_en = name_map.get(speaker_zh, speaker_zh)
        new_annotations = {}
        for dlg_id, traits in sdata.get("annotation", {}).items():
            new_traits = {}
            for trait, ann in traits.items():
                nl_zh = (ann.get("nat_lang") or "").strip()
                new_traits[trait] = {
                    **ann,
                    "nat_lang": get(cache, nl_zh, nl_zh) if nl_zh else "",
                }
            new_annotations[dlg_id] = new_traits
        out[speaker_en] = {
            "dlg_num": sdata.get("dlg_num"),
            "annotation": new_annotations,
        }
    return out


def rewrite_trait(trait_data: dict, cache: dict[str, str], name_map: dict[str, str]) -> dict:
    out = {}
    for speaker_zh, sdata in trait_data.items():
        speaker_en = name_map.get(speaker_zh, speaker_zh)
        new_traits = {}
        for trait, ann in sdata.items():
            nl_zh = (ann.get("nat_lang") or "").strip()
            new_traits[trait] = {**ann, "nat_lang": get(cache, nl_zh, nl_zh) if nl_zh else ""}
        out[speaker_en] = new_traits
    return out


def rewrite_3folds(folds_data: Any, name_map: dict[str, str]) -> Any:
    """Replace Chinese speaker names in 3_folds.json with English versions."""
    if isinstance(folds_data, dict):
        return {k: rewrite_3folds(v, name_map) for k, v in folds_data.items()}
    if isinstance(folds_data, list):
        return [name_map.get(x, x) if isinstance(x, str) else rewrite_3folds(x, name_map) for x in folds_data]
    if isinstance(folds_data, str):
        return name_map.get(folds_data, folds_data)
    return folds_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true", help="Use existing cache")
    args = ap.parse_args()

    # Load raw files
    print("[1/5] Loading raw Chinese files ...")
    dialogues = json.load(open(args.source_dir / "dialogue.json"))
    state_data = {
        "train": json.load(open(args.source_dir / "EPR-State Task" / "train_annotation.json")),
        "valid": json.load(open(args.source_dir / "EPR-State Task" / "valid_annotation.json")),
        "test": json.load(open(args.source_dir / "EPR-State Task" / "test_annotation.json")),
    }
    trait_data = json.load(open(args.source_dir / "EPR-Trait Task" / "trait_annotation.json"))
    folds_data = json.load(open(args.source_dir / "EPR-Trait Task" / "3_folds.json"))

    # Collect strings
    print("[2/5] Collecting unique strings ...")
    utterances, reasonings, names = collect_strings(dialogues, state_data, trait_data)
    print(f"  {len(utterances)} unique utterances")
    print(f"  {len(reasonings)} unique reasoning passages")
    print(f"  {len(names)} unique speaker names")

    # Load cache (resume support)
    cache = load_cache()
    print(f"[3/5] Loaded cache with {len(cache)} entries")

    # Translate names first (they're referenced by everything else)
    print("[4a/5] Translating speaker names ...")
    name_map = build_name_map(sorted(names), cache, args.model)
    print(f"  Sample: {list(name_map.items())[:5]}")

    # Translate utterances (parallel)
    print("[4b/5] Translating utterances ...")
    parallel_translate(sorted(utterances), cache, args.model, args.workers, "utt")

    # Translate reasoning passages (parallel)
    print("[4c/5] Translating reasoning passages ...")
    parallel_translate(sorted(reasonings), cache, args.model, args.workers, "rsn")

    # Rebuild output structure
    print("[5/5] Rebuilding English files ...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "EPR-State Task").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "EPR-Trait Task").mkdir(parents=True, exist_ok=True)

    en_dialogues = rewrite_dialogues(dialogues, cache, name_map)
    with open(args.output_dir / "dialogue.json", "w") as f:
        json.dump(en_dialogues, f, ensure_ascii=False, indent=2)
    print(f"  Wrote {args.output_dir / 'dialogue.json'}")

    for split_name, raw in state_data.items():
        en_state = rewrite_state(raw, cache, name_map)
        out_path = args.output_dir / "EPR-State Task" / f"{split_name}_annotation.json"
        with open(out_path, "w") as f:
            json.dump(en_state, f, ensure_ascii=False, indent=2)
        print(f"  Wrote {out_path}")

    en_trait = rewrite_trait(trait_data, cache, name_map)
    with open(args.output_dir / "EPR-Trait Task" / "trait_annotation.json", "w") as f:
        json.dump(en_trait, f, ensure_ascii=False, indent=2)
    print("  Wrote trait_annotation.json")

    en_folds = rewrite_3folds(folds_data, name_map)
    with open(args.output_dir / "EPR-Trait Task" / "3_folds.json", "w") as f:
        json.dump(en_folds, f, ensure_ascii=False, indent=2)
    print("  Wrote 3_folds.json")

    print(f"\n[DONE] English files written to {args.output_dir}")
    print(f"  Speaker name map: {NAME_MAP_PATH}")
    print(f"  Translation cache: {CACHE_PATH}")


if __name__ == "__main__":
    main()

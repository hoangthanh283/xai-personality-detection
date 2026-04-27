"""Post-process English Personality-Evd files to remove residual Chinese fragments.

Many translated nat_lang fields contain isolated Chinese words that qwen2.5:7b
left untranslated (e.g., 表现 inside otherwise English sentences). This script:
  1. Apply known Chinese→English mappings (covers ~95% of cases)
  2. Re-translate only the remaining problematic items via Ollama (small set)

Usage:
  python scripts/cleanup_chinese_residue.py [--retranslate]
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

from pypinyin import Style, pinyin

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "personality_evd_en" / "Dataset"
OLLAMA_URL = "http://localhost:11434/api/chat"

ZH_RE = re.compile(r"[一-鿿]+")

# Glossary covering top ~50 Chinese fragments observed
GLOSSARY = {
    # Most common "behavior/manifestation" — context dependent
    "表现": "behavior",
    "明显": "clearly",
    "因此": "therefore",
    "无法判断": "cannot be determined",
    "反驳": "rebut",
    "看法": "opinion",
    "指责": "blame",
    "挑剔": "picky",
    "质疑": "question",
    "波动": "fluctuation",
    "观点": "viewpoint",
    "明显体现出": "clearly exhibits",
    # Common templated UNKNOWN phrases
    "表现未显示出明显的神经质特征": "behavior did not clearly show neuroticism traits",
    "表现未能明显体现开放性的特征": "behavior did not clearly exhibit openness traits",
    "表现未显示出明显的开放性特征": "behavior did not clearly show openness traits",
    "表现未显示出明显的尽责性特征": "behavior did not clearly show conscientiousness traits",
    "表现未显示出明显的外向性特征": "behavior did not clearly show extraversion traits",
    "表现未显示出明显的宜人性特征": "behavior did not clearly show agreeableness traits",
    "表现未能明显体现尽责性的特征": "behavior did not clearly exhibit conscientiousness traits",
    "表现未能明显体现外向性的特征": "behavior did not clearly exhibit extraversion traits",
    "表现未能明显体现宜人性的特征": "behavior did not clearly exhibit agreeableness traits",
    "表现未能明显体现神经质的特征": "behavior did not clearly exhibit neuroticism traits",
    "的神经质程度": "neuroticism level",
    "的开放性程度": "openness level",
    "的尽责性程度": "conscientiousness level",
    "的外向性程度": "extraversion level",
    "的宜人性程度": "agreeableness level",
    # Trait labels (also used in `level` field)
    "开放性高": "openness high",
    "开放性低": "openness low",
    "尽责性高": "conscientiousness high",
    "尽责性低": "conscientiousness low",
    "外向性高": "extraversion high",
    "外向性低": "extraversion low",
    "宜人性高": "agreeableness high",
    "宜人性低": "agreeableness low",
    "神经质性高": "neuroticism high",
    "神经质性低": "neuroticism low",
    # Common adjectives/verbs
    "妮": "Ni",  # name suffix
    "神奇": "Shenqi",  # part of character name
    "女": "(female)",
    "男": "(male)",
}


def apply_glossary(text: str, glossary: dict[str, str]) -> str:
    """Apply find/replace from longest to shortest key (avoid partial overlap)."""
    for zh in sorted(glossary.keys(), key=len, reverse=True):
        text = text.replace(zh, glossary[zh])
    return text


def translate_via_ollama(text: str, model: str = "qwen2.5:7b") -> str:
    prompt = f"""Translate the following Chinese text to natural fluent English.
Translate ALL Chinese characters - do not leave any Chinese in the output.
Return ONLY the English translation.

Chinese: {text}
English:"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["message"]["content"].strip()


def pinyin_fallback(s: str) -> str:
    """Replace residual Chinese characters with Pinyin (fallback for names/proper nouns)."""

    def repl(match: re.Match) -> str:
        zh = match.group(0)
        py = pinyin(zh, style=Style.NORMAL)
        return " ".join(p[0].capitalize() for p in py if p)

    return ZH_RE.sub(repl, s)


def cleanup_string(s: str, retranslate: bool = False) -> str:
    if not s or not ZH_RE.search(s):
        return s
    s = apply_glossary(s, GLOSSARY)
    if not ZH_RE.search(s):
        return s
    # Fallback: pypinyin for remaining names/words
    s = pinyin_fallback(s)
    if not ZH_RE.search(s):
        return s
    if retranslate:
        try:
            return translate_via_ollama(s)
        except Exception as e:
            print(f"  retranslate failed: {e}")
    return s


def cleanup_dialogue(d: dict, retranslate: bool) -> dict:
    out = {}
    for sk, sdata in d.items():
        sk = cleanup_string(sk, retranslate=False)
        new_dlg = {}
        for dlg_id, turns in sdata.get("dialogue", {}).items():
            new_dlg[dlg_id] = [cleanup_string(t, retranslate) for t in turns]
        out[sk] = {"dlg_num": sdata.get("dlg_num"), "dialogue": new_dlg}
    return out


def cleanup_state(d: dict, retranslate: bool) -> dict:
    out = {}
    for sk, sdata in d.items():
        sk = cleanup_string(sk, retranslate=False)
        new_ann = {}
        for dlg_id, traits in sdata.get("annotation", {}).items():
            new_traits = {}
            for trait, ann in traits.items():
                new_traits[trait] = {
                    **ann,
                    "level": cleanup_string(ann.get("level", ""), retranslate),
                    "nat_lang": cleanup_string(ann.get("nat_lang", ""), retranslate),
                }
            new_ann[dlg_id] = new_traits
        out[sk] = {"dlg_num": sdata.get("dlg_num"), "annotation": new_ann}
    return out


def cleanup_trait(d: dict, retranslate: bool) -> dict:
    out = {}
    for sk, sdata in d.items():
        sk = cleanup_string(sk, retranslate=False)
        new_traits = {}
        for trait, ann in sdata.items():
            new_traits[trait] = {
                **ann,
                "level": cleanup_string(ann.get("level", ""), retranslate),
                "nat_lang": cleanup_string(ann.get("nat_lang", ""), retranslate),
            }
        out[sk] = new_traits
    return out


def count_chinese(d, count=[0]):
    if isinstance(d, str):
        if ZH_RE.search(d):
            count[0] += 1
    elif isinstance(d, dict):
        for v in d.values():
            count_chinese(v, count)
    elif isinstance(d, list):
        for v in d:
            count_chinese(v, count)
    return count[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--retranslate", action="store_true", help="After glossary, re-translate remaining items via Ollama"
    )
    args = ap.parse_args()

    files = {
        "dialogue.json": cleanup_dialogue,
        "EPR-State Task/train_annotation.json": cleanup_state,
        "EPR-State Task/valid_annotation.json": cleanup_state,
        "EPR-State Task/test_annotation.json": cleanup_state,
        "EPR-Trait Task/trait_annotation.json": cleanup_trait,
    }
    for fname, cleanup_fn in files.items():
        path = DATA_DIR / fname
        if not path.exists():
            continue
        d = json.load(open(path))
        before = count_chinese(d, count=[0])
        new_d = cleanup_fn(d, retranslate=args.retranslate)
        after = count_chinese(new_d, count=[0])
        with open(path, "w") as f:
            json.dump(new_d, f, ensure_ascii=False, indent=2)
        print(
            f"{fname}: {before} → {after} fields with Chinese ({100 * (before - after) / max(before, 1):.1f}% reduced)"
        )


if __name__ == "__main__":
    main()

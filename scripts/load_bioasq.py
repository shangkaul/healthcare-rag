# scripts/load_bioasq.py

import os
import json
import re
from glob import glob
from pathlib import Path
from tqdm import tqdm

SRC_DIR = Path("data/eval/raw")
OUT_FILE = Path("data/eval/bioasq_eval.jsonl")

def normalize_doc_id(doc_url: str) -> str:
    """Normalize PubMed links to pmid:<id>"""
    if not doc_url:
        return None
    pmid_match = re.search(r"(\d{5,9})", doc_url)
    if pmid_match:
        return f"pmid:{pmid_match.group(1)}"
    return doc_url.strip()

def load_all_files():
    files = glob(str(SRC_DIR / "*.json"))
    print(f"Found {len(files)} BioASQ files")
    return files

def parse_item(item, prefix):
    qid = item.get("id") or item.get("question_id")
    qtype = item.get("type") or item.get("questionType", "").lower()
    if qtype == "yesno":
        gold_exact = str(item.get("exact_answer", "")).lower()
    else:
        gold_exact = item.get("exact_answer")

    return {
        "id": f"{prefix}_{qid}",
        "question": item.get("body") or item.get("question", ""),
        "type": qtype,
        "gold_exact": gold_exact,
        "gold_ideal": (
            item.get("ideal_answer", [""])[0]
            if isinstance(item.get("ideal_answer"), list)
            else item.get("ideal_answer")
        ),
        "gold_documents": [normalize_doc_id(d) for d in item.get("documents", [])],
        "gold_snippets": [s.get("text", "") for s in item.get("snippets", [])],
    }

def main():
    os.makedirs(OUT_FILE.parent, exist_ok=True)
    out_f = open(OUT_FILE, "w", encoding="utf-8")

    files = load_all_files()
    count = 0

    for fpath in files:
        prefix = Path(fpath).stem.replace("_golden","").replace(".json","")
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in tqdm(data.get("questions", []), desc=prefix):
            rec = parse_item(item, prefix)
            out_f.write(json.dumps(rec) + "\n")
            count += 1

    out_f.close()
    print(f"✅ Wrote {count} records to {OUT_FILE}")

if __name__ == "__main__":
    main()

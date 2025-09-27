# scripts/check_eval_sanity.py

import json
from pathlib import Path
from collections import Counter

FILES = {
    "wiki": Path("data/corpus/wiki_corpus.jsonl"),
    "pubmed_corpus": Path("data/corpus/pubmed_corpus.jsonl"),
    "bioasq_eval": Path("data/eval/bioasq_eval.jsonl"),
    "pubmed_eval": Path("data/eval/pubmed_eval.jsonl"),
}


def quick_stats_jsonl(path, max_lines=3):
    count = 0
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            count += 1
            if i < max_lines:
                samples.append(obj)
    return count, samples


def check_bioasq(path):
    type_counter = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            type_counter[obj.get("type", "unknown")] += 1
    return type_counter


def main():
    print("=== Sanity Check Report ===\n")

    # Wiki
    n, samples = quick_stats_jsonl(FILES["wiki"])
    print(f"[Wiki Corpus] {n} entries")
    for s in samples:
        print(" sample:", {k: s[k] for k in list(s)[:3]})
    print()

    # PubMed Corpus
    n, samples = quick_stats_jsonl(FILES["pubmed_corpus"])
    print(f"[PubMed Corpus] {n} entries")
    for s in samples:
        print(" sample:", {k: s[k] for k in list(s)[:3]})
    print()

    # BioASQ Eval
    n, samples = quick_stats_jsonl(FILES["bioasq_eval"])
    type_counts = check_bioasq(FILES["bioasq_eval"])
    print(f"[BioASQ Eval] {n} questions")
    print(" types:", dict(type_counts))
    for s in samples:
        print(" sample:", {k: s[k] for k in list(s)[:5]})
    print()

    # PubMedQA Eval
    n, samples = quick_stats_jsonl(FILES["pubmed_eval"])
    print(f"[PubMedQA Eval] {n} questions")
    for s in samples:
        print(" sample:", {k: s[k] for k in list(s)[:5]})
    print()


if __name__ == "__main__":
    main()

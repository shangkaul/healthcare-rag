#!/usr/bin/env python3
"""
run: python scripts/hybrid_indexer.py --test "medicine for cold and cough" --topk 5

******Note: USES MINI EMBEDDING FOR NOW, UPDATE THE EMBEDDING FOR BETTER RESULTS.*****

Phase 3 — Indexing
Hybrid retriever: FAISS (dense) + BM25 (lexical)

Reads:
  data/corpus/*.jsonl

Writes:
  indexes/faiss.index
  indexes/passages.parquet
  indexes/bm25.pkl
  indexes/meta.json

Usage:
  # build/rebuild
  python scripts/build_index.py --rebuild --embedder mini

  # quick test query (after building)
  python scripts/build_index.py --test "ACE inhibitor side effects" --topk 5 --alpha 0.6
"""

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import List

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# -------------------------
# Paths & Config
# -------------------------
CORPUS_DIR = Path("data/corpus")
INDEX_DIR = Path("indexes")

FAISS_PATH = INDEX_DIR / "faiss.index"
PASSAGES_PATH = INDEX_DIR / "passages.parquet"
BM25_PATH = INDEX_DIR / "bm25.pkl"
META_PATH = INDEX_DIR / "meta.json"

EMBEDDERS = {
    "mini": "sentence-transformers/all-MiniLM-L6-v2",  # 384-dim, fast
    "bge":  "BAAI/bge-base-en-v1.5",                   # 768-dim, stronger
}

# -------------------------
# Helpers
# -------------------------
def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())

def read_corpus() -> pd.DataFrame:
    rows = []
    files = sorted(CORPUS_DIR.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files in {CORPUS_DIR}")

    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                text = obj.get("text") or obj.get("body") or obj.get("abstract")
                if not text:
                    continue
                pid = obj.get("id") or f"{path.stem}_{len(rows)}"
                rows.append({
                    "pid": pid,
                    "text": text,
                    "title": obj.get("title") or "",
                    "source": obj.get("source") or path.stem,
                    "chunk_index": obj.get("chunk_index", None),
                })
    df = pd.DataFrame(rows).dropna(subset=["text"])
    df = df.drop_duplicates(subset=["pid"], keep="first").reset_index(drop=True)
    # Ensure nullable int for chunk_index to be parquet-friendly
    if "chunk_index" in df.columns:
        df["chunk_index"] = df["chunk_index"].astype("Int64")
    return df

def build_embeddings(df: pd.DataFrame, model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    emb = model.encode(
        df["text"].tolist(),
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    return emb

def build_faiss(embeddings: np.ndarray) -> faiss.Index:
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine sim because embeddings are normalized
    index.add(embeddings)
    return index

def build_bm25(df: pd.DataFrame) -> BM25Okapi:
    tokenized = [tokenize(t) for t in df["text"].tolist()]
    bm25 = BM25Okapi(tokenized)
    # convenience mapping
    bm25.doc_ids = df["pid"].tolist()
    return bm25

def save_artifacts(df, index, bm25, embedder_key: str):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PASSAGES_PATH, index=False)
    faiss.write_index(index, str(FAISS_PATH))
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    meta = {
        "embedder_key": embedder_key,
        "embedder_name": EMBEDDERS[embedder_key],
        "dims": int(index.d),
        "num_docs": int(df.shape[0]),
        "files": {
            "faiss": str(FAISS_PATH),
            "bm25": str(BM25_PATH),
            "passages": str(PASSAGES_PATH),
        }
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def load_artifacts():
    df = pd.read_parquet(PASSAGES_PATH)
    index = faiss.read_index(str(FAISS_PATH))
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    meta = {}
    if META_PATH.exists():
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return df, index, bm25, meta

# -------------------------
# Search
# -------------------------
def hybrid_search(query: str, df, index, bm25, embedder_model, topk=5, alpha=0.6):
    # Dense
    qvec = embedder_model.encode([query], normalize_embeddings=True)[0].astype("float32")
    D, I = index.search(np.array([qvec]), topk)
    dense_scores = {df.iloc[i]["pid"]: float(s) for i, s in zip(I[0], D[0]) if i >= 0}

    # BM25
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    bm25_scores = {df.iloc[i]["pid"]: float(scores[i]) for i in range(len(scores))}

    # Optional: simple per-query min-max to bring scores to comparable scales
    def minmax(d: dict):
        if not d:
            return d
        vals = list(d.values())
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return {k: 0.0 for k in d}
        return {k: (v - lo) / (hi - lo) for k, v in d.items()}

    dense_n = minmax(dense_scores)
    bm25_n = minmax(bm25_scores)

    merged = {}
    for pid in set(dense_n) | set(bm25_n):
        merged[pid] = alpha * dense_n.get(pid, 0.0) + (1 - alpha) * bm25_n.get(pid, 0.0)

    ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:topk]
    results = []
    for pid, score in ranked:
        row = df.loc[df["pid"] == pid].iloc[0]
        results.append((pid, score, row["title"], row["source"], row["text"]))
    return results

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index from corpus")
    parser.add_argument("--embedder", choices=EMBEDDERS.keys(), default="mini",
                        help="Embedder to use when rebuilding or querying (if meta missing)")
    parser.add_argument("--test", type=str, help="Run a test query")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.6, help="Hybrid weight: dense vs BM25")
    args = parser.parse_args()

    if args.rebuild:
        print("📥 Reading corpus…")
        df = read_corpus()
        print(f"Loaded {len(df)} passages.")

        print(f"🔢 Building embeddings with '{EMBEDDERS[args.embedder]}'…")
        emb = build_embeddings(df, EMBEDDERS[args.embedder])

        print("📦 Building FAISS index…")
        index = build_faiss(emb)

        print("🔤 Building BM25 index…")
        bm25 = build_bm25(df)

        print("💾 Saving artifacts…")
        save_artifacts(df, index, bm25, args.embedder)

    # Load artifacts for testing/query
    df, index, bm25, meta = load_artifacts()

    # Choose embedder for querying:
    # 1) Use what the index was built with (meta), else
    # 2) fall back to CLI choice.
    embedder_key = meta.get("embedder_key", args.embedder)
    embedder_name = meta.get("embedder_name", EMBEDDERS[embedder_key])
    embedder_model = SentenceTransformer(embedder_name)

    if args.test:
        results = hybrid_search(args.test, df, index, bm25, embedder_model,
                                topk=args.topk, alpha=args.alpha)
        print(f"\n🔎 Results for query: {args.test}")
        for pid, score, title, source, text in results:
            print(f"[{pid}] ({source}) score={score:.3f} — {title}")
            print(f"  {text[:160].strip()}…\n")

if __name__ == "__main__":
    main()

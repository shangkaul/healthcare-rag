# scripts/rag_pipeline.py
# RAG: FAISS + BM25 → context → Chat Completions on Hugging Face (free serverless)
# Example:
#   python scripts/rag_pipeline.py \
#     --query "ACE inhibitor adverse reactions" \
#     --llm.backend hfchat \
#     --llm.model HuggingFaceTB/SmolLM3-3B \
#     --final_k 6 --max_new_tokens 256

from __future__ import annotations
import argparse, json, os, pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import faiss

# ---------------- .env ----------------
from dotenv import load_dotenv
ROOT_DOTENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ROOT_DOTENV, override=False)
HF_TOKEN = (
    os.getenv("HF_TOKEN")
    or os.getenv("huggingface_hub_token")
    or os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

# -------------- Paths -----------------
INDEX_DIR = Path("indexes")
FAISS_PATH = INDEX_DIR / "faiss.index"
BM25_PATH = INDEX_DIR / "bm25.pkl"
PASSAGES_PATH = INDEX_DIR / "passages.parquet"
META_PATH = INDEX_DIR / "meta.json"

@dataclass
class RetrievedDoc:
    pid: str
    text: str
    score: float
    source: str  # "faiss" | "bm25"

# -------------- Loads -----------------
def load_meta() -> Dict:
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {}

def load_passages() -> pd.DataFrame:
    df = pd.read_parquet(PASSAGES_PATH)
    assert "pid" in df.columns and "text" in df.columns, "passages.parquet must have pid and text"
    return df.reset_index(drop=True)

def build_bm25() -> Tuple[object, List[str]]:
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    doc_ids = getattr(bm25, "doc_ids", None)
    if doc_ids is None:
        df = load_passages()
        doc_ids = df["pid"].tolist()
    return bm25, list(map(str, doc_ids))

# ------------- Retrieval --------------
def bm25_search(bm25, doc_ids: List[str], query: str, k: int) -> List[RetrievedDoc]:
    toks = [t.lower() for t in query.split()]
    scores = bm25.get_scores(toks)
    idxs = np.argsort(scores)[::-1][:k]
    df = load_passages()
    pid_to_text = dict(zip(df["pid"].astype(str), df["text"].astype(str)))
    out: List[RetrievedDoc] = []
    for i in idxs:
        pid = str(doc_ids[i])
        out.append(RetrievedDoc(pid=pid, text=pid_to_text.get(pid, ""), score=float(scores[i]), source="bm25"))
    return out

def build_faiss_and_embedder(meta: Dict):
    index = faiss.read_index(str(FAISS_PATH))
    model_name = meta.get("embedder_name") or meta.get("embedder") or "sentence-transformers/all-MiniLM-L6-v2"
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(model_name)
    use_ip = isinstance(index, faiss.IndexFlatIP) or "IndexFlatIP" in type(index).__name__
    return index, embedder, use_ip

def faiss_search(index, embedder, use_ip: bool, query: str, k: int) -> List[RetrievedDoc]:
    q = embedder.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(q, k)
    D, I = D[0], I[0]
    df = load_passages()
    out: List[RetrievedDoc] = []
    for score, ridx in zip(D, I):
        if ridx < 0:
            continue
        row = df.iloc[int(ridx)]
        out.append(RetrievedDoc(pid=str(row["pid"]), text=str(row["text"]), score=float(score), source="faiss"))
    return out

# --------------- Fusion ---------------
def rrf(runs: Dict[str, List[RetrievedDoc]], final_k: int = 8, k_r: int = 60) -> List[RetrievedDoc]:
    scores: Dict[str, float] = {}
    pick: Dict[str, RetrievedDoc] = {}
    for _, docs in runs.items():
        for r, d in enumerate(docs, start=1):
            key = d.pid
            scores[key] = scores.get(key, 0.0) + 1.0 / (k_r + r)
            if key not in pick:
                pick[key] = d
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:final_k]
    return [pick[k] for k, _ in ranked]

# --------------- Prompt ----------------
PROMPT = (
    "You are a careful medical assistant. Answer using only the provided context. "
    "Cite sources as [PMID:xxxx]. If the answer can't be found, say you don't know.\n\n"
    "# Question\n{q}\n\n# Context\n{ctx}\n\n# Answer (with [PMID:xxxx] citations):\n"
)

# --------------- LLM base --------------
class LLM:
    tokenizer = None
    max_input = 2048
    reserve_for_answer = 256
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        raise NotImplementedError

class EchoLLM(LLM):
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        return "[EchoLLM]\n" + prompt

# ------- HF Chat Completions backend ----
class HFChatLLM(LLM):
    """
    Hugging Face Chat Completions API (provider='hf-inference').
    Uses HF_TOKEN from .env. Default model: HuggingFaceTB/SmolLM3-3B (chat).
    """
    def __init__(self, model_name: str, api_key: Optional[str]):
        if not api_key:
            raise ValueError("Missing HF token. Add HF_TOKEN=... (or huggingface_hub_token=...) in ../.env")
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(provider="hf-inference", api_key=api_key)
        self.model = model_name or "HuggingFaceTB/SmolLM3-3B"
        # Chat API doesn't expose a tokenizer here; keep conservative caps
        self.max_input = 2048
        self.reserve_for_answer = 256

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        # You can add a system instruction, but our prompt already includes guidance.
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            msg = completion.choices[0].message
            # Some SDKs expose .content; the snippet prints the whole object—handle both.
            content = getattr(msg, "content", None)
            return (content if isinstance(content, str) else str(msg)).strip()
        except Exception as e:
            raise RuntimeError(f"HF Chat Completions error: {e}")

# ---- Context builder (heuristic if no tokenizer) ----
def build_context_token_aware(
    docs: List[RetrievedDoc],
    tokenizer,                # None for HF chat; use char heuristic
    model_max_input: int,
    reserve_for_answer: int,
    query: str,
    prompt_template: str,
    per_doc_header_fmt: str = "[PMID:{pid}]\n",
) -> Tuple[str, List[str]]:
    if tokenizer is None:
        approx_chars_per_token = 4
        token_budget = max(64, model_max_input - reserve_for_answer)
        approx_chars = token_budget * approx_chars_per_token
        ctx_parts, cites, used = [], [], 0
        for d in docs:
            snippet = (d.text or "").strip()
            if not snippet:
                continue
            chunk = per_doc_header_fmt.format(pid=d.pid) + snippet
            take = chunk[: max(0, approx_chars - used)]
            if not take:
                break
            ctx_parts.append(take)
            cites.append(d.pid)
            used += len(take)
        return "\n\n".join(ctx_parts), cites

    def tok_len(txt: str) -> int:
        return len(tokenizer.encode(txt, add_special_tokens=False))

    placeholder = "<CTX>"
    base = prompt_template.format(q=query, ctx=placeholder)
    base_tokens = tok_len(base) - tok_len(placeholder)
    budget = max(64, model_max_input - base_tokens - reserve_for_answer)

    picked, cites, used = [], [], 0
    for d in docs:
        header = per_doc_header_fmt.format(pid=d.pid)
        full = header + (d.text or "").strip()
        need = tok_len(full)
        if used + need <= budget:
            picked.append(full); cites.append(d.pid); used += need
            continue
        remaining = budget - used
        if remaining <= 0:
            break
        lo, hi, best = 0, len(full), ""
        while lo < hi:
            mid = (lo + hi) // 2
            cand = full[:mid]
            if tok_len(cand) <= remaining:
                best = cand
                lo = mid + 1
            else:
                hi = mid
        if best:
            picked.append(best); cites.append(d.pid)
        break
    return "\n\n".join(picked), cites

# --------------- Pipeline ---------------
def run_pipeline(
    query: str,
    top_k_each: int,
    final_k: int,
    token_budget: int,        # used when tokenizer is None
    llm_backend: str,
    llm_model: Optional[str],
    use_rrf: bool,
    device: str,
    max_new_tokens: int,
):
    meta = load_meta()

    bm25, doc_ids = build_bm25()
    bm25_hits = bm25_search(bm25, doc_ids, query, k=top_k_each)

    faiss_index, embedder, use_ip = build_faiss_and_embedder(meta)
    faiss_hits = faiss_search(faiss_index, embedder, use_ip, query, k=top_k_each)

    runs = {}
    if bm25_hits: runs["bm25"] = bm25_hits
    if faiss_hits: runs["faiss"] = faiss_hits
    if not runs:
        return {"answer": "No retrievers returned results.", "citations": []}

    fused = rrf(runs, final_k=final_k) if (use_rrf and len(runs) > 1) else next(iter(runs.values()))[:final_k]

    # LLM selection
    if llm_backend == "hfchat":
        llm: LLM = HFChatLLM(model_name=llm_model or "HuggingFaceTB/SmolLM3-3B", api_key=HF_TOKEN)
    elif llm_backend == "echo":
        llm = EchoLLM()
    else:
        return {"answer": "Unsupported LLM backend. Use --llm.backend hfchat or echo.", "citations": []}

    # Build context (tokenizer-aware if available; otherwise char heuristic)
    ctx, cites = build_context_token_aware(
        fused,
        tokenizer=getattr(llm, "tokenizer", None),
        model_max_input=getattr(llm, "max_input", token_budget),
        reserve_for_answer=getattr(llm, "reserve_for_answer", 256),
        query=query,
        prompt_template=PROMPT,
    )
    prompt = PROMPT.format(q=query, ctx=ctx)

    answer = llm.generate(prompt, max_tokens=max_new_tokens)
    if cites and not any(f"[PMID:{pid}]" in answer for pid in cites):
        answer = answer.rstrip() + "\n\n" + "  ".join(f"[PMID:{pid}]" for pid in cites[:3])

    return {
        "query": query,
        "answer": answer,
        "citations": [{"pmid": pid} for pid in cites],
        "debug": {
            "bm25_top": [asdict(d) for d in bm25_hits[:min(len(bm25_hits), 5)]],
            "faiss_top": [asdict(d) for d in faiss_hits[:min(len(faiss_hits), 5)]],
        }
    }

# ---------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser(description="RAG: FAISS + BM25 → context → HF Chat Completions")
    ap.add_argument("--query", required=True)
    ap.add_argument("--top_k_each", type=int, default=8)
    ap.add_argument("--final_k", type=int, default=8)
    ap.add_argument("--token_budget", type=int, default=1200, help="Used when no tokenizer is available (hfchat).")
    ap.add_argument("--no_rrf", action="store_true")

    ap.add_argument("--llm.backend", dest="llm_backend", choices=["hfchat","echo"], default="hfchat")
    ap.add_argument("--llm.model", dest="llm_model", help="Chat model on HF Inference API, e.g. HuggingFaceTB/SmolLM3-3B")
    ap.add_argument("--device", choices=["cpu","mps"], default="cpu", help="(ignored for hfchat)")
    ap.add_argument("--max_new_tokens", type=int, default=256)

    args = ap.parse_args()

    out = run_pipeline(
        query=args.query,
        top_k_each=args.top_k_each,
        final_k=args.final_k,
        token_budget=args.token_budget,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        use_rrf=not args.no_rrf,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

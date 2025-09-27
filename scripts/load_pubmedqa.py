import os
import json
from typing import Any, List, Optional
from tqdm import tqdm
from datasets import load_dataset

# --- NLTK sentence tokenizer ---
import nltk
try:
    nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize
except Exception:
    nltk.download("punkt", quiet=True)
    from nltk.tokenize import sent_tokenize

# ----------------------------
# Robust context extraction
# ----------------------------
def join_list_safely(lst: List[Any]) -> str:
    """Join a list into text. If it's a list of single characters, join without spaces."""
    if not lst:
        return ""
    # If most elements are 1-char strings, treat it as characters → join without spaces
    str_elems = [e for e in lst if isinstance(e, str)]
    if str_elems and all(len(e) == 1 for e in str_elems) and len(str_elems) >= max(3, int(0.8 * len(lst))):
        return "".join(str_elems)
    # Otherwise join stringified elements with space
    return " ".join(str(e) for e in lst)

def extract_context_strict(item: dict) -> Optional[str]:
    """
    Return a clean string context or None.
    Only trust context-like fields. Do NOT pull in labels or unrelated keys.
    """
    if "context" in item:
        ctx = item["context"]
    elif "contexts" in item:           # some variants
        ctx = item["contexts"]
    elif "abstract" in item:           # rare variants
        ctx = item["abstract"]
    elif "passage" in item:            # rare variants
        ctx = item["passage"]
    else:
        return None

    # Normalize to string
    if ctx is None:
        return None
    if isinstance(ctx, str):
        text = ctx
    elif isinstance(ctx, list):
        text = join_list_safely(ctx)
    elif isinstance(ctx, dict):
        # prefer obvious text-like subkeys, avoid labels
        for k in ("text", "abstract", "context", "contexts", "passage"):
            if k in ctx and ctx[k]:
                v = ctx[k]
                if isinstance(v, str):
                    text = v
                elif isinstance(v, list):
                    text = join_list_safely(v)
                else:
                    text = str(v)
                break
        else:
            return None
    else:
        text = str(ctx)

    text = text.strip()
    # Guard: sometimes weird char-spaced tokens sneak in; fix common ones
    for spaced, normal in [("y e s", "yes"), ("n o", "no"), ("m a y b e", "maybe")]:
        text = text.replace(spaced, normal)
    return text or None

def chunk_text(text, chunk_size=800, overlap=150):
    """Sentence-aware chunking with word-count approx."""
    sentences = [s.strip() for s in sent_tokenize(text) if s and s.strip()]
    chunks, current, length = [], [], 0
    for sent in sentences:
        words = sent.split()
        if length + len(words) > chunk_size and current:
            chunks.append(" ".join(current))
            current = current[-overlap:]
            length = len(current)
        current.extend(words)
        length += len(words)
    if current:
        chunks.append(" ".join(current))
    # filter tiny leftovers
    return [c for c in chunks if len(c.split()) > 50]

# ----------------------------
# Paths
# ----------------------------
os.makedirs("data", exist_ok=True)
corpus_path = os.path.join("data", "pubmed_corpus.jsonl")
eval_path   = os.path.join("data", "pubmed_eval.jsonl")

print("📥 Downloading PubMedQA…")
pqa_labeled   = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
pqa_unlabeled = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled")

# ----------------------------
# Build CORPUS
# ----------------------------
print("📝 Building retrieval corpus…")
skipped_labeled = skipped_unlab = written = 0

with open(corpus_path, "w", encoding="utf-8") as f:
    # labeled contexts
    for i, item in tqdm(enumerate(pqa_labeled["train"]), total=len(pqa_labeled["train"]), desc="Labeled corpus"):
        ctx = extract_context_strict(item)
        if not ctx:
            skipped_labeled += 1
            continue
        for j, chunk in enumerate(chunk_text(ctx)):
            f.write(json.dumps({
                "id": f"pqa_labeled_{i}_{j}",
                "title": item.get("question", f"pqa_labeled_{i}"),
                "source": "pubmedqa_labeled",
                "chunk_index": j,
                "text": chunk
            }, ensure_ascii=False) + "\n")
            written += 1

    # unlabeled contexts (optional but recommended to make retrieval realistic)
    for i, item in tqdm(enumerate(pqa_unlabeled["train"]), total=len(pqa_unlabeled["train"]), desc="Unlabeled corpus"):
        ctx = extract_context_strict(item)
        if not ctx:
            skipped_unlab += 1
            continue
        for j, chunk in enumerate(chunk_text(ctx)):
            f.write(json.dumps({
                "id": f"pqa_unlabeled_{i}_{j}",
                "title": item.get("question", f"pqa_unlabeled_{i}"),
                "source": "pubmedqa_unlabeled",
                "chunk_index": j,
                "text": chunk
            }, ensure_ascii=False) + "\n")
            written += 1

print(f"✅ Corpus → {corpus_path}")
print(f"   Chunks written: {written} | Skipped labeled: {skipped_labeled} | Skipped unlabeled: {skipped_unlab}")

# ----------------------------
# Build EVAL (labeled only)
# ----------------------------
print("🧪 Building evaluation set…")
with open(eval_path, "w", encoding="utf-8") as f:
    for i, item in enumerate(pqa_labeled["train"]):
        ctx = extract_context_strict(item) or ""
        f.write(json.dumps({
            "id": f"pqa_eval_{i}",
            "question": item.get("question", ""),
            "context": ctx,
            "answer": item.get("final_decision", ""),  # yes / no / maybe
            "long_answer": item.get("long_answer", "")
        }, ensure_ascii=False) + "\n")
print(f"✅ Eval → {eval_path}")

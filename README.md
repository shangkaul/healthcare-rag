# Healthcare RAG Q&A (Hybrid + One-Shot)

A biomedical question-answering system built with **Retrieval-Augmented Generation (RAG)**, designed to provide cited, clinically-grounded answers from PubMed abstracts and curated medical Wikipedia articles.  
Evaluated on **BioASQ** (golden enriched sets) and **PubMedQA** (labeled), this repo serves as the reproducible backbone for a short research paper on healthcare RAG.

---

## ✨ Features

- **Hybrid retrieval**: Dense embeddings (FAISS) + BM25 lexical + optional reranker.
- **Biomedical grounding**: Only PubMed abstracts + curated Wikipedia medical pages.
- **One-shot prompting**: Single exemplar per answer type improves calibration.
- **Citation enforcement**: All answers must include at least one PMID/Wiki citation.
- **Evaluation harness**: Benchmarked on BioASQ (yes/no, factoid, list, summary) and PubMedQA (yes/no/maybe).
- **Observability**: Latency, token usage, retrieval recall, and citation coverage logged.
- **Guardrails**: Abstains if no supporting evidence is retrieved.

---

## 📂 Repo Layout

```
data/
  corpus/                   # retrieval base
    wiki_corpus.jsonl
    pubmed_corpus.jsonl
  eval/                     # evaluation only (never indexed)
    bioasq_eval.jsonl
    pubmed_eval.jsonl
raw/                        # raw downloads (BioASQ golden sets, PubMedQA mirrors)

indexes/
  faiss.index
  passages.parquet
  bm25.lexicon

rag/
  pipeline.py               # retrieval → rerank → context → generation → citation

scripts/
  wiki_load.py              # build wiki corpus
  load_pubmedqa.py          # build pubmed corpus + eval
  load_bioasq.py            # merge golden files → bioasq_eval.jsonl
  build_index.py            # embeddings + FAISS + BM25

eval/
  run_eval.py               # BioASQ + PubMedQA metrics

ui/                         # optional demo (Next.js or FastAPI endpoint)
```

---

## 🚀 Quickstart

### 0. Install dependencies
```bash
pip install -r requirements.txt
# or minimal
pip install faiss-cpu rank-bm25 sentence-transformers datasets wikipedia-api nltk tqdm
```

### 1. Prepare corpora
```bash
python scripts/wiki_load.py
python scripts/load_pubmedqa.py
```

### 2. Normalize BioASQ
Download BioASQ golden sets into `raw/bioasq/`  
```bash
python scripts/load_bioasq.py   # writes data/eval/bioasq_eval.jsonl
```

### 3. Build index
```bash
python scripts/build_index.py
```

### 4. Run evaluation
```bash
python eval/run_eval.py --dataset bioasq
python eval/run_eval.py --dataset pubmedqa
```

---

## 📊 Metrics

Baseline targets to aim for:

- **BioASQ retrieval**: Recall@20 ≥ 0.65, MRR@20 ≥ 0.45 (with reranker).  
- **BioASQ answers**:  
  - Yes/No Accuracy ≥ 0.75  
  - Factoid/List F1 ≥ 0.45  
  - Summary ROUGE-L ≥ 0.40  
- **PubMedQA**: Accuracy ≥ 0.70 with RAG (vs ~0.6 zero-shot baseline).

---

## 📑 Research Paper Context

This system underpins a short research paper on **healthcare RAG evaluation**.

**Research questions:**
1. Does hybrid retrieval (dense+BM25+reranker) improve Recall@k/MRR over single methods?  
2. Does one-shot prompting improve answer quality vs zero-shot?  
3. Do enforced citations reduce hallucination rate without hurting accuracy?  
4. How much do Wikipedia medical articles help compared to PubMed-only retrieval?

**Paper deliverables:**
- Retrieval + generation benchmarks on BioASQ and PubMedQA.  
- Ablations: dense-only, BM25-only, hybrid; reranker on/off; zero-shot vs one-shot.  
- Safety metrics: citation coverage, abstain rate.  
- Error analysis + case studies.  

---

## ⚠️ Disclaimer

This project is for **research purposes only**.  
It is **not a medical device** and **does not provide medical advice**.  
Always consult qualified healthcare professionals and original primary sources.

---

## 📅 Roadmap

- [ ] Add self-query expansion for recall boost  
- [ ] Auto few-shot selection from BioASQ exemplars  
- [ ] Answer verifier LLM pass (check ≥2 citations)  
- [ ] Temporal biasing (prefer recent PubMed abstracts)  
- [ ] Lightweight Next.js UI with citations and streaming  

---

## 📜 License

MIT (research use encouraged; no clinical use).

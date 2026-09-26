# Phase 3 Baseline Inventory

This document freezes and catalogs the exact state, configuration, parameters, and algorithms of the Query Reformulation Engine prior to any Phase 3 investigation or modification.

---

## 1. Frozen Baseline Retrieval Metrics (Phase 2)

| Variant | Recall@5 | Recall@10 | MRR | NDCG@5 | NDCG@10 |
|:---|---:|---:|---:|---:|---:|
| **Original + Vector** | 0.5000 | 0.6400 | 0.4754 | 0.3756 | 0.4238 |
| **Original + BM25** | 0.1000 | 0.1000 | 0.1540 | 0.1000 | 0.1000 |
| **Original + Hybrid** | 0.5100 | 0.6400 | 0.5177 | 0.4012 | 0.4448 |
| **Reformulation + Hybrid** | 0.5400 | 0.6800 | 0.5681 | 0.4288 | 0.4701 |
| **Full Pipeline (Phase 2)** | 0.5400 | 0.6900 | 0.5596 | 0.4234 | 0.4726 |

*Baseline evaluation artifact locations:*
- `evaluation/results/evaluation_results.json`
- `evaluation/results/evaluation_report.md`

---

## 2. Component Inventory

### A. Query Validation ([`reformulation/validator.py`](file:///C:/Users/dell/OneDrive/Desktop/Query_Engine/Query_Reformulation_Engine/reformulation/validator.py))
- **Function**: `validate_query(query: str) -> None`
- **Behavior**: Enforces non-empty string, min length 3 characters, max length 500 characters, rejects control characters and non-printable characters.

### B. Query Normalization ([`reformulation/normalizer.py`](file:///C:/Users/dell/OneDrive/Desktop/Query_Engine/Query_Reformulation_Engine/reformulation/normalizer.py))
- **Function**: `normalize_query(query: str) -> str`
- **Behavior**:
  - Acronym protection (e.g. `IEEE`, `IIIT`) before repeated-letter reduction.
  - Typo correction dictionary (e.g. `attendence` -> `attendance`, `allotmnt` -> `allotment`, `revalution` -> `re-evaluation`, `reqmnt` -> `requirement`, `withdrawl` -> `withdrawal`).
  - Abbreviation expansion (e.g. `sem` -> `semester`, `dept` -> `department`, `prof` -> `professor`, `univ` -> `university`, `min` -> `minimum`, `max` -> `maximum`).
  - Course code normalization (e.g. `cse301` / `Cse301` -> `CSE301`).
  - Sentence-case capitalization.

### C. Query Reformulation ([`reformulation/reformulator.py`](file:///C:/Users/dell/OneDrive/Desktop/Query_Engine/Query_Reformulation_Engine/reformulation/reformulator.py))
- **Class**: `QueryReformulator`
- **Model**: Gemini via `LLMClient` (or cached reformulations in `evaluation/cache/reformulations.json`).
- **Default Variant Count**: `max_variants = 5`.
- **Prompt**: Requests concise search-oriented alternatives preserving entities, acronyms, and technical terms; returns JSON format `{"reformulations": [...]}`.

### D. Semantic & Lexical Filtering ([`reformulation/semantic_similarity.py`](file:///C:/Users/dell/OneDrive/Desktop/Query_Engine/Query_Reformulation_Engine/reformulation/semantic_similarity.py))
- **Class**: `SemanticSimilarity`
- **Similarity Threshold**: `threshold = 0.85` (cosine similarity on `BAAI/bge-small-en-v1.5` embeddings).
- **Ranking / Tie-Breaking Key**: `(is_original, len(tokens), informative_count, -total_words, cand.lower())`.
- **Deduplication / Pruning Rule**: Candidate $C$ is rejected if for any accepted representative $R$:
  $$\operatorname{CosineSim}(E_C, E_R) \ge 0.85 \quad \text{AND} \quad |T_C \setminus T_R| == 0$$
- **Presentation Ordering**: Caller input sequence preserved with normalized original query at index 0.

### E. Vector Retrieval ([`retrieval/vector_retriever.py`](file:///C:/Users/dell/OneDrive/Desktop/Query_Engine/Query_Reformulation_Engine/retrieval/vector_retriever.py))
- **Class**: `VectorRetriever`
- **Encoder**: `BAAI/bge-small-en-v1.5` (384 dimensions) via `EmbeddingEncoder.embed_query()`.
- **Backend RPC**: `match_chunks_vector(query_embedding, match_threshold=0.0, match_count=20, filter_domain=None)`.
- **Score**: Cosine similarity float $[0.0, 1.0]$.

### F. Keyword Retrieval ([`retrieval/keyword_retriever.py`](file:///C:/Users/dell/OneDrive/Desktop/Query_Engine/Query_Reformulation_Engine/retrieval/keyword_retriever.py))
- **Class**: `KeywordRetriever`
- **Backend RPC**: `match_chunks_keyword(query_text, match_count=20, filter_domain=None)`.
- **Score**: PostgreSQL `ts_rank_cd(tsv, websearch_to_tsquery('english', query_text))`.

### G. Hybrid Single-Query Retrieval ([`retrieval/hybrid_retriever.py`](file:///C:/Users/dell/OneDrive/Desktop/Query_Engine/Query_Reformulation_Engine/retrieval/hybrid_retriever.py))
- **Class**: `HybridRetriever`
- **Fusion Formula**: Reciprocal Rank Fusion:
  $$\operatorname{RRF}(d) = \frac{\mathbb{I}(d \in V)}{k + \operatorname{rank}_V(d)} + \frac{\mathbb{I}(d \in K)}{k + \operatorname{rank}_K(d)}$$
  where $k = 60$.
- **Sorting**: `(-rrf_score, chunk_id)` for deterministic tie-breaking.

### H. Search Pipeline ([`reformulation/search_pipeline.py`](file:///C:/Users/dell/OneDrive/Desktop/Query_Engine/Query_Reformulation_Engine/reformulation/search_pipeline.py))
- **Class**: `SearchPipeline`
- **Multi-Query Aggregation**: For each filtered candidate query $q_j$, calls `HybridRetriever.retrieve()` (yielding ranked list $L_j$), and accumulates multi-query RRF scores:
  $$\operatorname{FinalScore}(d) = \sum_{j=1}^{|Q_{\text{filtered}}|} \frac{\mathbb{I}(d \in L_j)}{k + \operatorname{rank}_{L_j}(d)}$$
- **Top-K Parameters**: `vector_match_count = 20`, `keyword_match_count = 20`, `k = 60`, `final_k = 20`.

---

## 3. Dataset & Indexing Inventory

- **Corpus**: 157 documents in `Data/documents/`.
- **Chunks**: 696 chunks in `Data/chunks/chunks.json`.
- **Golden Evaluation Dataset**: 50 queries across 10 categories in `evaluation/golden_queries.json`.
- **Ground Truth**: 100 relevant chunk annotations.

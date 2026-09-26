# Retrieval Quality Evaluation Framework

This module provides the golden evaluation dataset, metric calculation algorithms, and evaluation orchestration framework for the Query Reformulation Engine.

---

## 1. Research Question

> **Does query normalization, LLM reformulation, semantic/lexical filtering, hybrid retrieval, and RRF improve retrieval quality compared with simpler retrieval baselines?**

This framework allows comparing retrieval pipeline variants under identical experimental controls using fixed ground-truth chunk-level labels across 696 indexed academic document chunks.

---

## 2. Benchmark Architecture

### Golden Dataset (`evaluation/golden_queries.json`)
The dataset contains **50 curated evaluation queries** balanced across 10 distinct search categories with **100 chunk-level ground-truth relevance annotations**:

1. **Keyword Queries** (5 queries): Simple noun phrases (`semester fees`, `hostel allotment`, `CSE301 prerequisites`).
2. **Natural Language Questions** (5 queries): Full conversational interrogatives (`Can I get a refund if I withdraw from the semester?`).
3. **Abbreviations** (5 queries): Queries containing common campus abbreviations (`sem fees`, `dept regulations`, `univ admission`).
4. **Noisy / Typo Queries** (5 queries): Common user misspellings (`attendence shortage`, `hostel allotmnt process`, `revalution fee`).
5. **Course Codes** (5 queries): Course-specific code queries (`cse301 syllabus`, `ECE201 course outline`).
6. **Numeric Constraints** (5 queries): Queries with percentages, limits, fines, and credits (`attendance below 75 percent`, `minimum CGPA 6.0`).
7. **Policy Queries** (5 queries): Institutional governance and regulations (`semester withdrawal refund policy`, `zero tolerance anti ragging policy`).
8. **Eligibility Queries** (5 queries): Qualification requirements (`who is eligible for chancellor merit scholarship`, `campus placement eligibility`).
9. **Procedure Queries** (5 queries): Step-by-step institutional workflows (`how to apply for bonafide certificate`, `lost student ID replacement`).
10. **Lexical Diversity Queries** (5 queries): Queries where specific jargon or technical terminology is critical (`LibNet OPAC book reservation`, `802.1X PEAP MSCHAPv2`).

---

## 3. Evaluated Variants

The framework compares 5 core variants:

| Variant Key | Name | Architecture |
|---|---|---|
| `baseline_vector` | Original + Vector | $Q \rightarrow \text{Dense Vector} \rightarrow \text{Cosine Match}$ |
| `baseline_keyword` | Original + BM25 | $Q \rightarrow \text{Full-Text Search (ts\_rank\_cd)}$ |
| `baseline_hybrid` | Original + Hybrid | $Q \rightarrow (\text{Vector} + \text{BM25}) \rightarrow \text{RRF}$ |
| `reformulation_hybrid` | Reformulation + Hybrid | $Q + \text{Variants} \rightarrow (\text{Vector} + \text{BM25}) \rightarrow \text{RRF}$ (No normalizer / filter) |
| `full_pipeline` | Full Pipeline (Ours) | $Q \rightarrow \text{Validate} \rightarrow \text{Normalize} \rightarrow \text{Reformulate} \rightarrow \text{Semantic Filter} \rightarrow \text{Hybrid} \rightarrow \text{RRF}$ |

---

## 4. Standard Metrics

1. **Recall@5 & Recall@10**: Proportion of ground-truth relevant chunks retrieved within top 5 (and top 10).
$$\text{Recall}@K = \frac{|\text{top\_k(retrieved)} \cap \text{relevant}|}{|\text{relevant}|}$$

2. **MRR (Mean Reciprocal Rank)**:
$$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_1(q)}$$

3. **NDCG@5 & NDCG@10 (Normalized Discounted Cumulative Gain)**:
$$\text{DCG}@K = \sum_{i=1}^{\min(K, |R|)} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \quad \text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$

---

## 5. Usage

### Run Quick 5-Query Sample Test
```bash
py -3 evaluation/run_evaluation.py --sample-size 5
```

### Run Full 50-Query Benchmark
```bash
py -3 evaluation/run_evaluation.py
```

### Outputs
- Machine-readable JSON: `evaluation/results/evaluation_results.json`
- Human-readable Markdown: `evaluation/results/evaluation_report.md`

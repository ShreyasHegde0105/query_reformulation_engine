# Retrieval Quality Evaluation Report

**Run Timestamp:** 2026-09-26T02:56:21.069153+00:00
**Corpus Size:** 696 document chunks
**Embedding Model:** BAAI/bge-small-en-v1.5
**RRF Damping (k):** 60
**Total Evaluation Queries:** 50

## 1. Overall Variant Benchmark

| Variant | Recall@5 | Recall@10 | MRR | NDCG@5 | NDCG@10 |
|:---|---:|---:|---:|---:|---:|
| Original + Vector | 0.5000 | 0.6400 | 0.4754 | 0.3756 | 0.4238 |
| Original + BM25 | 0.1000 | 0.1000 | 0.1540 | 0.1000 | 0.1000 |
| Original + Hybrid | 0.5100 | 0.6400 | 0.5177 | 0.4012 | 0.4448 |
| Reformulation + Hybrid | 0.5400 | 0.6800 | 0.5694 | 0.4288 | 0.4701 |
| Full Pipeline (Ours) | 0.5400 | 0.6900 | 0.5615 | 0.4234 | 0.4726 |

## 2. Category-Specific Breakdown

### Category: `abbreviation`

| Variant | Recall@5 | Recall@10 | MRR | NDCG@5 |
|:---|---:|---:|---:|---:|
| Original + Vector | 0.4000 | 0.4000 | 0.1844 | 0.2474 |
| Original + BM25 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Original + Hybrid | 0.4000 | 0.4000 | 0.1844 | 0.2474 |
| Reformulation + Hybrid | 0.6000 | 0.7000 | 0.3733 | 0.4254 |
| Full Pipeline (Ours) | 0.5000 | 0.7000 | 0.3733 | 0.3645 |

### Category: `course_code`

| Variant | Recall@5 | Recall@10 | MRR | NDCG@5 |
|:---|---:|---:|---:|---:|
| Original + Vector | 0.6000 | 0.7000 | 0.6400 | 0.4965 |
| Original + BM25 | 0.1000 | 0.1000 | 0.2000 | 0.1574 |
| Original + Hybrid | 0.6000 | 0.7000 | 0.6400 | 0.5297 |
| Reformulation + Hybrid | 0.7000 | 0.7000 | 0.8154 | 0.6372 |
| Full Pipeline (Ours) | 0.7000 | 0.7000 | 0.8154 | 0.6194 |

### Category: `eligibility`

| Variant | Recall@5 | Recall@10 | MRR | NDCG@5 |
|:---|---:|---:|---:|---:|
| Original + Vector | 0.6000 | 0.6000 | 0.5667 | 0.5035 |
| Original + BM25 | 0.4000 | 0.4000 | 0.7000 | 0.5149 |
| Original + Hybrid | 0.6000 | 0.6000 | 0.8667 | 0.6446 |
| Reformulation + Hybrid | 0.5000 | 0.6000 | 0.6900 | 0.5110 |
| Full Pipeline (Ours) | 0.5000 | 0.6000 | 0.6900 | 0.5110 |

### Category: `keyword`

| Variant | Recall@5 | Recall@10 | MRR | NDCG@5 |
|:---|---:|---:|---:|---:|
| Original + Vector | 0.2000 | 0.5000 | 0.2944 | 0.1353 |
| Original + BM25 | 0.3000 | 0.3000 | 0.2400 | 0.1929 |
| Original + Hybrid | 0.3000 | 0.5000 | 0.4444 | 0.2675 |
| Reformulation + Hybrid | 0.4000 | 0.6000 | 0.5356 | 0.3284 |
| Full Pipeline (Ours) | 0.4000 | 0.7000 | 0.4382 | 0.2703 |

### Category: `lexical_diversity`

| Variant | Recall@5 | Recall@10 | MRR | NDCG@5 |
|:---|---:|---:|---:|---:|
| Original + Vector | 0.6000 | 0.9000 | 0.4733 | 0.4460 |
| Original + BM25 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Original + Hybrid | 0.6000 | 0.9000 | 0.4733 | 0.4460 |
| Reformulation + Hybrid | 0.5000 | 0.7000 | 0.4810 | 0.3668 |
| Full Pipeline (Ours) | 0.5000 | 0.7000 | 0.4821 | 0.3668 |

### Category: `natural_language`

| Variant | Recall@5 | Recall@10 | MRR | NDCG@5 |
|:---|---:|---:|---:|---:|
| Original + Vector | 0.5000 | 0.7000 | 0.3983 | 0.2321 |
| Original + BM25 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Original + Hybrid | 0.5000 | 0.7000 | 0.3983 | 0.2321 |
| Reformulation + Hybrid | 0.6000 | 0.7000 | 0.3917 | 0.3718 |
| Full Pipeline (Ours) | 0.6000 | 0.7000 | 0.3917 | 0.3718 |

### Category: `numeric`

| Variant | Recall@5 | Recall@10 | MRR | NDCG@5 |
|:---|---:|---:|---:|---:|
| Original + Vector | 0.5000 | 0.6000 | 0.7000 | 0.4526 |
| Original + BM25 | 0.1000 | 0.1000 | 0.2000 | 0.0675 |
| Original + Hybrid | 0.5000 | 0.6000 | 0.7000 | 0.4526 |
| Reformulation + Hybrid | 0.5000 | 0.6000 | 0.8118 | 0.5176 |
| Full Pipeline (Ours) | 0.5000 | 0.6000 | 0.8118 | 0.5176 |

### Category: `policy`

| Variant | Recall@5 | Recall@10 | MRR | NDCG@5 |
|:---|---:|---:|---:|---:|
| Original + Vector | 0.7000 | 1.0000 | 0.5167 | 0.5010 |
| Original + BM25 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Original + Hybrid | 0.7000 | 1.0000 | 0.5167 | 0.5010 |
| Reformulation + Hybrid | 0.8000 | 0.9000 | 0.6067 | 0.5150 |
| Full Pipeline (Ours) | 0.8000 | 0.9000 | 0.6167 | 0.5386 |

### Category: `procedure`

| Variant | Recall@5 | Recall@10 | MRR | NDCG@5 |
|:---|---:|---:|---:|---:|
| Original + Vector | 0.7000 | 0.8000 | 0.7667 | 0.5556 |
| Original + BM25 | 0.1000 | 0.1000 | 0.2000 | 0.0675 |
| Original + Hybrid | 0.7000 | 0.8000 | 0.7400 | 0.5045 |
| Reformulation + Hybrid | 0.7000 | 0.9000 | 0.8500 | 0.5363 |
| Full Pipeline (Ours) | 0.7000 | 0.9000 | 0.8500 | 0.5695 |

### Category: `typo`

| Variant | Recall@5 | Recall@10 | MRR | NDCG@5 |
|:---|---:|---:|---:|---:|
| Original + Vector | 0.2000 | 0.2000 | 0.2133 | 0.1865 |
| Original + BM25 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Original + Hybrid | 0.2000 | 0.2000 | 0.2133 | 0.1865 |
| Reformulation + Hybrid | 0.1000 | 0.4000 | 0.1389 | 0.0787 |
| Full Pipeline (Ours) | 0.2000 | 0.4000 | 0.1456 | 0.1048 |

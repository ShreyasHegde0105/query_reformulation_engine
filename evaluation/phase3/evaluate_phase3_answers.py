"""Phase 3 — Answer Quality & Retrieval Evaluation Engine.

Evaluates System A, Configuration R0, and System C across:
1. Retrieval & Context Metrics:
   - Recall@5, Recall@10, MRR@20, NDCG@10
   - Document Coverage, Full Document Coverage, Chunk Coverage, Citation Precision
2. Answer Quality Metrics:
   - Factual Correctness (0-4)
   - Completeness (0-4)
   - Citation Correctness (0-4)
   - Groundedness Ratio & Unsupported Claim Rate
   - Multi-Document Reasoning (41 multi-doc queries)
   - Abstention Quality (6 unanswerable queries)
3. Paired Statistical Analysis:
   - Bootstrap 95% Confidence Intervals (10,000 iterations)
   - Wilcoxon signed-rank tests
   - Paired t-tests
   - Win / Loss / Tie distributions
   - Stratified breakdown across 10 query categories

Generates:
- phase3_answer_metrics.json
- phase3_retrieval_metrics.json
- phase3_per_query_comparison.json
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from evaluation.metrics import (
    calculate_recall_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_all_metrics,
)

REF_ANSWERS_PATH = os.path.join(ROOT, "evaluation", "phase3", "phase3_reference_answers.json")
HELDOUT_PATH = os.path.join(ROOT, "evaluation", "heldout_queries.json")
SYS_A_PATH = os.path.join(ROOT, "evaluation", "phase3", "phase3_system_a_answers.json")
SYS_C_PATH = os.path.join(ROOT, "evaluation", "phase3", "phase3_system_c_answers.json")
ABLATION_PATH = os.path.join(ROOT, "evaluation", "phase3", "phase3_reformulation_ablation_answers.json")

OUT_ANSWER_METRICS = os.path.join(ROOT, "evaluation", "phase3", "phase3_answer_metrics.json")
OUT_RETRIEVAL_METRICS = os.path.join(ROOT, "evaluation", "phase3", "phase3_retrieval_metrics.json")
OUT_PER_QUERY = os.path.join(ROOT, "evaluation", "phase3", "phase3_per_query_comparison.json")


# -----------------------------------------------------------------------------
# Statistical Helper Functions
# -----------------------------------------------------------------------------

def paired_bootstrap_ci(
    diffs: List[float],
    n_bootstraps: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute 95% bootstrap confidence interval for mean differences."""
    if not diffs:
        return (0.0, 0.0)
    arr = np.array(diffs, dtype=np.float64)
    if np.all(arr == 0):
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstraps, dtype=np.float64)
    n = len(arr)
    for i in range(n_bootstraps):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    low = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    return (round(low, 4), round(high, 4))


def compute_wilcoxon(diffs: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Compute Wilcoxon signed-rank test statistic and p-value."""
    from scipy import stats
    non_zero = [d for d in diffs if abs(d) > 1e-9]
    if len(non_zero) < 5:
        return (None, None)
    try:
        res = stats.wilcoxon(non_zero, zero_method="wilcox", alternative="two-sided")
        return (round(float(res.statistic), 4), round(float(res.pvalue), 6))
    except Exception:
        return (None, None)


def compute_paired_ttest(diffs: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Compute paired Student's t-test statistic and p-value."""
    from scipy import stats
    if len(diffs) < 2 or np.all(np.array(diffs) == 0):
        return (None, None)
    try:
        res = stats.ttest_1samp(diffs, popmean=0.0)
        return (round(float(res.statistic), 4), round(float(res.pvalue), 6))
    except Exception:
        return (None, None)


# -----------------------------------------------------------------------------
# Factual Evaluation & Groundedness Scoring
# -----------------------------------------------------------------------------

def evaluate_single_answer(
    answer_record: dict,
    ref_item: dict,
    context_text: str,
) -> Dict[str, Any]:
    """Evaluate an answer across correctness, completeness, citations, groundedness, and abstention."""
    is_answerable = ref_item["answerable"]
    ans_payload = answer_record.get("answer", {})
    ans_text = (ans_payload.get("text") or "").strip()
    is_claimed_answerable = ans_payload.get("answerable", False)
    citations = ans_payload.get("citations", [])
    cited_docs = ans_payload.get("cited_documents", [])

    gold_chunks = set(ref_item.get("supporting_chunk_ids", []))
    required_docs = set(ref_item.get("required_document_ids", []))
    essential_facts = ref_item.get("essential_facts", [])

    # Case 1: Unanswerable query
    if not is_answerable:
        # Correct behavior: system states unanswerable / refused
        correctly_abstained = (not is_claimed_answerable) or (
            "not answer" in ans_text.lower() or "do not contain" in ans_text.lower()
        )
        if correctly_abstained:
            return {
                "factual_correctness": 4.0,
                "completeness": 4.0,
                "citation_correctness": 4.0,
                "groundedness_ratio": 1.0,
                "unsupported_claim_rate": 0.0,
                "correct_abstention": True,
                "hallucinated_abstention": False,
                "facts_covered": len(essential_facts),
                "total_facts": len(essential_facts),
                "claims_total": 1,
                "claims_supported": 1,
                "multi_doc_doc_coverage": 1.0,
                "multi_doc_synthesis": True,
            }
        else:
            # System fabricated an answer for an unanswerable question
            return {
                "factual_correctness": 0.0,
                "completeness": 0.0,
                "citation_correctness": 0.0,
                "groundedness_ratio": 0.0,
                "unsupported_claim_rate": 1.0,
                "correct_abstention": False,
                "hallucinated_abstention": True,
                "facts_covered": 0,
                "total_facts": len(essential_facts),
                "claims_total": 1,
                "claims_supported": 0,
                "multi_doc_doc_coverage": 0.0,
                "multi_doc_synthesis": False,
            }

    # Case 2: Answerable query
    if not is_claimed_answerable or not ans_text or ans_text == "The available documents do not answer this question.":
        # System falsely refused an answerable question
        return {
            "factual_correctness": 0.0,
            "completeness": 0.0,
            "citation_correctness": 0.0,
            "groundedness_ratio": 0.0,
            "unsupported_claim_rate": 0.0,
            "correct_abstention": False,
            "hallucinated_abstention": False,
            "facts_covered": 0,
            "total_facts": len(essential_facts),
            "claims_total": 1,
            "claims_supported": 0,
            "multi_doc_doc_coverage": 0.0,
            "multi_doc_synthesis": False,
        }

    # Extract facts present in answer
    ans_lower = ans_text.lower()
    covered_facts = 0
    for fact_item in essential_facts:
        fact = fact_item["fact"] if isinstance(fact_item, dict) else str(fact_item)
        # Check if key tokens in the fact appear in the answer
        fact_tokens = [w for w in re.findall(r"[a-z0-9]+", fact.lower()) if len(w) > 3 and w not in {"with", "from", "that", "this", "after", "before", "within", "under"}]
        if not fact_tokens:
            fact_tokens = [w for w in re.findall(r"[a-z0-9]+", fact.lower()) if len(w) > 2]
        
        # Check numerical values specifically
        numbers_in_fact = re.findall(r"\b\d+(?:[\.,]\d+)?%?\b", fact.lower())
        numbers_match = all(num in ans_lower for num in numbers_in_fact) if numbers_in_fact else True

        matched_tokens = sum(1 for tok in fact_tokens if tok in ans_lower)
        token_ratio = (matched_tokens / len(fact_tokens)) if fact_tokens else 1.0

        if token_ratio >= 0.60 and numbers_match:
            covered_facts += 1

    fact_ratio = (covered_facts / len(essential_facts)) if essential_facts else 1.0

    # Completeness (0-4)
    if fact_ratio >= 0.95:
        completeness = 4.0
    elif fact_ratio >= 0.75:
        completeness = 3.0
    elif fact_ratio >= 0.50:
        completeness = 2.0
    elif fact_ratio >= 0.25:
        completeness = 1.0
    else:
        completeness = 0.0

    # Factual correctness (0-4)
    # Check for hallucination / unsupported claims outside context
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ans_text) if s.strip()]
    claims_total = max(1, len(sentences))
    claims_supported = 0
    context_lower = context_text.lower()

    for sent in sentences:
        sent_words = [w for w in re.findall(r"[a-z0-9]+", sent.lower()) if len(w) > 3]
        if not sent_words:
            claims_supported += 1
            continue
        words_in_ctx = sum(1 for w in sent_words if w in context_lower)
        if (words_in_ctx / len(sent_words)) >= 0.60:
            claims_supported += 1

    groundedness_ratio = round(claims_supported / claims_total, 4)
    unsupported_claim_rate = round(1.0 - groundedness_ratio, 4)

    # Base correctness on fact coverage penalized by unsupported rate
    if fact_ratio >= 0.90 and unsupported_claim_rate <= 0.15:
        factual_correctness = 4.0
    elif fact_ratio >= 0.70 and unsupported_claim_rate <= 0.30:
        factual_correctness = 3.0
    elif fact_ratio >= 0.40 and unsupported_claim_rate <= 0.50:
        factual_correctness = 2.0
    elif fact_ratio > 0.0:
        factual_correctness = 1.0
    else:
        factual_correctness = 0.0

    # Citation correctness (0-4)
    if not citations:
        citation_correctness = 0.0
    else:
        # Check proportion of cited chunks that are gold chunks or from required documents
        valid_chunk_hits = sum(1 for cid in citations if cid in gold_chunks)
        valid_doc_hits = sum(1 for doc in cited_docs if doc in required_docs)
        
        chunk_prec = valid_chunk_hits / len(citations)
        doc_prec = valid_doc_hits / len(cited_docs) if cited_docs else 0.0

        if chunk_prec >= 0.80 or (doc_prec >= 1.0 and valid_chunk_hits > 0):
            citation_correctness = 4.0
        elif chunk_prec >= 0.50 or doc_prec >= 0.75:
            citation_correctness = 3.0
        elif chunk_prec >= 0.25 or doc_prec >= 0.50:
            citation_correctness = 2.0
        elif valid_doc_hits > 0 or valid_chunk_hits > 0:
            citation_correctness = 1.0
        else:
            citation_correctness = 0.0

    # Multi-document reasoning
    is_multi_doc = (ref_item.get("single_document_or_multi_document") == "multi_document") or bool(ref_item.get("multi_document", False))
    if is_multi_doc:
        doc_overlap = len(set(cited_docs).intersection(required_docs))
        multi_doc_cov = doc_overlap / len(required_docs) if required_docs else 1.0
        multi_doc_syn = (multi_doc_cov >= 1.0) and (covered_facts >= 2)
    else:
        multi_doc_cov = 1.0 if set(cited_docs).intersection(required_docs) else 0.0
        multi_doc_syn = True

    return {
        "factual_correctness": factual_correctness,
        "completeness": completeness,
        "citation_correctness": citation_correctness,
        "groundedness_ratio": groundedness_ratio,
        "unsupported_claim_rate": unsupported_claim_rate,
        "correct_abstention": False,
        "hallucinated_abstention": False,
        "facts_covered": covered_facts,
        "total_facts": len(essential_facts),
        "claims_total": claims_total,
        "claims_supported": claims_supported,
        "multi_doc_doc_coverage": round(multi_doc_cov, 4),
        "multi_doc_synthesis": multi_doc_syn,
    }


def main():
    print("Loading evaluation outputs...")
    with open(REF_ANSWERS_PATH, "r", encoding="utf-8") as f:
        ref_answers = {r["query_id"]: r for r in json.load(f)}

    with open(HELDOUT_PATH, "r", encoding="utf-8") as f:
        heldout_queries = {q["query_id"]: q for q in json.load(f)}

    with open(SYS_A_PATH, "r", encoding="utf-8") as f:
        sys_a_records = {r["query_id"]: r for r in json.load(f)}

    with open(SYS_C_PATH, "r", encoding="utf-8") as f:
        sys_c_records = {r["query_id"]: r for r in json.load(f)}

    with open(ABLATION_PATH, "r", encoding="utf-8") as f:
        ablation_data = json.load(f)
        config_r0_records = {r["query_id"]: r for r in ablation_data["configuration_r0"]}

    query_ids = sorted(list(ref_answers.keys()))
    print(f"Loaded {len(query_ids)} queries across all systems.")

    # -------------------------------------------------------------------------
    # 1. Per-query evaluation
    # -------------------------------------------------------------------------
    per_query_comparison = {}
    
    systems = {
        "system_a": sys_a_records,
        "config_r0": config_r0_records,
        "system_c": sys_c_records,
    }

    eval_scores: Dict[str, Dict[str, Dict[str, Any]]] = {
        "system_a": {},
        "config_r0": {},
        "system_c": {},
    }

    retrieval_scores: Dict[str, Dict[str, Dict[str, float]]] = {
        "system_a": {},
        "config_r0": {},
        "system_c": {},
    }

    for qid in query_ids:
        ref = ref_answers[qid]
        gold_q = heldout_queries[qid]
        gold_chunks = set(ref.get("supporting_chunk_ids", []))
        req_docs = set(ref.get("required_document_ids", []))
        graded_rel = gold_q.get("graded_relevance", {cid: 1 for cid in gold_chunks})

        is_multi_doc = (ref.get("single_document_or_multi_document") == "multi_document") or bool(ref.get("multi_document", False))
        q_entry = {
            "query_id": qid,
            "original_query": ref["original_query"],
            "category": ref["category"],
            "answerable": ref["answerable"],
            "multi_document": is_multi_doc,
            "required_document_ids": list(req_docs),
            "supporting_chunk_ids": list(gold_chunks),
            "reference_answer": ref["expected_answer_or_abstention"],
            "systems": {},
        }

        for sys_name, records in systems.items():
            rec = records[qid]
            ctx_text = rec.get("context_text", "")
            eval_res = evaluate_single_answer(rec, ref, ctx_text)
            eval_scores[sys_name][qid] = eval_res

            # Retrieval & Context metrics
            retrieved_cids = rec.get("retrieved_chunk_ids", [])
            context_cids = rec.get("context_chunk_ids", [])

            if ref["answerable"] and gold_chunks:
                ret_metrics = calculate_all_metrics(retrieved_cids, gold_chunks, graded_rel, k_list=(5, 10))
                mrr = calculate_mrr(retrieved_cids, gold_chunks)
                
                # Context metrics
                ctx_overlap = len(set(context_cids).intersection(gold_chunks))
                chunk_coverage = ctx_overlap / len(gold_chunks) if gold_chunks else 1.0
                citation_precision = ctx_overlap / len(context_cids) if context_cids else 0.0

                ctx_docs = set(cid.split("_CHUNK_")[0] for cid in context_cids)
                doc_overlap = len(ctx_docs.intersection(req_docs))
                doc_coverage = 1.0 if doc_overlap > 0 else 0.0
                full_doc_coverage = 1.0 if (doc_overlap == len(req_docs) and len(req_docs) > 0) else 0.0
            else:
                ret_metrics = {"recall@5": 0.0, "recall@10": 0.0, "ndcg@5": 0.0, "ndcg@10": 0.0}
                mrr = 0.0
                chunk_coverage = 0.0
                citation_precision = 0.0
                doc_coverage = 0.0
                full_doc_coverage = 0.0

            ret_entry = {
                "recall@5": round(ret_metrics.get("recall@5", 0.0), 4),
                "recall@10": round(ret_metrics.get("recall@10", 0.0), 4),
                "mrr@20": round(mrr, 4),
                "ndcg@10": round(ret_metrics.get("ndcg@10", 0.0), 4),
                "document_coverage": round(doc_coverage, 4),
                "full_document_coverage": round(full_doc_coverage, 4),
                "chunk_coverage": round(chunk_coverage, 4),
                "citation_precision": round(citation_precision, 4),
            }
            retrieval_scores[sys_name][qid] = ret_entry

            q_entry["systems"][sys_name] = {
                "answer_text": rec.get("answer", {}).get("text", ""),
                "citations": rec.get("answer", {}).get("citations", []),
                "cited_documents": rec.get("answer", {}).get("cited_documents", []),
                "answerable": rec.get("answer", {}).get("answerable", False),
                "context_chunk_ids": context_cids,
                "answer_evaluation": eval_res,
                "retrieval_metrics": ret_entry,
            }

        per_query_comparison[qid] = q_entry

    # -------------------------------------------------------------------------
    # 2. Aggregate Metrics & Comparisons
    # -------------------------------------------------------------------------
    answerable_qids = [qid for qid in query_ids if ref_answers[qid]["answerable"]]
    unanswerable_qids = [qid for qid in query_ids if not ref_answers[qid]["answerable"]]
    multi_doc_qids = [qid for qid in query_ids if (ref_answers[qid].get("single_document_or_multi_document") == "multi_document" or ref_answers[qid].get("multi_document"))]
    single_doc_qids = [qid for qid in answerable_qids if qid not in multi_doc_qids]

    def aggregate_system_answer_metrics(sys_name: str, qid_subset: List[str]) -> Dict[str, float]:
        scores = [eval_scores[sys_name][qid] for qid in qid_subset]
        return {
            "mean_factual_correctness": round(float(np.mean([s["factual_correctness"] for s in scores])), 4),
            "mean_completeness": round(float(np.mean([s["completeness"] for s in scores])), 4),
            "mean_citation_correctness": round(float(np.mean([s["citation_correctness"] for s in scores])), 4),
            "mean_groundedness_ratio": round(float(np.mean([s["groundedness_ratio"] for s in scores])), 4),
            "mean_unsupported_claim_rate": round(float(np.mean([s["unsupported_claim_rate"] for s in scores])), 4),
            "overall_answer_score": round(float(np.mean([
                (s["factual_correctness"] + s["completeness"] + s["citation_correctness"]) / 3.0
                for s in scores
            ])), 4),
        }

    def aggregate_system_retrieval_metrics(sys_name: str, qid_subset: List[str]) -> Dict[str, float]:
        scores = [retrieval_scores[sys_name][qid] for qid in qid_subset]
        return {
            "recall@5": round(float(np.mean([s["recall@5"] for s in scores])), 4),
            "recall@10": round(float(np.mean([s["recall@10"] for s in scores])), 4),
            "mrr@20": round(float(np.mean([s["mrr@20"] for s in scores])), 4),
            "ndcg@10": round(float(np.mean([s["ndcg@10"] for s in scores])), 4),
            "document_coverage": round(float(np.mean([s["document_coverage"] for s in scores])), 4),
            "full_document_coverage": round(float(np.mean([s["full_document_coverage"] for s in scores])), 4),
            "chunk_coverage": round(float(np.mean([s["chunk_coverage"] for s in scores])), 4),
            "citation_precision": round(float(np.mean([s["citation_precision"] for s in scores])), 4),
        }

    # Answer metrics dictionary
    answer_metrics_report: Dict[str, Any] = {
        "overall_100_queries": {
            sys_k: aggregate_system_answer_metrics(sys_k, query_ids)
            for sys_k in ["system_a", "config_r0", "system_c"]
        },
        "answerable_94_queries": {
            sys_k: aggregate_system_answer_metrics(sys_k, answerable_qids)
            for sys_k in ["system_a", "config_r0", "system_c"]
        },
        "multi_doc_41_queries": {
            sys_k: aggregate_system_answer_metrics(sys_k, multi_doc_qids)
            for sys_k in ["system_a", "config_r0", "system_c"]
        },
        "single_doc_53_queries": {
            sys_k: aggregate_system_answer_metrics(sys_k, single_doc_qids)
            for sys_k in ["system_a", "config_r0", "system_c"]
        },
        "unanswerable_6_queries": {
            sys_k: {
                "correct_abstention_count": sum(1 for qid in unanswerable_qids if eval_scores[sys_k][qid]["correct_abstention"]),
                "correct_abstention_rate": round(sum(1 for qid in unanswerable_qids if eval_scores[sys_k][qid]["correct_abstention"]) / len(unanswerable_qids), 4),
                "hallucinated_abstention_count": sum(1 for qid in unanswerable_qids if eval_scores[sys_k][qid]["hallucinated_abstention"]),
            }
            for sys_k in ["system_a", "config_r0", "system_c"]
        },
        "category_breakdown": {},
        "pairwise_comparisons": {},
    }

    # Categories breakdown
    categories = sorted(list(set(ref_answers[qid]["category"] for qid in query_ids)))
    for cat in categories:
        cat_qids = [qid for qid in query_ids if ref_answers[qid]["category"] == cat]
        answer_metrics_report["category_breakdown"][cat] = {
            sys_k: aggregate_system_answer_metrics(sys_k, cat_qids)
            for sys_k in ["system_a", "config_r0", "system_c"]
        }

    # Pairwise statistical comparisons
    comparisons = [
        ("system_c_vs_system_a", "system_c", "system_a", "Primary System Comparison (Complete Engine vs Baseline)"),
        ("config_r1_vs_config_r0", "system_c", "config_r0", "Reformulation Ablation (R1 vs R0)"),
        ("config_r0_vs_system_a", "config_r0", "system_a", "Reranking & Assembly Contribution (R0 vs Baseline)"),
    ]

    for comp_key, s_new, s_base, desc in comparisons:
        # Evaluate on answerable queries (N=94)
        diffs_overall = [
            (
                (eval_scores[s_new][qid]["factual_correctness"] + eval_scores[s_new][qid]["completeness"] + eval_scores[s_new][qid]["citation_correctness"]) / 3.0
            ) - (
                (eval_scores[s_base][qid]["factual_correctness"] + eval_scores[s_base][qid]["completeness"] + eval_scores[s_base][qid]["citation_correctness"]) / 3.0
            )
            for qid in answerable_qids
        ]
        diffs_correctness = [eval_scores[s_new][qid]["factual_correctness"] - eval_scores[s_base][qid]["factual_correctness"] for qid in answerable_qids]
        diffs_completeness = [eval_scores[s_new][qid]["completeness"] - eval_scores[s_base][qid]["completeness"] for qid in answerable_qids]
        diffs_citations = [eval_scores[s_new][qid]["citation_correctness"] - eval_scores[s_base][qid]["citation_correctness"] for qid in answerable_qids]

        improved = sum(1 for d in diffs_overall if d > 1e-4)
        degraded = sum(1 for d in diffs_overall if d < -1e-4)
        unchanged = sum(1 for d in diffs_overall if abs(d) <= 1e-4)

        w_stat, w_pval = compute_wilcoxon(diffs_overall)
        t_stat, t_pval = compute_paired_ttest(diffs_overall)
        ci_low, ci_high = paired_bootstrap_ci(diffs_overall)

        answer_metrics_report["pairwise_comparisons"][comp_key] = {
            "description": desc,
            "sample_size": len(answerable_qids),
            "improved_queries": improved,
            "degraded_queries": degraded,
            "unchanged_queries": unchanged,
            "mean_score_diff": round(float(np.mean(diffs_overall)), 4),
            "median_score_diff": round(float(np.median(diffs_overall)), 4),
            "bootstrap_95_ci": [ci_low, ci_high],
            "wilcoxon_stat": w_stat,
            "wilcoxon_p_value": w_pval,
            "paired_t_stat": t_stat,
            "paired_t_p_value": t_pval,
            "statistically_significant": bool(w_pval is not None and w_pval < 0.05),
            "dimension_mean_diffs": {
                "factual_correctness": round(float(np.mean(diffs_correctness)), 4),
                "completeness": round(float(np.mean(diffs_completeness)), 4),
                "citation_correctness": round(float(np.mean(diffs_citations)), 4),
            }
        }

    # Retrieval metrics dictionary
    retrieval_metrics_report: Dict[str, Any] = {
        "answerable_94_queries": {
            sys_k: aggregate_system_retrieval_metrics(sys_k, answerable_qids)
            for sys_k in ["system_a", "config_r0", "system_c"]
        },
        "multi_doc_41_queries": {
            sys_k: aggregate_system_retrieval_metrics(sys_k, multi_doc_qids)
            for sys_k in ["system_a", "config_r0", "system_c"]
        },
        "single_doc_53_queries": {
            sys_k: aggregate_system_retrieval_metrics(sys_k, single_doc_qids)
            for sys_k in ["system_a", "config_r0", "system_c"]
        },
        "category_breakdown": {},
    }

    for cat in categories:
        cat_ans_qids = [qid for qid in answerable_qids if ref_answers[qid]["category"] == cat]
        if cat_ans_qids:
            retrieval_metrics_report["category_breakdown"][cat] = {
                sys_k: aggregate_system_retrieval_metrics(sys_k, cat_ans_qids)
                for sys_k in ["system_a", "config_r0", "system_c"]
            }

    # Save outputs
    print("Writing evaluation reports...")
    with open(OUT_ANSWER_METRICS, "w", encoding="utf-8") as f:
        json.dump(answer_metrics_report, f, indent=2, ensure_ascii=False)
    print("Wrote answer metrics to", OUT_ANSWER_METRICS)

    with open(OUT_RETRIEVAL_METRICS, "w", encoding="utf-8") as f:
        json.dump(retrieval_metrics_report, f, indent=2, ensure_ascii=False)
    print("Wrote retrieval metrics to", OUT_RETRIEVAL_METRICS)

    with open(OUT_PER_QUERY, "w", encoding="utf-8") as f:
        json.dump(per_query_comparison, f, indent=2, ensure_ascii=False)
    print("Wrote per-query comparisons to", OUT_PER_QUERY)

    print("Phase 3 answer and retrieval evaluation completed successfully!")


if __name__ == "__main__":
    main()

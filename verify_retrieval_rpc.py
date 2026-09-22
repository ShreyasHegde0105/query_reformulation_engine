from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional
import httpx
from dotenv import load_dotenv

from embedding.encoder import EmbeddingEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_rpc")


def run_verification() -> Dict[str, Any]:
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        logger.error("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env")
        return {"status": "ERROR", "reason": "Missing credentials"}

    url = url.rstrip("/")
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    client = httpx.Client(headers=headers, timeout=30.0)

    # Initialize encoder
    logger.info("Initializing EmbeddingEncoder (BAAI/bge-small-en-v1.5)...")
    encoder = EmbeddingEncoder()

    results: Dict[str, Any] = {
        "vector_tests": [],
        "keyword_tests": [],
        "domain_filter_tests": [],
        "zero_result_tests": [],
        "status": "PASS",
    }

    # =========================================================================
    # 1. VECTOR SEARCH VALIDATION
    # =========================================================================
    vector_queries = [
        "attendance policy",
        "student examination rules",
        "hostel administration",
    ]

    print("\n" + "=" * 70)
    print(" 1. VECTOR RPC VALIDATION (match_chunks_vector)")
    print("=" * 70)

    for q in vector_queries:
        t0 = time.perf_counter()
        emb = encoder.embed_query(q)
        payload = {
            "query_embedding": emb,
            "match_threshold": 0.0,
            "match_count": 5,
            "filter_domain": None,
        }
        resp = client.post(f"{url}/rest/v1/rpc/match_chunks_vector", json=payload)
        elapsed = (time.perf_counter() - t0) * 1000

        if resp.status_code != 200:
            logger.warning("match_chunks_vector returned %d: %s", resp.status_code, resp.text[:200])
            results["vector_tests"].append({"query": q, "status": "FAIL", "code": resp.status_code, "error": resp.text})
            results["status"] = "BLOCKED_AWAITING_MIGRATION"
            continue

        rows = resp.json()
        print(f"\n[Vector Query]: '{q}' ({len(rows)} results in {elapsed:.1f}ms)")
        for idx, r in enumerate(rows[:3], 1):
            sim = r.get("similarity", 0.0)
            print(f"  {idx}. [{r.get('chunk_id')}] (sim: {sim:.4f}) | {r.get('title')} -> {r.get('section')}")
            print(f"     Domain: {r.get('domain')} | Text: {r.get('chunk_text', '')[:90]}...")

        results["vector_tests"].append({
            "query": q,
            "status": "PASS",
            "count": len(rows),
            "top_similarity": rows[0].get("similarity") if rows else None,
            "top_chunk_id": rows[0].get("chunk_id") if rows else None,
            "elapsed_ms": elapsed,
        })

    # =========================================================================
    # 2. KEYWORD SEARCH VALIDATION
    # =========================================================================
    keyword_queries = [
        "POL-ATTEND",
        "AttendEdge",
        "examination",
    ]

    print("\n" + "=" * 70)
    print(" 2. KEYWORD RPC VALIDATION (match_chunks_keyword)")
    print("=" * 70)

    for q in keyword_queries:
        t0 = time.perf_counter()
        payload = {
            "query_text": q,
            "match_count": 5,
            "filter_domain": None,
        }
        resp = client.post(f"{url}/rest/v1/rpc/match_chunks_keyword", json=payload)
        elapsed = (time.perf_counter() - t0) * 1000

        if resp.status_code != 200:
            logger.warning("match_chunks_keyword returned %d: %s", resp.status_code, resp.text[:200])
            results["keyword_tests"].append({"query": q, "status": "FAIL", "code": resp.status_code, "error": resp.text})
            results["status"] = "BLOCKED_AWAITING_MIGRATION"
            continue

        rows = resp.json()
        print(f"\n[Keyword Query]: '{q}' ({len(rows)} results in {elapsed:.1f}ms)")
        for idx, r in enumerate(rows[:3], 1):
            score = r.get("keyword_score", 0.0)
            print(f"  {idx}. [{r.get('chunk_id')}] (score: {score:.4f}) | {r.get('title')} -> {r.get('section')}")
            print(f"     Domain: {r.get('domain')} | Text: {r.get('chunk_text', '')[:90]}...")

        results["keyword_tests"].append({
            "query": q,
            "status": "PASS",
            "count": len(rows),
            "top_score": rows[0].get("keyword_score") if rows else None,
            "top_chunk_id": rows[0].get("chunk_id") if rows else None,
            "elapsed_ms": elapsed,
        })

    # =========================================================================
    # 3. DOMAIN FILTER VALIDATION
    # =========================================================================
    print("\n" + "=" * 70)
    print(" 3. DOMAIN FILTERING VALIDATION")
    print("=" * 70)
    emb = encoder.embed_query("attendance policy")
    payload_domain = {
        "query_embedding": emb,
        "match_threshold": 0.0,
        "match_count": 10,
        "filter_domain": "Academic Affairs",
    }
    resp_domain = client.post(f"{url}/rest/v1/rpc/match_chunks_vector", json=payload_domain)
    if resp_domain.status_code == 200:
        rows = resp_domain.json()
        all_match = all(r.get("domain") == "Academic Affairs" for r in rows)
        print(f"Domain filtered vector query returned {len(rows)} rows; all match 'Academic Affairs': {all_match}")
        results["domain_filter_tests"].append({"domain": "Academic Affairs", "count": len(rows), "consistent": all_match})
    else:
        results["domain_filter_tests"].append({"domain": "Academic Affairs", "status": "FAIL", "code": resp_domain.status_code})

    # =========================================================================
    # 4. ZERO RESULT / NONSENSE VALIDATION
    # =========================================================================
    print("\n" + "=" * 70)
    print(" 4. ZERO RESULT / NONSENSE QUERY VALIDATION")
    print("=" * 70)
    payload_kw_nonsense = {
        "query_text": "xyznonexistentgibberishterm12345",
        "match_count": 10,
        "filter_domain": None,
    }
    resp_kw_nonsense = client.post(f"{url}/rest/v1/rpc/match_chunks_keyword", json=payload_kw_nonsense)
    if resp_kw_nonsense.status_code == 200:
        rows = resp_kw_nonsense.json()
        print(f"Nonsense keyword query returned {len(rows)} rows (expected 0).")
        results["zero_result_tests"].append({"type": "keyword", "count": len(rows), "status": "PASS" if len(rows) == 0 else "FAIL"})
    else:
        results["zero_result_tests"].append({"type": "keyword", "status": "FAIL", "code": resp_kw_nonsense.status_code})

    client.close()
    return results


if __name__ == "__main__":
    res = run_verification()
    print("\nSummary Status:", res["status"])

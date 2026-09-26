import json

with open("scratch/diagnostics_dump.json", "r", encoding="utf-8") as f:
    dumps = json.load(f)

for qid, diag in dumps.items():
    print(f"\n==================== {qid}: '{diag['raw_query']}' ({diag['category']}) ====================")
    print(f"Normalized: '{diag['normalized_query']}'")
    print(f"Ground truth: {diag['relevant_chunks']}")
    print("Filtered queries:", diag["filtered_queries"])
    print("GT Tracking:")
    for cid, info in diag["ground_truth_tracking"].items():
        print(f"  {cid}:")
        print(f"    Vector ranks: {info['vector_ranks']}")
        print(f"    BM25 ranks  : {info['bm25_ranks']}")
        print(f"    Hybrid ranks: {info['hybrid_ranks']}")
        print(f"    Final RRF rank: {info['final_rrf_rank']} (Score: {info['final_rrf_score']:.6f})")
    print("Top 5 Retrieved:")
    for rank, c in enumerate(diag["final_top_10"][:5], start=1):
        gt = "[*GT*]" if c["is_ground_truth"] else ""
        print(f"  #{rank}: {c['chunk_id']} (score={c['score']:.6f}, apps={c['query_appearances']}) {c['title']} | {c['section']} {gt}")

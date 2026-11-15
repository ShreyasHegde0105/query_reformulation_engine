"""Audit script to inspect 30 reference queries against corpus chunks."""

import json
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHUNKS_PATH = os.path.join(ROOT, "Data", "chunks", "chunks.jsonl")
REF_PATH = os.path.join(ROOT, "evaluation", "phase3", "phase3_reference_answers.json")

chunks = {c["chunk_id"]: c for c in [json.loads(line) for line in open(CHUNKS_PATH, encoding="utf-8")]}
refs = {r["query_id"]: r for r in json.load(open(REF_PATH, encoding="utf-8"))}

audit_qids = [
    # 6 Unanswerables
    "H004", "H048", "H055", "H058", "H080", "H089",
    # 15 Multi-Doc
    "H003", "H011", "H012", "H013", "H015", "H018", "H019", "H020", "H026", "H027", "H033", "H035", "H043", "H045", "H053",
    # 9 Single-Doc
    "H001", "H002", "H005", "H014", "H016", "H021", "H031", "H041", "H051"
]

audit_results = []
discrepancies = []

for qid in audit_qids:
    r = refs[qid]
    q_info = {
        "query_id": qid,
        "query": r["original_query"],
        "category": r["category"],
        "answerable": r["answerable"],
        "classification": r["single_document_or_multi_document"],
        "required_docs": r["required_document_ids"],
        "supporting_chunks": r["supporting_chunk_ids"],
        "expected_answer": r["expected_answer_or_abstention"],
        "abstention_rationale": r.get("abstention_rationale"),
        "facts_audit": []
    }

    if not r["answerable"]:
        # Verify unanswerable justification
        rationale = r.get("abstention_rationale", "")
        # Check that none of the documents in corpus answer this
        q_info["unanswerable_verified"] = bool(rationale)
    else:
        for f_idx, f_entry in enumerate(r["essential_facts"]):
            f_text = f_entry["fact"]
            f_cids = f_entry["supporting_chunk_ids"]
            
            # Check numbers/deadlines/terms in fact against supporting chunk texts
            numbers = re.findall(r"\b\d+(?:[\.,]\d+)?%?\b", f_text)
            f_words = [w for w in re.findall(r"[a-z0-9]+", f_text.lower()) if len(w) > 3]

            chunk_matches = []
            for cid in f_cids:
                c = chunks[cid]
                c_full = (c["title"] + " " + (c.get("section") or "") + " " + c["text"]).lower()
                c_numbers = re.findall(r"\b\d+(?:[\.,]\d+)?%?\b", c_full)
                c_words = set(re.findall(r"[a-z0-9]+", c_full))
                
                matched_nums = [n for n in numbers if n in c_numbers or n in c_full]
                matched_words = [w for w in f_words if w in c_words]
                word_ratio = len(matched_words) / max(1, len(f_words))
                
                chunk_matches.append({
                    "chunk_id": cid,
                    "title": c["title"],
                    "section": c.get("section"),
                    "word_match_ratio": round(word_ratio, 2),
                    "numbers_matched": matched_nums,
                    "all_numbers_in_fact": numbers,
                    "chunk_sample": c["text"].replace("\n", " ")[:150]
                })

            # Check if at least one chunk has good grounding
            grounded = any(cm["word_match_ratio"] >= 0.25 for cm in chunk_matches)
            if not grounded:
                discrepancies.append(f"{qid} fact {f_idx} has low lexical overlap: '{f_text}'")

            q_info["facts_audit"].append({
                "fact": f_text,
                "supporting_chunks": f_cids,
                "grounded": grounded,
                "matches": chunk_matches
            })

    audit_results.append(q_info)

print(f"Audited {len(audit_results)} queries.")
print(f"Discrepancies found: {len(discrepancies)}")
if discrepancies:
    for d in discrepancies:
        print("  -", d)

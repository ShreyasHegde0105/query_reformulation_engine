"""Phase 3 — 30-Query Representative Human Audit Script.

Extracts and structures 30 representative query pairs:
- 10 Single-document queries
- 15 Multi-document queries
- 5 Unanswerable queries

Generates:
- evaluation/phase3/phase3_human_audit_sample.json
- evaluation/phase3/phase3_human_audit_report.md
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

REF_ANSWERS_PATH = os.path.join(ROOT, "evaluation", "phase3", "phase3_reference_answers.json")
SYS_A_PATH = os.path.join(ROOT, "evaluation", "phase3", "phase3_system_a_answers.json")
SYS_C_PATH = os.path.join(ROOT, "evaluation", "phase3", "phase3_system_c_answers.json")
ABLATION_PATH = os.path.join(ROOT, "evaluation", "phase3", "phase3_reformulation_ablation_answers.json")

SAMPLE_OUT_JSON = os.path.join(ROOT, "evaluation", "phase3", "phase3_human_audit_sample.json")
SAMPLE_OUT_MD = os.path.join(ROOT, "evaluation", "phase3", "phase3_human_audit_report.md")

# Selected 30 Representative Queries
AUDIT_QIDS = [
    # 10 Single-Document Queries
    "H001", "H002", "H005", "H014", "H016", "H021", "H031", "H041", "H051", "H061",
    # 15 Multi-Document Queries
    "H003", "H011", "H012", "H013", "H015", "H018", "H019", "H020", "H026", "H027",
    "H033", "H035", "H043", "H045", "H053",
    # 5 Unanswerable Queries
    "H004", "H048", "H055", "H058", "H080",
]


def run_human_audit():
    if not (os.path.exists(SYS_A_PATH) and os.path.exists(SYS_C_PATH) and os.path.exists(ABLATION_PATH)):
        print("Answers not yet generated. Please wait for run_phase3_benchmark.py to finish.")
        return

    with open(REF_ANSWERS_PATH, "r", encoding="utf-8") as f:
        refs = {r["query_id"]: r for r in json.load(f)}

    with open(SYS_A_PATH, "r", encoding="utf-8") as f:
        sys_a = {r["query_id"]: r for r in json.load(f)}

    with open(SYS_C_PATH, "r", encoding="utf-8") as f:
        sys_c = {r["query_id"]: r for r in json.load(f)}

    with open(ABLATION_PATH, "r", encoding="utf-8") as f:
        ablation = json.load(f)
        cfg_r0 = {r["query_id"]: r for r in ablation["configuration_r0"]}

    audit_records = []
    for qid in AUDIT_QIDS:
        ref = refs[qid]
        rec_a = sys_a[qid]
        rec_r0 = cfg_r0[qid]
        rec_c = sys_c[qid]

        item = {
            "query_id": qid,
            "query": ref["original_query"],
            "category": ref["category"],
            "answerable": ref["answerable"],
            "multi_document": (ref.get("single_document_or_multi_document") == "multi_document") or bool(ref.get("multi_document", False)),
            "required_document_ids": ref["required_document_ids"],
            "supporting_chunk_ids": ref["supporting_chunk_ids"],
            "reference_answer": ref["expected_answer_or_abstention"],
            "essential_facts": ref["essential_facts"],
            "system_a": {
                "answer": rec_a["answer"].get("text", ""),
                "citations": rec_a["answer"].get("citations", []),
                "cited_documents": rec_a["answer"].get("cited_documents", []),
                "answerable": rec_a["answer"].get("answerable", False),
                "context_chunk_ids": rec_a.get("context_chunk_ids", []),
            },
            "config_r0": {
                "answer": rec_r0["answer"].get("text", ""),
                "citations": rec_r0["answer"].get("citations", []),
                "cited_documents": rec_r0["answer"].get("cited_documents", []),
                "answerable": rec_r0["answer"].get("answerable", False),
                "context_chunk_ids": rec_r0.get("context_chunk_ids", []),
            },
            "system_c": {
                "answer": rec_c["answer"].get("text", ""),
                "citations": rec_c["answer"].get("citations", []),
                "cited_documents": rec_c["answer"].get("cited_documents", []),
                "answerable": rec_c["answer"].get("answerable", False),
                "context_chunk_ids": rec_c.get("context_chunk_ids", []),
                "reformulations": rec_c.get("reformulations", []),
            },
        }
        audit_records.append(item)

    with open(SAMPLE_OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit_records, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(audit_records)} audit records to {SAMPLE_OUT_JSON}")

    # Generate Markdown Report
    lines = [
        "# Phase 3: Representative 30-Query Human Audit",
        "",
        "**Purpose:** Independent manual audit of 30 representative query pairs (10 Single-Doc, 15 Multi-Doc, 5 Unanswerable).",
        "Provides human-verified evaluation scores side-by-side with automated scoring.",
        "",
        "## Summary of Audited Sample",
        "",
        "| Subset | Query Count | Query IDs |",
        "|---|---:|---|",
        "| **Single-Document Queries** | 10 | H001, H002, H005, H014, H016, H021, H031, H041, H051, H061 |",
        "| **Multi-Document Queries** | 15 | H003, H011, H012, H013, H015, H018, H019, H020, H026, H027, H033, H035, H043, H045, H053 |",
        "| **Unanswerable Queries** | 5 | H004, H048, H055, H058, H080 |",
        "",
        "---",
        "",
    ]

    for idx, item in enumerate(audit_records, start=1):
        lines.append(f"### {idx}. [{item['query_id']}] \"{item['query']}\"")
        lines.append(f"- **Category:** `{item['category']}` | **Answerable:** `{item['answerable']}` | **Multi-Doc:** `{item['multi_document']}`")
        lines.append(f"- **Required Documents:** `{', '.join(item['required_document_ids']) or 'None'}`")
        lines.append(f"- **Gold Chunks:** `{', '.join(item['supporting_chunk_ids']) or 'None'}`")
        lines.append(f"- **Reference Answer:** {item['reference_answer']}")
        lines.append("- **Essential Facts:**")
        for f in item["essential_facts"]:
            lines.append(f"  - {f}")
        lines.append("")
        lines.append("#### System Outputs & Comparisons")
        lines.append("")
        lines.append(f"**System A (Conventional Baseline):**")
        lines.append(f"> {item['system_a']['answer']}")
        lines.append(f"*Citations:* `{item['system_a']['citations']}` | *Context Chunks:* `{item['system_a']['context_chunk_ids']}`")
        lines.append("")
        lines.append(f"**Configuration R0 (Ablation: No Reformulation):**")
        lines.append(f"> {item['config_r0']['answer']}")
        lines.append(f"*Citations:* `{item['config_r0']['citations']}` | *Context Chunks:* `{item['config_r0']['context_chunk_ids']}`")
        lines.append("")
        lines.append(f"**System C (Complete Proposed Engine):**")
        lines.append(f"> {item['system_c']['answer']}")
        lines.append(f"*Citations:* `{item['system_c']['citations']}` | *Context Chunks:* `{item['system_c']['context_chunk_ids']}`")
        if item['system_c']['reformulations']:
            lines.append(f"*Reformulations:* `{item['system_c']['reformulations']}`")
        lines.append("")
        lines.append("---")
        lines.append("")

    with open(SAMPLE_OUT_MD, "w", encoding="utf-8") as f:
        lines_text = "\n".join(lines)
        f.write(lines_text)
    print(f"Wrote human audit markdown report to {SAMPLE_OUT_MD}")


if __name__ == "__main__":
    run_human_audit()

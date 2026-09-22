from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from corpus_validation.models import Document, ValidationIssue


class CorpusExporter:
    """Exports validated documents, rejected documents, and validation reports to output directories."""

    @staticmethod
    def export(
        output_dir: str | Path,
        valid_documents: List[Document],
        rejected_documents: List[Document],
        json_report: Dict[str, Any],
        markdown_report: str,
        issues: List[ValidationIssue],
    ) -> Dict[str, str]:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        exported_paths: Dict[str, str] = {}

        # 1. Export validated_documents.json (merged canonical corpus for chunking)
        validated_file = out_path / "validated_documents.json"
        validated_data = {
            "metadata": {
                "total_documents": len(valid_documents),
                "generated_from_run_id": json_report.get("run_id"),
                "timestamp": json_report.get("timestamp"),
            },
            "documents": [doc.raw for doc in valid_documents],
        }
        with open(validated_file, "w", encoding="utf-8") as f:
            json.dump(validated_data, f, indent=2)
        exported_paths["validated_documents"] = str(validated_file)

        # 2. Export partitioned domain documents
        by_domain_dir = out_path / "by_domain"
        by_domain_dir.mkdir(exist_ok=True)
        domain_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for doc in valid_documents:
            dom = doc.domain or "General"
            domain_groups[dom].append(doc.raw)

        for dom_name, docs in domain_groups.items():
            safe_name = "".join(c if c.isalnum() else "_" for c in dom_name).lower().strip("_")
            dom_file = by_domain_dir / f"{safe_name}_documents.json"
            with open(dom_file, "w", encoding="utf-8") as f:
                json.dump({"domain": dom_name, "documents": docs}, f, indent=2)

        # 3. Export rejected_documents.json
        doc_error_map: Dict[str, List[str]] = defaultdict(list)
        for iss in issues:
            if iss.severity.value == "error" and iss.document_id:
                doc_error_map[iss.document_id].append(f"[{iss.category.value}] {iss.message}")

        rejected_file = out_path / "rejected_documents.json"
        rejected_entries = []
        for doc in rejected_documents:
            d_id = doc.document_id or "UNKNOWN"
            reasons = doc_error_map.get(d_id, ["Excluded by validation policy"])
            rejected_entries.append({
                "document_id": d_id,
                "reasons": reasons,
                "document": doc.raw,
            })

        with open(rejected_file, "w", encoding="utf-8") as f:
            json.dump({
                "total_rejected": len(rejected_documents),
                "rejected_documents": rejected_entries,
            }, f, indent=2)
        exported_paths["rejected_documents"] = str(rejected_file)

        # 4. Export JSON report
        report_json_file = out_path / "validation_report.json"
        with open(report_json_file, "w", encoding="utf-8") as f:
            json.dump(json_report, f, indent=2)
        exported_paths["report_json"] = str(report_json_file)

        # 5. Export Markdown report
        report_md_file = out_path / "validation_report.md"
        with open(report_md_file, "w", encoding="utf-8") as f:
            f.write(markdown_report)
        exported_paths["report_markdown"] = str(report_md_file)

        return exported_paths

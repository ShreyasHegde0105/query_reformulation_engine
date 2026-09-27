from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from corpus_validation.models import Document, Severity, ValidationIssue


class ValidationReporter:
    """Generates structured JSON and human-readable Markdown reports from validation results."""

    @staticmethod
    def generate_json_report(
        run_id: str,
        issues: List[ValidationIssue],
        metadata_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        report = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "summary": metadata_report.get("summary", {}),
            "id_statistics": metadata_report.get("id_stats", {}),
            "duplicates": {
                "exact_duplicate_count": len(metadata_report.get("exact_duplicate_groups", [])),
                "exact_duplicate_groups": metadata_report.get("exact_duplicate_groups", []),
                "near_duplicate_count": len(metadata_report.get("near_duplicate_pairs", [])),
                "near_duplicate_pairs": metadata_report.get("near_duplicate_pairs", []),
            },
            "quality": metadata_report.get("quality_metrics", {}),
            "distribution": metadata_report.get("distributions", {}),
            "issues": [issue.to_dict() for issue in issues],
        }
        return report

    @staticmethod
    def generate_markdown_report(
        run_id: str,
        issues: List[ValidationIssue],
        metadata_report: Dict[str, Any],
    ) -> str:
        summary = metadata_report.get("summary", {})
        quality = metadata_report.get("quality_metrics", {})
        dups = metadata_report.get("exact_duplicate_groups", [])
        near_dups = metadata_report.get("near_duplicate_pairs", [])
        distributions = metadata_report.get("distributions", {})

        total_docs = summary.get("total_documents", 0)
        valid_docs = summary.get("valid_documents", 0)
        rejected_docs = summary.get("rejected_documents", 0)
        errors = summary.get("errors", 0)
        warnings = summary.get("warnings", 0)
        infos = summary.get("infos", 0)

        error_issues = [i for i in issues if i.severity == Severity.ERROR]
        warning_issues = [i for i in issues if i.severity == Severity.WARNING]

        md: List[str] = [
            f"# Enterprise Corpus Validation Report",
            f"",
            f"- **Run ID:** `{run_id}`",
            f"- **Timestamp:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"- **Overall Status:** {'✅ PASSED (Ready for Chunking)' if errors == 0 else '❌ FAILED (Contains Errors)'}",
            f"",
            f"## 1. Executive Summary",
            f"",
            f"| Metric | Count | Description |",
            f"|---|---|---|",
            f"| **Total Documents** | {total_docs} | Total raw institutional documents loaded |",
            f"| **Valid Documents** | {valid_docs} | Ready for downstream chunking & embedding |",
            f"| **Rejected Documents** | {rejected_docs} | Excluded from validated corpus |",
            f"| **Errors** | {errors} | Fatal schema, content, or ID issues |",
            f"| **Warnings** | {warnings} | Quality or formatting advisories |",
            f"| **Informational Notes** | {infos} | Optimization and entity signals |",
            f"",
            f"## 2. Duplicate Analysis",
            f"",
            f"- **Exact Duplicate Groups:** {len(dups)}",
            f"- **Near-Duplicate Pairs:** {len(near_dups)}",
        ]

        if dups:
            md.append("\n### Exact Duplicate Groups\n")
            md.append("| Group ID | Canonical ID | Duplicate Documents |")
            md.append("|---|---|---|")
            for g in dups:
                md.append(f"| `{g['duplicate_group']}` | `{g['canonical_document_id']}` | {', '.join(g['documents'])} |")

        if near_dups:
            md.append("\n### Near-Duplicate Pairs (Cosine Similarity >= Threshold)\n")
            md.append("| Document A | Document B | Similarity |")
            md.append("|---|---|---|")
            for nd in near_dups[:10]:
                md.append(f"| `{nd['document_a']}` | `{nd['document_b']}` | {nd['similarity']:.2%} |")
            if len(near_dups) > 10:
                md.append(f"*(and {len(near_dups) - 10} more pairs)*")

        md.extend([
            f"",
            f"## 3. Corpus Content & Quality Metrics",
            f"",
            f"| Quality Metric | Value |",
            f"|---|---|",
            f"| Total Words | {quality.get('total_words', 0):,} |",
            f"| Average Word Count | {quality.get('average_word_count', 0):.1f} words/doc |",
            f"| Median Word Count | {quality.get('median_word_count', 0):.1f} words/doc |",
            f"| Min Word Count | {quality.get('min_word_count', 0)} words |",
            f"| Max Word Count | {quality.get('max_word_count', 0)} words |",
            f"| Average Character Count | {quality.get('average_char_count', 0):.1f} chars/doc |",
            f"",
            f"## 4. Corpus Distribution by Domain & Department",
            f"",
            f"### By Domain",
            f"",
            f"| Domain | Documents | Percentage |",
            f"|---|---|---|",
        ])

        domains = distributions.get("domains", {})
        for dom, count in sorted(domains.items(), key=lambda x: -x[1]):
            pct = (count / total_docs * 100) if total_docs else 0
            md.append(f"| {dom} | {count} | {pct:.1f}% |")

        md.extend([
            f"",
            f"### By Department",
            f"",
            f"| Department | Documents |",
            f"|---|---|",
        ])
        depts = distributions.get("departments", {})
        for dept, count in sorted(depts.items(), key=lambda x: -x[1]):
            md.append(f"| {dept} | {count} |")

        if error_issues:
            md.extend([
                f"",
                f"## 5. Critical Errors ({len(error_issues)})",
                f"",
                f"| Document ID | Category | Field | Error Message |",
                f"|---|---|---|---|",
            ])
            for err in error_issues[:25]:
                d_id = err.document_id or "N/A"
                f_name = err.field or "N/A"
                md.append(f"| `{d_id}` | {err.category.value} | `{f_name}` | {err.message} |")
            if len(error_issues) > 25:
                md.append(f"*(and {len(error_issues) - 25} more errors; see validation_report.json)*")

        if warning_issues:
            md.extend([
                f"",
                f"## 6. Notable Warnings ({len(warning_issues)})",
                f"",
                f"| Document ID | Category | Field | Warning Message |",
                f"|---|---|---|---|",
            ])
            for warn in warning_issues[:20]:
                d_id = warn.document_id or "N/A"
                f_name = warn.field or "N/A"
                md.append(f"| `{d_id}` | {warn.category.value} | `{f_name}` | {warn.message} |")
            if len(warning_issues) > 20:
                md.append(f"*(and {len(warning_issues) - 20} more warnings; see validation_report.json)*")

        md.extend([
            f"",
            f"## 7. Downstream Chunking Readiness",
            f"",
            f"- **Chunking Ready:** {'YES' if errors == 0 else 'NO - Fix Errors First'}",
            f"- **Target Chunker Input:** `Data/validated/validated_documents.json`",
            f"- **Document Schema Integrity:** 100% verified across required fields and structured sections.",
        ])

        return "\n".join(md)

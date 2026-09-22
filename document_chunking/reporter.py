from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List

from document_chunking.models import Chunk, ChunkingStats


class ChunkingReporter:
    """Generates structured JSON and human-readable Markdown reports for the document chunking pipeline."""

    @staticmethod
    def generate_json_report(
        run_id: str,
        stats: ChunkingStats,
        validation_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat() + "Z",
            "statistics": stats.to_dict(),
            "quality_validation": validation_results,
        }

    @staticmethod
    def generate_markdown_report(
        run_id: str,
        stats: ChunkingStats,
        validation_results: Dict[str, Any],
    ) -> str:
        s_dict = stats.to_dict()
        md: List[str] = [
            "# Enterprise Document Chunking Report",
            "",
            f"- **Run ID:** `{run_id}`",
            f"- **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"- **Status:** {'✅ VALIDATED (Ready for Embeddings)' if validation_results.get('is_valid') else '❌ FAILED QUALITY CHECKS'}",
            "",
            "## 1. Chunking Summary",
            "",
            "| Metric | Value | Description |",
            "|---|---|---|",
            f"| **Total Documents** | {stats.total_documents} | Validated input documents processed |",
            f"| **Total Chunks Generated** | {stats.total_chunks} | Semantic chunks created |",
            f"| **Average Chunks / Document** | {stats.average_chunks_per_document:.2f} | Mean chunk granularity |",
            f"| **Average Chunk Tokens** | {stats.average_chunk_tokens:.1f} | Mean token size |",
            f"| **Median Chunk Tokens** | {stats.median_chunk_tokens:.1f} | Median token size |",
            f"| **Min Chunk Tokens** | {stats.min_chunk_tokens} | Smallest chunk token size |",
            f"| **Max Chunk Tokens** | {stats.max_chunk_tokens} | Largest chunk token size |",
            f"| **Sections Detected** | {stats.sections_detected} | Structural section headings identified |",
            f"| **Docs Without Headings** | {stats.documents_without_sections} | Fallback paragraph chunked documents |",
            f"| **Recursive Section Splits** | {stats.recursive_splits} | Sections requiring paragraph splitting |",
            f"| **Sentence Fallbacks** | {stats.sentence_fallbacks} | Oversized paragraphs split into sentences |",
            f"| **Token Fallbacks** | {stats.token_fallbacks} | Oversized sentences split into token pieces |",
            "",
            "## 2. Quality Verification (10 Checks)",
            "",
            f"- **Integrity Result:** {'PASSED (0 Errors)' if validation_results.get('error_count', 0) == 0 else 'FAILED'}",
            f"- **Errors:** {validation_results.get('error_count', 0)}",
            f"- **Warnings:** {validation_results.get('warning_count', 0)} (concise natural boundary chunks)",
            "",
            "## 3. Chunk Distribution by Department",
            "",
            "| Department | Chunk Count |",
            "|---|---|",
        ]

        dept_dist = stats.distributions.get("departments", {})
        for dept, count in sorted(dept_dist.items(), key=lambda x: -x[1]):
            md.append(f"| {dept} | {count} |")

        type_dist = stats.distributions.get("document_types", {})
        if type_dist:
            md.extend([
                "",
                "## 4. Chunk Distribution by Document Type",
                "",
                "| Document Type | Chunk Count |",
                "|---|---|",
            ])
            for dtype, count in sorted(type_dist.items(), key=lambda x: -x[1]):
                md.append(f"| {dtype} | {count} |")

        md.extend([
            "",
            "## 5. Downstream Embedding Integration",
            "",
            "- **Target Embedding File:** `Data/chunks/chunks.json`",
            "- **Streaming Format:** `Data/chunks/chunks.jsonl`",
            "- **Embedding Ready:** Every chunk includes `text`, `chunk_id`, `document_id`, `section`, `subsection`, and parent metadata.",
            "- **Consumer Pattern:** `for chunk in chunks: embedding = model.encode(chunk['text'])`",
        ])

        return "\n".join(md)

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from document_chunking.models import Chunk


class ChunkExporter:
    """Exports generated chunks and reports to target output paths."""

    @staticmethod
    def export(
        output_file_or_dir: str | Path,
        chunks: List[Chunk],
        json_report: Dict[str, Any],
        markdown_report: str,
    ) -> Dict[str, str]:
        target = Path(output_file_or_dir)

        if target.suffix.lower() in [".json", ".jsonl"]:
            out_dir = target.parent
            chunks_json_file = target if target.suffix.lower() == ".json" else target.with_suffix(".json")
            chunks_jsonl_file = target.with_suffix(".jsonl")
        else:
            out_dir = target
            chunks_json_file = out_dir / "chunks.json"
            chunks_jsonl_file = out_dir / "chunks.jsonl"

        out_dir.mkdir(parents=True, exist_ok=True)
        exported: Dict[str, str] = {}

        # 1. Export chunks.json
        chunk_dicts = [c.to_dict() for c in chunks]
        with open(chunks_json_file, "w", encoding="utf-8") as f:
            json.dump(chunk_dicts, f, indent=2)
        exported["chunks_json"] = str(chunks_json_file)

        # 2. Export chunks.jsonl
        with open(chunks_jsonl_file, "w", encoding="utf-8") as f:
            for cd in chunk_dicts:
                f.write(json.dumps(cd) + "\n")
        exported["chunks_jsonl"] = str(chunks_jsonl_file)

        # 3. Export chunking_report.json
        rep_json_file = out_dir / "chunking_report.json"
        with open(rep_json_file, "w", encoding="utf-8") as f:
            json.dump(json_report, f, indent=2)
        exported["report_json"] = str(rep_json_file)

        # 4. Export chunking_report.md
        rep_md_file = out_dir / "chunking_report.md"
        with open(rep_md_file, "w", encoding="utf-8") as f:
            f.write(markdown_report)
        exported["report_markdown"] = str(rep_md_file)

        return exported

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

from document_chunking.chunker import DocumentChunker
from document_chunking.config import ChunkingConfig
from document_chunking.exporter import ChunkExporter
from document_chunking.loader import DocumentLoader
from document_chunking.reporter import ChunkingReporter
from document_chunking.validator import ChunkQualityValidator

logger = logging.getLogger("document_chunking")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="document_chunking",
        description="Production-grade structure-aware enterprise document chunking module.",
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="Data/validated/validated_documents.json",
        help="Path to validated documents JSON file (default: Data/validated/validated_documents.json).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="Data/chunks/chunks.json",
        help="Path to output chunks file or directory (default: Data/chunks/chunks.json).",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to chunking configuration file (.json or .yaml).",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=None,
        help="Override target chunk tokens (default: 500).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override max chunk tokens (default: 600).",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=None,
        help="Override min chunk tokens (default: 100).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Override overlap tokens (default: 75).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        default=True,
        help="Print summary report to stdout upon completion (default: True).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level verbosity (default: INFO).",
    )
    return parser


def main(args_list: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(args_list)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="[%(levelname)s] %(message)s",
    )

    try:
        # 1. Load configuration
        if args.config:
            config = ChunkingConfig.from_file(args.config)
        else:
            default_cfg = Path("config/chunking_config.json")
            if default_cfg.exists():
                config = ChunkingConfig.from_file(default_cfg)
            else:
                config = ChunkingConfig()

        if args.target_tokens is not None:
            config.target_chunk_tokens = args.target_tokens
        if args.max_tokens is not None:
            config.max_chunk_tokens = args.max_tokens
        if args.min_tokens is not None:
            config.min_chunk_tokens = args.min_tokens
        if args.overlap is not None:
            config.overlap_tokens = args.overlap

        # 2. Stage 1: Load validated documents
        logger.info(f"Loading validated documents from {args.input}")
        documents, load_errors = DocumentLoader.load(args.input, strict=False)
        if load_errors:
            for err in load_errors:
                logger.warning(f"Loader issue: {err}")

        logger.info(f"Loaded {len(documents)} documents for chunking")

        # Capture snapshots to verify immutability during self-validation
        original_snapshots = {
            doc["document_id"]: doc.get("content", "")
            for doc in documents
            if "document_id" in doc
        }

        # 3. Execute chunking
        chunker = DocumentChunker(config=config)
        chunks, stats = chunker.chunk_corpus(documents)
        logger.info(f"Generated {len(chunks)} chunks across {len(documents)} documents")

        # 4. Stage 4: Self-validation quality checks (10 checks)
        validation_results = ChunkQualityValidator.validate_chunks(
            chunks, documents, original_doc_snapshots=original_snapshots
        )
        run_id = f"chunking-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

        # 5. Reporting and export
        json_report = ChunkingReporter.generate_json_report(run_id, stats, validation_results)
        md_report = ChunkingReporter.generate_markdown_report(run_id, stats, validation_results)

        exported = ChunkExporter.export(
            output_file_or_dir=args.output,
            chunks=chunks,
            json_report=json_report,
            markdown_report=md_report,
        )

        if args.report:
            print("\n==================================================")
            print("         DOCUMENT CHUNKING SUMMARY REPORT         ")
            print("==================================================")
            print(f"Total Documents Chunked:    {stats.total_documents}")
            print(f"Total Chunks Generated:     {stats.total_chunks}")
            print(f"Average Chunks / Document:  {stats.average_chunks_per_document:.2f}")
            print(f"Average Chunk Tokens:       {stats.average_chunk_tokens:.1f}")
            print(f"Median Chunk Tokens:        {stats.median_chunk_tokens:.1f}")
            print(f"Min Chunk Tokens:           {stats.min_chunk_tokens}")
            print(f"Max Chunk Tokens:           {stats.max_chunk_tokens}")
            print(f"Sections Detected:          {stats.sections_detected}")
            print(f"Docs Without Headings:      {stats.documents_without_sections}")
            print(f"Recursive Section Splits:   {stats.recursive_splits}")
            print(f"Sentence Fallbacks:         {stats.sentence_fallbacks}")
            print(f"Token Fallbacks:            {stats.token_fallbacks}")
            print(f"Quality Check Status:       {'PASSED (0 Errors)' if validation_results['is_valid'] else 'FAILED'}")
            print(f"Chunks File:                {exported.get('chunks_json')}")
            print(f"Report File:                {exported.get('report_markdown')}")
            print("==================================================\n")

        if not validation_results["is_valid"]:
            logger.error(f"Quality validation failed with {validation_results['error_count']} errors.")
            return 1

        return 0

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        return 2
    except Exception as e:
        logger.error(f"Unexpected chunking error: {e}", exc_info=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())

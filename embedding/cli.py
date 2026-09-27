from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from embedding.config import IngestionConfig
from embedding.ingest import IngestionPipeline


def setup_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="embedding_ingest",
        description="Ingestion pipeline for BGE embedding generation and Supabase storage.",
    )
    parser.add_argument(
        "-i", "--chunks",
        type=str,
        default="Data/chunks/chunks.jsonl",
        help="Path to chunks.jsonl file (default: Data/chunks/chunks.jsonl)",
    )
    parser.add_argument(
        "-v", "--validated-docs",
        type=str,
        default="Data/validated/validated_documents.json",
        help="Path to validated_documents.json (default: Data/validated/validated_documents.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="Embedding model name (default: BAAI/bge-small-en-v1.5)",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Batch size for embedding inference (default: 32)",
    )
    parser.add_argument(
        "--upsert-batch-size",
        type=int,
        default=50,
        help="Batch size for Supabase database upserts (default: 50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and generate embeddings without sending HTTP requests to Supabase.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser


def main(args: list[str] | None = None) -> int:
    parser = build_parser()
    parsed_args = parser.parse_args(args)
    setup_logging(parsed_args.log_level)

    config = IngestionConfig.from_env(
        input_chunks_path=Path(parsed_args.chunks),
        validated_docs_path=Path(parsed_args.validated_docs),
        embedding_model_name=parsed_args.model,
        embedding_batch_size=parsed_args.embedding_batch_size,
        db_batch_size=parsed_args.upsert_batch_size,
        # None (flag absent) defers to the DRY_RUN env var instead of forcing False.
        dry_run=True if parsed_args.dry_run else None,
    )

    pipeline = IngestionPipeline(config=config)
    try:
        results = pipeline.run()
        print("\n" + "=" * 50)
        print("          INGESTION PIPELINE SUMMARY")
        print("=" * 50)
        print(f"Status:            {results.get('status')}")
        print(f"Model:             {results.get('model')}")
        print(f"Dimension:         {results.get('dimension')}")
        print(f"Documents:         {results.get('documents_count')}")
        print(f"Chunks:            {results.get('chunks_count')}")
        print(f"Embeddings:        {results.get('embeddings_count')}")
        print(f"Elapsed Seconds:   {results.get('elapsed_seconds')}s")
        print("=" * 50)
        return 0
    except Exception as exc:
        logging.getLogger("embedding.cli").error("Ingestion failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

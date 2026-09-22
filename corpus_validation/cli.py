from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from corpus_validation.config import ValidationConfig
from corpus_validation.validator import CorpusValidator

logger = logging.getLogger("corpus_validation")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="corpus_validation",
        description="Production-grade enterprise corpus validation module for search & query reformulation pipelines.",
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=str,
        help="Path to input corpus (directory containing JSON/JSONL files, single JSON file, or JSONL file).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="Data/validated",
        help="Directory to save validated documents, rejected documents, and validation reports (default: Data/validated).",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to validation configuration file (.json or .yaml).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: treat warnings as failures (exits with code 1 if warnings are detected).",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        default=True,
        help="Exit with code 1 if any validation errors or rejected documents are detected (default: True).",
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
        if args.config:
            config = ValidationConfig.from_file(args.config)
        else:
            # Check default config location
            default_cfg = Path("config/validation_config.json")
            if default_cfg.exists():
                config = ValidationConfig.from_file(default_cfg)
            else:
                config = ValidationConfig()

        validator = CorpusValidator(config=config)
        valid_docs, rejected_docs, issues, report = validator.validate_corpus(
            input_path=args.input,
            output_dir=args.output,
        )

        summary = report.get("summary", {})
        errors_count = summary.get("errors", 0)
        warnings_count = summary.get("warnings", 0)

        if args.report:
            print("\n==================================================")
            print("         CORPUS VALIDATION SUMMARY REPORT         ")
            print("==================================================")
            print(f"Total Documents Evaluated: {summary.get('total_documents', 0)}")
            print(f"Valid Documents:           {summary.get('valid_documents', 0)}")
            print(f"Rejected Documents:        {summary.get('rejected_documents', 0)}")
            print(f"Errors:                    {errors_count}")
            print(f"Warnings:                  {warnings_count}")
            print(f"Informational Notes:       {summary.get('infos', 0)}")
            print(f"Exact Duplicate Groups:    {report.get('duplicates', {}).get('exact_duplicate_count', 0)}")
            print(f"Near Duplicate Pairs:      {report.get('duplicates', {}).get('near_duplicate_count', 0)}")
            print(f"Artifacts Saved To:        {Path(args.output).resolve()}")
            print("==================================================\n")

        # Determine exit code
        if errors_count > 0 and args.fail_on_error:
            logger.error(f"Validation finished with {errors_count} critical errors.")
            return 1

        if args.strict and warnings_count > 0:
            logger.warning(f"Strict mode enabled: failing due to {warnings_count} warnings.")
            return 1

        logger.info("Corpus successfully validated. Downstream chunking pipeline can proceed.")
        return 0

    except FileNotFoundError as e:
        logger.error(f"Configuration or input path error: {e}")
        return 2
    except Exception as e:
        logger.error(f"Unexpected runtime error during corpus validation: {e}", exc_info=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())

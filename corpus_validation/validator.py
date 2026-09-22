from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from corpus_validation.aggregator import ValidationAggregator
from corpus_validation.config import ValidationConfig
from corpus_validation.exporter import CorpusExporter
from corpus_validation.loader import CorpusLoader
from corpus_validation.models import Document, ValidationIssue
from corpus_validation.reporter import ValidationReporter

logger = logging.getLogger(__name__)


class CorpusValidator:
    """Main facade for running end-to-end enterprise corpus validation."""

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.aggregator = ValidationAggregator(self.config)

    def validate_corpus(
        self,
        input_path: str | Path,
        output_dir: Optional[str | Path] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[List[Document], List[Document], List[ValidationIssue], Dict[str, Any]]:
        run_id = run_id or f"validation-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        logger.info(f"Initializing corpus validation run: {run_id}")

        # 1. Load documents
        raw_documents, load_issues = CorpusLoader.load(input_path)

        # 2. Run validation pipeline
        valid_docs, rejected_docs, all_issues, metadata_report = self.aggregator.run(
            documents=raw_documents,
            initial_issues=load_issues,
        )

        # 3. Generate reports
        json_report = ValidationReporter.generate_json_report(
            run_id=run_id,
            issues=all_issues,
            metadata_report=metadata_report,
        )
        md_report = ValidationReporter.generate_markdown_report(
            run_id=run_id,
            issues=all_issues,
            metadata_report=metadata_report,
        )

        # 4. Export artifacts if output_dir provided
        if output_dir:
            logger.info(f"Exporting validated corpus and reports to {output_dir}")
            CorpusExporter.export(
                output_dir=output_dir,
                valid_documents=valid_docs,
                rejected_documents=rejected_docs,
                json_report=json_report,
                markdown_report=md_report,
                issues=all_issues,
            )

        return valid_docs, rejected_docs, all_issues, json_report

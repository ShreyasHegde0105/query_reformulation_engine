from __future__ import annotations

import logging
from typing import Any, Dict, List, Set, Tuple

from corpus_validation.config import ValidationConfig
from corpus_validation.models import (
    Document,
    DuplicateGroup,
    NearDuplicatePair,
    QualityMetrics,
    Severity,
    ValidationCategory,
    ValidationIssue,
    ValidationSummary,
)
from corpus_validation.validators import (
    ContentValidator,
    CoverageValidator,
    DuplicateDetector,
    EntityValidator,
    GroundTruthValidator,
    IdValidator,
    InternalConsistencyValidator,
    MetadataValidator,
    QualityValidator,
    SchemaValidator,
)

logger = logging.getLogger(__name__)


class ValidationAggregator:
    """Coordinates modular validators, aggregates validation issues, and partitions valid vs rejected documents."""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.schema_validator = SchemaValidator(config)
        self.id_validator = IdValidator(config)
        self.content_validator = ContentValidator(config)
        self.duplicate_detector = DuplicateDetector(config)
        self.metadata_validator = MetadataValidator(config)
        self.entity_validator = EntityValidator(config)
        self.consistency_validator = InternalConsistencyValidator(config)
        self.quality_validator = QualityValidator(config)
        self.coverage_validator = CoverageValidator(config)
        self.ground_truth_validator = GroundTruthValidator(config)

    def run(
        self, documents: List[Document], initial_issues: List[ValidationIssue] | None = None
    ) -> Tuple[List[Document], List[Document], List[ValidationIssue], Dict[str, Any]]:
        all_issues: List[ValidationIssue] = list(initial_issues or [])

        logger.info(f"Starting validation pipeline for {len(documents)} documents")

        # 1. Schema Validation
        logger.info("Running Stage 1: Schema Validation")
        schema_issues = self.schema_validator.validate(documents)
        all_issues.extend(schema_issues)

        # 2. Document ID Validation
        logger.info("Running Stage 2: Document ID Validation")
        id_issues = self.id_validator.validate(documents)
        all_issues.extend(id_issues)

        # 3. Content Validation
        logger.info("Running Stage 3: Content Validation")
        content_issues = self.content_validator.validate(documents)
        all_issues.extend(content_issues)

        # 4. Duplicate Detection (Exact & Near)
        logger.info("Running Stage 4 & 5: Duplicate Detection")
        dup_issues = self.duplicate_detector.validate(documents)
        all_issues.extend(dup_issues)

        # 5. Metadata Validation
        logger.info("Running Stage 6: Metadata Validation")
        meta_issues = self.metadata_validator.validate(documents)
        all_issues.extend(meta_issues)

        # 6. Entity Consistency
        logger.info("Running Stage 7: Entity Consistency Validation")
        entity_issues = self.entity_validator.validate(documents)
        all_issues.extend(entity_issues)

        # 7. Internal Consistency
        logger.info("Running Stage 8: Internal Consistency Validation")
        consistency_issues = self.consistency_validator.validate(documents)
        all_issues.extend(consistency_issues)

        # 8. Enterprise Search Quality
        logger.info("Running Stage 9: Enterprise Search Quality Validation")
        quality_issues = self.quality_validator.validate(documents)
        all_issues.extend(quality_issues)

        # 9. Coverage Analysis
        logger.info("Running Stage 10: Coverage Analysis")
        coverage_issues = self.coverage_validator.validate(documents)
        all_issues.extend(coverage_issues)

        # 10. Ground Truth Reconciliation
        logger.info("Running Stage 11: Ground Truth Reconciliation")
        gt_issues = self.ground_truth_validator.validate(documents)
        all_issues.extend(gt_issues)

        # Determine rejected documents based on ERROR severity issues
        error_doc_ids: Set[str] = {
            issue.document_id
            for issue in all_issues
            if issue.severity == Severity.ERROR and issue.document_id is not None
        }

        # If configured to filter exact duplicates from the validated corpus
        if self.config.remove_exact_duplicates:
            for group in self.duplicate_detector.exact_duplicate_groups:
                for dup_id in group.documents:
                    if dup_id != group.canonical_document_id:
                        error_doc_ids.add(dup_id)

        valid_documents: List[Document] = []
        rejected_documents: List[Document] = []

        for doc in documents:
            doc_id = doc.document_id
            if doc_id and doc_id in error_doc_ids:
                rejected_documents.append(doc)
            elif not doc_id:
                rejected_documents.append(doc)
            else:
                valid_documents.append(doc)

        # Count issues
        errors_count = sum(1 for i in all_issues if i.severity == Severity.ERROR)
        warnings_count = sum(1 for i in all_issues if i.severity == Severity.WARNING)
        infos_count = sum(1 for i in all_issues if i.severity == Severity.INFO)

        summary = ValidationSummary(
            total_documents=len(documents),
            valid_documents=len(valid_documents),
            rejected_documents=len(rejected_documents),
            warnings=warnings_count,
            errors=errors_count,
            infos=infos_count,
        )

        metadata_report: Dict[str, Any] = {
            "summary": summary.to_dict(),
            "id_stats": self.id_validator.stats,
            "exact_duplicate_groups": [g.to_dict() for g in self.duplicate_detector.exact_duplicate_groups],
            "near_duplicate_pairs": [p.to_dict() for p in self.duplicate_detector.near_duplicate_pairs],
            "quality_metrics": self.coverage_validator.quality_metrics.to_dict(),
            "distributions": self.coverage_validator.distributions,
        }

        logger.info(
            f"Validation complete: {len(valid_documents)} valid, {len(rejected_documents)} rejected, "
            f"{errors_count} errors, {warnings_count} warnings, {infos_count} infos"
        )

        return valid_documents, rejected_documents, all_issues, metadata_report

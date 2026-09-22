from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set

from corpus_validation.config import ValidationConfig
from corpus_validation.models import Document, Severity, ValidationCategory, ValidationIssue
from corpus_validation.validators.base import BaseValidator


class GroundTruthValidator(BaseValidator):
    """Validates corpus documents against the authoritative University World Model."""

    def __init__(self, config: ValidationConfig):
        super().__init__(config)
        self.world_model: Dict[str, Any] = {}
        self.known_dept_ids: Set[str] = set()
        self.known_dept_names: Set[str] = set()
        self.known_systems: Set[str] = set()
        self.known_roles: Set[str] = set()
        self.canonical_doc_ids: Set[str] = set()
        self._load_world_model()

    def _load_world_model(self) -> None:
        if not self.config.world_model_path:
            return

        wm_path = Path(self.config.world_model_path)
        if not wm_path.exists():
            return

        try:
            with open(wm_path, "r", encoding="utf-8") as f:
                self.world_model = json.load(f)

            # Extract departments
            for dept in self.world_model.get("departments", []):
                if isinstance(dept, dict):
                    if "department_id" in dept:
                        self.known_dept_ids.add(dept["department_id"])
                    if "name" in dept:
                        self.known_dept_names.add(dept["name"])

            # Extract systems
            for sys in self.world_model.get("systems", []):
                if isinstance(sys, dict) and "name" in sys:
                    self.known_systems.add(sys["name"])

            # Extract roles
            for role in self.world_model.get("roles", []):
                if isinstance(role, dict) and "name" in role:
                    self.known_roles.add(role["name"])

            # Extract canonical documents from clusters
            ecosystem = self.world_model.get("document_ecosystem", {})
            clusters = ecosystem.get("document_clusters", [])
            for cluster in clusters:
                for doc in cluster.get("canonical_documents", []):
                    if isinstance(doc, dict) and "document_id" in doc:
                        self.canonical_doc_ids.add(doc["document_id"])
        except Exception:
            pass

    def validate(self, documents: List[Document]) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        if not self.world_model:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.GROUND_TRUTH,
                    severity=Severity.INFO,
                    message=f"World model ground-truth file not found or empty at {self.config.world_model_path}; skipping world-model reconciliation",
                )
            )
            return issues

        corpus_doc_ids = {doc.document_id for doc in documents if doc.document_id}

        # 1. Canonical document coverage
        if self.canonical_doc_ids:
            missing_canonical = self.canonical_doc_ids - corpus_doc_ids
            for missing_id in sorted(list(missing_canonical)):
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.GROUND_TRUTH,
                        severity=Severity.WARNING,
                        document_id=missing_id,
                        field="document_id",
                        message=f"Canonical document '{missing_id}' specified in world model cluster is missing from corpus",
                    )
                )

        # 2. Document-level checks against world model
        for doc in documents:
            doc_id = doc.document_id

            # Department ID check
            if doc.department_id and self.known_dept_ids:
                if doc.department_id not in self.known_dept_ids:
                    issues.append(
                        ValidationIssue(
                            category=ValidationCategory.GROUND_TRUTH,
                            severity=Severity.WARNING,
                            document_id=doc_id,
                            field="department_id",
                            message=f"Department ID '{doc.department_id}' is not in authoritative world model departments",
                            details={"department_id": doc.department_id},
                        )
                    )

            # Systems check
            if doc.systems and self.known_systems:
                for sys_name in doc.systems:
                    if sys_name and sys_name not in self.known_systems:
                        # Check partial or case-insensitive match
                        matched = any(sys_name.lower() == ks.lower() for ks in self.known_systems)
                        if not matched:
                            issues.append(
                                ValidationIssue(
                                    category=ValidationCategory.GROUND_TRUTH,
                                    severity=Severity.INFO,
                                    document_id=doc_id,
                                    field="systems",
                                    message=f"System '{sys_name}' is not registered in authoritative world model systems catalog",
                                    details={"system": sys_name},
                                )
                            )

        return issues

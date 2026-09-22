from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Set

from corpus_validation.config import ValidationConfig
from corpus_validation.models import Document, Severity, ValidationCategory, ValidationIssue
from corpus_validation.validators.base import BaseValidator

AMBIGUOUS_ACRONYMS = {
    "NOC": "No-Objection Certificate / No-Dues Clearance",
    "TC": "Transfer Certificate / Terms & Conditions",
    "PR": "Public Relations / Placement Registration / Purchase Requisition",
    "EC": "Executive Council / Electronic City Campus / Exam Center",
    "AC": "Academic Council / Air Conditioning",
    "SAC": "Senate Academic Committee / Student Activity Centre",
    "IA": "Industry Advisory Board / Internal Assessment",
    "FT": "Field Trip / Full Time",
    "OD": "On-Duty Leave / Overdue",
    "MC": "Medical Condonation / Main Campus",
    "ID": "Student Identity Card / System Identifier",
    "DOR": "Date of Registration / Director of Research",
    "LMS": "Learning Management System / Leave Management System",
}


class EntityValidator(BaseValidator):
    """Validates cross-document entity consistency (department mappings, system names, and polysemous acronyms)."""

    def validate(self, documents: List[Document]) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []

        # 1. Department ID -> Department Name consistency across documents
        dept_id_to_names: Dict[str, Set[str]] = defaultdict(set)
        for doc in documents:
            if doc.department_id and doc.department:
                dept_id_to_names[doc.department_id.strip()].add(doc.department.strip())

        for d_id, names in dept_id_to_names.items():
            if len(names) > 1:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.ENTITY,
                        severity=Severity.WARNING,
                        field="department",
                        message=f"Department ID '{d_id}' maps inconsistently to multiple department names: {sorted(list(names))}",
                        details={"department_id": d_id, "conflicting_names": sorted(list(names))},
                    )
                )

        # 2. Cross-document system names casing / variations
        system_casing_map: Dict[str, Set[str]] = defaultdict(set)
        for doc in documents:
            for sys in doc.systems:
                if sys and sys.strip():
                    cleaned_sys = sys.strip()
                    system_casing_map[cleaned_sys.lower()].add(cleaned_sys)

        for lower_sys, forms in system_casing_map.items():
            if len(forms) > 1:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.ENTITY,
                        severity=Severity.INFO,
                        field="systems",
                        message=f"System name appears with multiple casing variations across corpus: {sorted(list(forms))}",
                        details={"system_variations": sorted(list(forms))},
                    )
                )

        # 3. Controlled Polysemy & Ambiguous Acronym check
        for doc in documents:
            content = doc.content or ""
            doc_id = doc.document_id
            for acronym, desc in AMBIGUOUS_ACRONYMS.items():
                pattern = rf"\b{acronym}\b"
                if re.search(pattern, content):
                    # Acronym appears in document content. Check if the document has disambiguation context
                    if acronym not in doc.keywords and acronym not in " ".join(doc.retrieval_signals):
                        issues.append(
                            ValidationIssue(
                                category=ValidationCategory.ENTITY,
                                severity=Severity.INFO,
                                document_id=doc_id,
                                field="content",
                                message=f"Document uses ambiguous acronym '{acronym}' ({desc}); consider adding it to keywords or retrieval_signals for search disambiguation",
                                details={"acronym": acronym, "context": desc},
                            )
                        )

        return issues

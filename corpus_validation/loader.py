from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from corpus_validation.models import Document, Severity, ValidationCategory, ValidationIssue

logger = logging.getLogger(__name__)


class CorpusLoader:
    """Loads enterprise documents from directories, JSON files, or JSONL files."""

    @staticmethod
    def load(source_path: str | Path) -> Tuple[List[Document], List[ValidationIssue]]:
        path = Path(source_path)
        documents: List[Document] = []
        issues: List[ValidationIssue] = []

        if not path.exists():
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.SCHEMA,
                    severity=Severity.ERROR,
                    message=f"Corpus source path does not exist: {source_path}",
                )
            )
            return documents, issues

        if path.is_dir():
            json_files = sorted(list(path.glob("*.json")) + list(path.glob("*.jsonl")))
            if not json_files:
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.SCHEMA,
                        severity=Severity.WARNING,
                        message=f"No .json or .jsonl files found in directory: {source_path}",
                    )
                )
                return documents, issues

            for f in json_files:
                sub_docs, sub_issues = CorpusLoader._load_file(f)
                documents.extend(sub_docs)
                issues.extend(sub_issues)

        elif path.is_file():
            sub_docs, sub_issues = CorpusLoader._load_file(path)
            documents.extend(sub_docs)
            issues.extend(sub_issues)

        logger.info(f"Loaded {len(documents)} documents from {source_path} ({len(issues)} load issues)")
        return documents, issues

    @staticmethod
    def _load_file(file_path: Path) -> Tuple[List[Document], List[ValidationIssue]]:
        documents: List[Document] = []
        issues: List[ValidationIssue] = []

        if file_path.suffix.lower() == ".jsonl":
            return CorpusLoader._load_jsonl(file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = json.load(f)
        except json.JSONDecodeError as e:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.SCHEMA,
                    severity=Severity.ERROR,
                    message=f"Malformed JSON in file {file_path.name}: {str(e)}",
                    details={"file": str(file_path), "error": str(e)},
                )
            )
            return documents, issues
        except Exception as e:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.SCHEMA,
                    severity=Severity.ERROR,
                    message=f"Error reading file {file_path.name}: {str(e)}",
                    details={"file": str(file_path), "error": str(e)},
                )
            )
            return documents, issues

        domain = None
        if isinstance(content, dict):
            domain = content.get("domain")
            raw_docs = content.get("documents", [])
            if not isinstance(raw_docs, list):
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.SCHEMA,
                        severity=Severity.ERROR,
                        message=f"Expected 'documents' to be a list in {file_path.name}, got {type(raw_docs).__name__}",
                        details={"file": str(file_path)},
                    )
                )
                return documents, issues
        elif isinstance(content, list):
            raw_docs = content
        else:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.SCHEMA,
                    severity=Severity.ERROR,
                    message=f"Unexpected root JSON structure in {file_path.name}: expected dict or list, got {type(content).__name__}",
                    details={"file": str(file_path)},
                )
            )
            return documents, issues

        for idx, doc_raw in enumerate(raw_docs):
            if not isinstance(doc_raw, dict):
                issues.append(
                    ValidationIssue(
                        category=ValidationCategory.SCHEMA,
                        severity=Severity.ERROR,
                        message=f"Document at index {idx} in {file_path.name} is not a valid JSON object",
                        details={"file": str(file_path), "index": idx},
                    )
                )
                continue

            doc = Document(raw=doc_raw, source_file=str(file_path), domain=domain)
            documents.append(doc)

        return documents, issues

    @staticmethod
    def _load_jsonl(file_path: Path) -> Tuple[List[Document], List[ValidationIssue]]:
        documents: List[Document] = []
        issues: List[ValidationIssue] = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        doc_raw = json.loads(line_str)
                        if not isinstance(doc_raw, dict):
                            issues.append(
                                ValidationIssue(
                                    category=ValidationCategory.SCHEMA,
                                    severity=Severity.ERROR,
                                    message=f"Line {line_no} in {file_path.name} is not a JSON object",
                                    details={"file": str(file_path), "line": line_no},
                                )
                            )
                            continue
                        doc = Document(raw=doc_raw, source_file=str(file_path))
                        documents.append(doc)
                    except json.JSONDecodeError as e:
                        issues.append(
                            ValidationIssue(
                                category=ValidationCategory.SCHEMA,
                                severity=Severity.ERROR,
                                message=f"Malformed JSON at line {line_no} in {file_path.name}: {str(e)}",
                                details={"file": str(file_path), "line": line_no, "error": str(e)},
                            )
                        )
        except Exception as e:
            issues.append(
                ValidationIssue(
                    category=ValidationCategory.SCHEMA,
                    severity=Severity.ERROR,
                    message=f"Error reading JSONL file {file_path.name}: {str(e)}",
                    details={"file": str(file_path), "error": str(e)},
                )
            )

        return documents, issues

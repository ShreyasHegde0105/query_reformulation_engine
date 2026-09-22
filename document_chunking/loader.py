from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

try:
    from corpus_validation.models import Document as ValidatedDocument
except ImportError:
    ValidatedDocument = None  # type: ignore

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads and validates input documents from the validated corpus repository.
    
    Supports reading:
    - Root dict containing a 'documents' list (e.g. Data/validated/validated_documents.json)
    - Root list of documents
    - JSONL files containing one document JSON object per line
    
    Validates required fields:
    - document_id
    - title
    - content
    
    Fails gracefully if malformed input is encountered without modifying source data.
    """

    @classmethod
    def load(
        cls,
        source_path: Union[str, Path],
        strict: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Load documents from file and validate required schema fields.
        
        Args:
            source_path: Path to the validated corpus JSON or JSONL file.
            strict: If True, raises ValueError upon encountering a malformed document.
            
        Returns:
            Tuple of (valid_documents_list, list_of_error_messages).
        """
        p = Path(source_path)
        if not p.exists():
            raise FileNotFoundError(f"Validated corpus file not found: {source_path}")

        documents: List[Dict[str, Any]] = []
        errors: List[str] = []

        if p.suffix.lower() == ".jsonl":
            raw_items = cls._load_jsonl(p, errors)
        else:
            raw_items = cls._load_json(p, errors)

        for idx, item in enumerate(raw_items):
            # Extract underlying dict if wrapped in a Document model
            if ValidatedDocument is not None and isinstance(item, ValidatedDocument):
                doc_dict = item.raw
            elif isinstance(item, dict):
                doc_dict = item
            else:
                msg = f"Item at index {idx} in {p.name} is not a valid document object: {type(item).__name__}"
                errors.append(msg)
                if strict:
                    raise ValueError(msg)
                continue

            # Validate required core fields
            doc_id = doc_dict.get("document_id")
            title = doc_dict.get("title")
            content = doc_dict.get("content")

            if not doc_id or not str(doc_id).strip():
                msg = f"Document at index {idx} in {p.name} is missing required 'document_id'"
                errors.append(msg)
                if strict:
                    raise ValueError(msg)
                continue

            if title is None:
                msg = f"Document '{doc_id}' is missing required 'title'"
                errors.append(msg)
                if strict:
                    raise ValueError(msg)
                continue

            if content is None:
                msg = f"Document '{doc_id}' is missing required 'content'"
                errors.append(msg)
                if strict:
                    raise ValueError(msg)
                continue

            documents.append(doc_dict)

        logger.info(
            f"Successfully loaded {len(documents)} validated documents from {source_path} ({len(errors)} issues)"
        )
        return documents, errors

    @classmethod
    def _load_json(cls, file_path: Path, errors: List[str]) -> List[Any]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"Malformed JSON in {file_path.name}: {e}")
            return []
        except Exception as e:
            errors.append(f"Failed to read {file_path.name}: {e}")
            return []

        if isinstance(data, dict):
            if "documents" in data and isinstance(data["documents"], list):
                return data["documents"]
            return [data]
        elif isinstance(data, list):
            return data
        else:
            errors.append(f"Unexpected JSON root structure in {file_path.name}: expected dict or list, got {type(data).__name__}")
            return []

    @classmethod
    def _load_jsonl(cls, file_path: Path, errors: List[str]) -> List[Any]:
        items: List[Any] = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        items.append(json.loads(line_str))
                    except json.JSONDecodeError as e:
                        errors.append(f"Malformed JSON at line {line_no} in {file_path.name}: {e}")
        except Exception as e:
            errors.append(f"Failed to read JSONL file {file_path.name}: {e}")
        return items

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ValidationConfig:
    min_content_length: int = 200
    max_content_length: int = 50000
    min_word_count: int = 30
    max_word_count: int = 10000
    max_whitespace_ratio: float = 0.35
    max_repeated_ngram_ratio: float = 0.40
    near_duplicate_threshold: float = 0.88
    enable_near_duplicate_detection: bool = True
    fail_on_schema_error: bool = True
    remove_exact_duplicates: bool = False
    allowed_statuses: List[str] = field(default_factory=lambda: [
        "Active", "Draft", "Deprecated", "Archived", "Under Review"
    ])
    required_document_fields: List[str] = field(default_factory=lambda: [
        "document_id", "title", "content", "department", "department_id", "document_type"
    ])
    required_metadata_fields: List[str] = field(default_factory=lambda: [
        "version", "effective_date", "status", "audience", "keywords",
        "retrieval_signals", "systems", "roles", "related_documents"
    ])
    world_model_path: Optional[str] = "Data/demo_university_world_model.json"
    id_pattern: str = r"^[A-Z0-9]+(-[A-Z0-9]+)+$"
    placeholder_patterns: List[str] = field(default_factory=lambda: [
        "lorem ipsum", "placeholder", "sample text", "todo:", "tbd",
        r"\[insert", "<insert", "xxx+", "replace this"
    ])

    @classmethod
    def from_file(cls, path: str | Path) -> ValidationConfig:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(p, "r", encoding="utf-8") as f:
            if p.suffix.lower() in [".yaml", ".yml"]:
                try:
                    import yaml  # type: ignore
                    data = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML is required to parse .yaml configuration files. Use JSON or install pyyaml.")
            else:
                data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ValidationConfig:
        config_kwargs = {}
        for f in cls.__dataclass_fields__:
            if f in data:
                config_kwargs[f] = data[f]
        return cls(**config_kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_content_length": self.min_content_length,
            "max_content_length": self.max_content_length,
            "min_word_count": self.min_word_count,
            "max_word_count": self.max_word_count,
            "max_whitespace_ratio": self.max_whitespace_ratio,
            "max_repeated_ngram_ratio": self.max_repeated_ngram_ratio,
            "near_duplicate_threshold": self.near_duplicate_threshold,
            "enable_near_duplicate_detection": self.enable_near_duplicate_detection,
            "fail_on_schema_error": self.fail_on_schema_error,
            "remove_exact_duplicates": self.remove_exact_duplicates,
            "allowed_statuses": self.allowed_statuses,
            "required_document_fields": self.required_document_fields,
            "required_metadata_fields": self.required_metadata_fields,
            "world_model_path": self.world_model_path,
            "id_pattern": self.id_pattern,
            "placeholder_patterns": self.placeholder_patterns,
        }

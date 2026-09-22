from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict


@dataclasses.dataclass
class ChunkingConfig:
    target_chunk_tokens: int = 500
    max_chunk_tokens: int = 600
    min_chunk_tokens: int = 100
    overlap_tokens: int = 75
    include_section_heading_in_text: bool = True
    detect_numbered_sections: bool = True
    detect_subsections: bool = True
    enable_paragraph_split: bool = True
    enable_sentence_split: bool = True
    enable_token_fallback: bool = True
    section_heading_pattern: str = r"^(?:#{1,6}\s+)?\s*(\d+)\.\s+([^\n\r]+?)$"
    subsection_heading_pattern: str = r"^(?:#{1,6}\s+)?\s*(\d+\.\d+)\s+([^:\n\r]+)(?::|\s*-|\n|$)"

    @classmethod
    def from_file(cls, path: str | Path) -> ChunkingConfig:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Chunking configuration file not found: {path}")

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
    def from_dict(cls, data: Dict[str, Any]) -> ChunkingConfig:
        config_kwargs = {}
        for f in cls.__dataclass_fields__:
            if f in data:
                config_kwargs[f] = data[f]
        return cls(**config_kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_chunk_tokens": self.target_chunk_tokens,
            "max_chunk_tokens": self.max_chunk_tokens,
            "min_chunk_tokens": self.min_chunk_tokens,
            "overlap_tokens": self.overlap_tokens,
            "include_section_heading_in_text": self.include_section_heading_in_text,
            "detect_numbered_sections": self.detect_numbered_sections,
            "detect_subsections": self.detect_subsections,
            "enable_paragraph_split": self.enable_paragraph_split,
            "enable_sentence_split": self.enable_sentence_split,
            "enable_token_fallback": self.enable_token_fallback,
            "section_heading_pattern": self.section_heading_pattern,
            "subsection_heading_pattern": self.subsection_heading_pattern,
        }

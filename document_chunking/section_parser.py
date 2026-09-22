from __future__ import annotations

import re
from typing import List, Optional, Tuple

from document_chunking.config import ChunkingConfig
from document_chunking.models import Section, SubsectionInfo
from document_chunking.normalizer import TextNormalizer


class SectionParser:
    """Parses structured enterprise documents into numbered sections and subsections.
    
    Adheres strictly to the document structure present in the corpus.
    Never invents artificial section names.
    Supports configurable heading and subsection patterns.
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config

        pattern = config.section_heading_pattern or r"^(?:#{1,6}\s+)?\s*(\d+)\.\s+([^\n\r]+?)$"
        try:
            self.section_regex = re.compile(pattern, re.MULTILINE)
        except re.error:
            self.section_regex = re.compile(
                r"^(?:#{1,6}\s+)?\s*(\d+)\.\s+([^\n\r]+?)$", re.MULTILINE
            )

        sub_pattern = config.subsection_heading_pattern or r"^(?:#{1,6}\s+)?\s*(\d+\.\d+)\s+([^:\n\r]+)(?::|\s*-|\n|$)"
        try:
            self.subsection_regex = re.compile(sub_pattern, re.MULTILINE)
        except re.error:
            self.subsection_regex = re.compile(
                r"^(?:#{1,6}\s+)?\s*(\d+\.\d+)\s+([^:\n\r]+)(?::|\s*-|\n|$)", re.MULTILINE
            )

    def _is_valid_heading(self, match_str: str, title_part: str) -> bool:
        """Heuristic validation to ensure a numbered line is genuinely a heading and not a numbered list sentence."""
        clean_title = title_part.strip()
        if not clean_title:
            return False
        # Headings are concise, typically under 160 characters
        if len(clean_title) > 160:
            return False
        # Numbered list items typically end in a period, whereas section headings do not
        # (unless the heading ends with a closing parenthesis)
        if clean_title.endswith(".") and not clean_title.endswith(").") and not re.search(r"\(\d+.*?\)\.$", clean_title):
            # Check if it looks like a full prose sentence
            words = clean_title.split()
            if len(words) > 15:
                return False
        return True

    def parse(self, text: str) -> List[Section]:
        if not text or not str(text).strip():
            return []

        # Stage 2: Text Normalization
        normalized_text = TextNormalizer.normalize_for_chunking(text)
        if not normalized_text:
            return []

        if not self.config.detect_numbered_sections:
            return [
                Section(
                    title="",
                    section_number=None,
                    content=normalized_text,
                    start_char=0,
                    end_char=len(normalized_text),
                )
            ]

        # Find all candidate section heading matches
        all_matches = list(self.section_regex.finditer(normalized_text))
        valid_matches = [
            m for m in all_matches if self._is_valid_heading(m.group(0), m.group(2) if len(m.groups()) >= 2 else m.group(0))
        ]

        if not valid_matches:
            # Case 1: No sections detected in document (e.g. FAQ or flat narrative)
            return [
                Section(
                    title="",
                    section_number=None,
                    content=normalized_text,
                    start_char=0,
                    end_char=len(normalized_text),
                )
            ]

        sections: List[Section] = []

        # Check for any preamble text before the first section heading
        first_match_start = valid_matches[0].start()
        preamble = normalized_text[:first_match_start].strip()

        for idx, match in enumerate(valid_matches):
            sec_num = match.group(1).strip()
            sec_title_raw = match.group(0).strip()
            # Strip markdown hashes from title representation if present
            clean_title = re.sub(r"^#{1,6}\s+", "", sec_title_raw).strip()

            content_start = match.end()
            content_end = (
                valid_matches[idx + 1].start()
                if idx + 1 < len(valid_matches)
                else len(normalized_text)
            )

            section_body = normalized_text[content_start:content_end].strip()

            # If there was preamble before the very first section, include it in section 1
            if idx == 0 and preamble:
                section_body = f"{preamble}\n\n{section_body}".strip()
                sec_start_char = 0
            else:
                sec_start_char = match.start()

            # Detect subsections in section_body if configured
            subsections_list: List[SubsectionInfo] = []
            single_subsec_name: Optional[str] = None

            if self.config.detect_subsections and section_body:
                for sub_m in self.subsection_regex.finditer(section_body):
                    sub_num = sub_m.group(1).strip()
                    sub_title_part = sub_m.group(2).strip()
                    clean_subsec_title = f"{sub_num} {sub_title_part}".strip()
                    subsections_list.append(
                        SubsectionInfo(
                            number=sub_num,
                            title=clean_subsec_title,
                            start_char=sub_m.start(),
                            end_char=sub_m.end(),
                        )
                    )

                if len(subsections_list) == 1:
                    single_subsec_name = subsections_list[0].title

            sections.append(
                Section(
                    title=clean_title,
                    section_number=sec_num,
                    content=section_body,
                    subsection=single_subsec_name,
                    subsections=subsections_list,
                    start_char=sec_start_char,
                    end_char=content_end,
                )
            )

        return sections

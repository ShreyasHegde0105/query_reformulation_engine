from __future__ import annotations

import re
from typing import List, Optional, Tuple

from document_chunking.config import ChunkingConfig
from document_chunking.models import Section, SubsectionInfo
from document_chunking.token_counter import BaseTokenCounter


class RecursiveSplitter:
    """Recursively splits document sections into retrieval-friendly chunks.
    
    Hierarchy: Section -> Paragraph -> Sentence -> Token Fallback.
    """

    def __init__(self, config: ChunkingConfig, token_counter: BaseTokenCounter):
        self.config = config
        self.token_counter = token_counter

        # Regex for splitting paragraphs and structured list items
        self.paragraph_split_regex = re.compile(
            r"\n\s*\n|\n(?=\s*(?:\d+\.\d+|[a-z0-9]\))\s+)"
        )
        # Regex for splitting sentences: looks for punctuation followed by whitespace and capital letter/digit
        self.sentence_split_regex = re.compile(
            r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])"
        )

        # Internal fallback metrics counters
        self.recursive_splits = 0
        self.sentence_fallbacks = 0
        self.token_fallbacks = 0

    def split_section(
        self, section: Section
    ) -> List[Tuple[str, int, int, Optional[str]]]:
        """Splits a section into a list of (text_chunk, start_offset, end_offset, subsection).
        
        Returns text chunks with source-relative character offsets and detected subsection.
        """
        raw_content = section.content.strip()
        if not raw_content:
            # If section has title but empty content (e.g. consecutive headings)
            if section.title:
                return [(section.title, 0, len(section.title), section.subsection)]
            return []

        # Determine base text and heading prefix
        heading_prefix = (
            f"{section.title}\n\n"
            if (section.title and self.config.include_section_heading_in_text)
            else ""
        )
        full_section_text = f"{heading_prefix}{raw_content}".strip()
        total_tokens = self.token_counter.count_tokens(full_section_text)

        # -------------------------------------------------------------
        # Level 1: Section-level chunking
        # If the entire section fits within max_chunk_tokens, keep intact
        # -------------------------------------------------------------
        if total_tokens <= self.config.max_chunk_tokens:
            subsec = section.subsection or (section.subsections[0].title if section.subsections else None)
            return [(full_section_text, 0, len(raw_content), subsec)]

        # -------------------------------------------------------------
        # Level 2: Paragraph-level recursive splitting
        # -------------------------------------------------------------
        self.recursive_splits += 1

        paragraphs = [
            p.strip()
            for p in self.paragraph_split_regex.split(raw_content)
            if p.strip()
        ]
        if not paragraphs:
            paragraphs = [raw_content]

        # Break down any oversized paragraphs into sentences
        units: List[str] = []
        for p in paragraphs:
            p_tokens = self.token_counter.count_tokens(p)
            if (
                p_tokens <= self.config.max_chunk_tokens
                or not self.config.enable_sentence_split
            ):
                units.append(p)
            else:
                # Level 3: Sentence-level splitting
                self.sentence_fallbacks += 1
                sentences = [
                    s.strip()
                    for s in self.sentence_split_regex.split(p)
                    if s.strip()
                ]
                for s in sentences:
                    s_tokens = self.token_counter.count_tokens(s)
                    if (
                        s_tokens <= self.config.max_chunk_tokens
                        or not self.config.enable_token_fallback
                    ):
                        units.append(s)
                    else:
                        # Level 4: Token-level fallback for giant sentences
                        self.token_fallbacks += 1
                        token_pieces = self._split_giant_sentence(s)
                        units.extend(token_pieces)

        # -------------------------------------------------------------
        # Assemble units into chunks with target_chunk_tokens and overlap
        # -------------------------------------------------------------
        chunks: List[str] = []
        current_chunk_units: List[str] = []
        current_chunk_tokens = 0

        heading_tokens = (
            self.token_counter.count_tokens(heading_prefix) if heading_prefix else 0
        )

        for unit in units:
            unit_tokens = self.token_counter.count_tokens(unit)

            # Check if adding unit exceeds limits:
            # We close chunk if adding unit exceeds max_chunk_tokens,
            # OR if current_chunk_tokens is already at or above target_chunk_tokens.
            if current_chunk_units and (
                (current_chunk_tokens + unit_tokens + heading_tokens > self.config.max_chunk_tokens)
                or (current_chunk_tokens + heading_tokens >= self.config.target_chunk_tokens)
            ):
                chunk_body = "\n\n".join(current_chunk_units).strip()
                chunks.append(chunk_body)

                # Section-internal overlap: carry trailing units up to overlap_tokens
                overlap_units: List[str] = []
                overlap_tokens_count = 0
                for prev_unit in reversed(current_chunk_units):
                    u_tok = self.token_counter.count_tokens(prev_unit)
                    if overlap_tokens_count + u_tok <= self.config.overlap_tokens:
                        overlap_units.insert(0, prev_unit)
                        overlap_tokens_count += u_tok
                    else:
                        break

                current_chunk_units = list(overlap_units)
                current_chunk_tokens = overlap_tokens_count

            current_chunk_units.append(unit)
            current_chunk_tokens += unit_tokens

        if current_chunk_units:
            final_body = "\n\n".join(current_chunk_units).strip()
            # If the final trailing chunk is tiny and can be merged into the previous chunk without exceeding max_tokens
            if chunks and (self.token_counter.count_tokens(final_body) < self.config.min_chunk_tokens):
                combined = f"{chunks[-1]}\n\n{final_body}".strip()
                if (
                    self.token_counter.count_tokens(combined) + heading_tokens
                    <= self.config.max_chunk_tokens
                ):
                    chunks[-1] = combined
                else:
                    chunks.append(final_body)
            else:
                chunks.append(final_body)

        # Build final chunk tuples with source offsets and subsection detection
        result_chunks: List[Tuple[str, int, int, Optional[str]]] = []
        for body in chunks:
            if heading_prefix and not body.startswith(section.title):
                final_text = f"{section.title}\n\n{body}".strip()
            else:
                final_text = body.strip()

            # Find approximate character offsets in raw_content
            search_sample = body[:60]
            start_pos = raw_content.find(search_sample)
            if start_pos == -1:
                start_pos = 0
            end_pos = min(len(raw_content), start_pos + len(body))

            # Detect subsection for this specific chunk
            chunk_subsec: Optional[str] = None
            if section.subsections:
                # Find matching subsections present in this chunk's body
                matching_subsecs = [
                    s for s in section.subsections
                    if s.number in body or (s.title and s.title.split()[-1] in body)
                ]
                if len(matching_subsecs) == 1:
                    chunk_subsec = matching_subsecs[0].title
                elif len(matching_subsecs) > 1:
                    chunk_subsec = matching_subsecs[0].title
            elif section.subsection:
                chunk_subsec = section.subsection

            result_chunks.append((final_text, start_pos, end_pos, chunk_subsec))

        return result_chunks

    def _split_giant_sentence(self, sentence: str) -> List[str]:
        """Splits an unusually long sentence at token boundaries with overlap."""
        tokens = self.token_counter.tokenize(sentence)
        if not tokens:
            return [sentence]

        pieces: List[str] = []
        target = max(50, self.config.target_chunk_tokens)
        overlap = min(target // 2, self.config.overlap_tokens)

        i = 0
        while i < len(tokens):
            chunk_tokens = tokens[i : i + target]
            pieces.append(" ".join(chunk_tokens))
            if i + target >= len(tokens):
                break
            i += max(1, target - overlap)

        return pieces

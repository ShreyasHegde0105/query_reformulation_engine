from __future__ import annotations

import logging
from collections import Counter
from statistics import median
from typing import Any, Dict, List, Optional, Tuple, Union

from document_chunking.config import ChunkingConfig
from document_chunking.models import Chunk, ChunkingStats
from document_chunking.normalizer import TextNormalizer
from document_chunking.section_parser import SectionParser
from document_chunking.splitter import RecursiveSplitter
from document_chunking.token_counter import BaseTokenCounter, StandardTokenCounter

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Production-quality structure-aware enterprise document chunker.
    
    Adheres strictly to the Document -> Section -> Paragraph -> Sentence -> Token hierarchy.
    Preserves complete parent metadata and lineage while generating deterministic chunk IDs.
    """

    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
        token_counter: Optional[BaseTokenCounter] = None,
    ):
        self.config = config or ChunkingConfig()
        self.token_counter = token_counter or StandardTokenCounter()
        self.section_parser = SectionParser(self.config)
        self.splitter = RecursiveSplitter(self.config, self.token_counter)

    def chunk_document(self, doc: Union[Dict[str, Any], Any]) -> List[Chunk]:
        """Chunk a single enterprise document into structured semantic chunks."""
        # Handle dict or Document wrapper
        doc_dict: Dict[str, Any] = getattr(doc, "raw", doc)
        if not isinstance(doc_dict, dict):
            logger.warning("Skipping document with non-dictionary format")
            return []

        doc_id = doc_dict.get("document_id")
        if not doc_id or not str(doc_id).strip():
            logger.warning("Skipping document without document_id")
            return []

        title = doc_dict.get("title", "")
        content = doc_dict.get("content", "")
        if not content or not str(content).strip():
            logger.warning(f"Document '{doc_id}' has empty content")
            return []

        content_str = str(content)

        # Parse sections
        sections = self.section_parser.parse(content_str)
        if not sections:
            return []

        raw_chunks: List[Tuple[str, Optional[str], Optional[str], Optional[str], int, int]] = []

        for sec in sections:
            sec_chunks = self.splitter.split_section(sec)
            for chunk_text, rel_start, rel_end, chunk_subsec in sec_chunks:
                abs_start = sec.start_char + rel_start
                abs_end = min(len(content_str), sec.start_char + rel_end)
                raw_chunks.append((
                    chunk_text,
                    sec.title if sec.title else None,
                    sec.section_number,
                    chunk_subsec,
                    abs_start,
                    abs_end,
                ))

        total_chunks = len(raw_chunks)
        chunks: List[Chunk] = []

        # Common metadata from parent document
        dept = doc_dict.get("department", "")
        dept_id = doc_dict.get("department_id", "")
        doc_type = doc_dict.get("document_type", "")
        version = doc_dict.get("version")
        eff_date = doc_dict.get("effective_date")
        status = doc_dict.get("status")
        audience = doc_dict.get("audience", [])
        keywords = doc_dict.get("keywords", [])
        retrieval_signals = doc_dict.get("retrieval_signals", [])
        systems = doc_dict.get("systems", [])
        roles = doc_dict.get("roles", [])
        related_documents = doc_dict.get("related_documents", [])

        for idx, (text, sec_title, sec_num, subsec, start_c, end_c) in enumerate(raw_chunks):
            chunk_id = f"{doc_id}_CHUNK_{idx + 1:03d}"
            tok_count = self.token_counter.count_tokens(text)
            char_count = len(text)

            chunk = Chunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                title=title,
                department=dept,
                department_id=dept_id,
                document_type=doc_type,
                text=text,
                chunk_index=idx,
                total_chunks=total_chunks,
                token_count=tok_count,
                character_count=char_count,
                section=sec_title,
                section_number=sec_num,
                subsection=subsec,
                start_char=start_c,
                end_char=end_c,
                version=version,
                effective_date=eff_date,
                status=status,
                audience=audience if isinstance(audience, list) else [],
                keywords=keywords if isinstance(keywords, list) else [],
                retrieval_signals=retrieval_signals if isinstance(retrieval_signals, list) else [],
                systems=systems if isinstance(systems, list) else [],
                roles=roles if isinstance(roles, list) else [],
                related_documents=related_documents if isinstance(related_documents, list) else [],
            )
            chunks.append(chunk)

        return chunks

    def chunk_corpus(
        self, documents: List[Union[Dict[str, Any], Any]]
    ) -> Tuple[List[Chunk], ChunkingStats]:
        """Chunk an entire list of enterprise documents and calculate corpus statistics."""
        all_chunks: List[Chunk] = []
        stats = ChunkingStats(total_documents=len(documents))

        dept_counter: Counter[str] = Counter()
        type_counter: Counter[str] = Counter()
        domain_counter: Counter[str] = Counter()
        sections_detected_count = 0
        docs_without_sections = 0

        chunk_token_counts: List[int] = []

        # Reset splitter internal metrics
        self.splitter.recursive_splits = 0
        self.splitter.sentence_fallbacks = 0
        self.splitter.token_fallbacks = 0

        for doc in documents:
            doc_dict: Dict[str, Any] = getattr(doc, "raw", doc)
            doc_chunks = self.chunk_document(doc_dict)
            if not doc_chunks:
                continue

            all_chunks.extend(doc_chunks)

            dept = doc_dict.get("department", "Unknown")
            dept_counter[dept] += len(doc_chunks)
            doc_type = doc_dict.get("document_type", "Unknown")
            type_counter[doc_type] += len(doc_chunks)

            domain = doc_dict.get("domain") or getattr(doc, "domain", None) or dept
            domain_counter[domain] += len(doc_chunks)

            # Check sections
            unique_sections = {c.section for c in doc_chunks if c.section}
            if unique_sections:
                sections_detected_count += len(unique_sections)
            else:
                docs_without_sections += 1

            for c in doc_chunks:
                chunk_token_counts.append(c.token_count)

        stats.total_chunks = len(all_chunks)
        stats.sections_detected = sections_detected_count
        stats.documents_without_sections = docs_without_sections
        stats.recursive_splits = self.splitter.recursive_splits
        stats.sentence_fallbacks = self.splitter.sentence_fallbacks
        stats.token_fallbacks = self.splitter.token_fallbacks

        if stats.total_documents > 0:
            stats.average_chunks_per_document = stats.total_chunks / stats.total_documents

        if chunk_token_counts:
            stats.average_chunk_tokens = sum(chunk_token_counts) / len(chunk_token_counts)
            stats.median_chunk_tokens = float(median(chunk_token_counts))
            stats.min_chunk_tokens = min(chunk_token_counts)
            stats.max_chunk_tokens = max(chunk_token_counts)

        stats.distributions = {
            "departments": dict(dept_counter),
            "document_types": dict(type_counter),
            "domains": dict(domain_counter),
        }

        return all_chunks, stats

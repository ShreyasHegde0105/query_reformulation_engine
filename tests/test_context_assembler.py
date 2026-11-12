"""Unit tests for Phase 2.4 Multi-Document RAG Context Assembly."""

import json
import unittest
from unittest.mock import MagicMock

from reformulation.answer_generator import AnswerGenerator
from reformulation.context_assembler import AssembledContext, ContextChunk, RAGContextAssembler
from reformulation.llm.client import LLMClient
from retrieval.models import RetrievedChunk


def _make_chunk(
    chunk_id: str,
    doc_id: str,
    title: str = "Policy Title",
    section: str = "Section Name",
    text: str = "Policy text content.",
    token_count: int = 50,
    score: float = 0.9,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        title=title,
        domain="Academic",
        chunk_index=0,
        chunk_text=text,
        section=section,
        token_count=token_count,
        score=score,
    )


class TestRAGContextAssembler(unittest.TestCase):
    """Test suite for RAGContextAssembler adhering to Phase 2.4 requirements."""

    def setUp(self):
        self.assembler = RAGContextAssembler(
            max_tokens=500,
            max_total_chunks=5,
            max_chunks_per_doc=2,
            redundancy_threshold=0.70,
        )

    def test_multiple_chunks_from_one_document(self):
        """Allows multiple chunks from the same document up to max_chunks_per_doc."""
        chunks = [
            _make_chunk("C1", "DOC1", text="Section 1 discusses eligibility rules in detail."),
            _make_chunk("C2", "DOC1", text="Section 2 describes fee refund schedule and deadlines."),
            _make_chunk("C3", "DOC1", text="Section 3 outlines grievance appeal mechanisms."),
        ]
        context = self.assembler.assemble_context("eligibility and refund", chunks)
        # Should select at most 2 chunks from DOC1
        doc1_chunks = [c for c in context.chunks if c.document_id == "DOC1"]
        self.assertEqual(len(doc1_chunks), 2)
        self.assertEqual([c.chunk_id for c in doc1_chunks], ["C1", "C2"])

    def test_multiple_relevant_documents(self):
        """Preserves top chunks across multiple distinct documents."""
        chunks = [
            _make_chunk("C1", "DOC1", text="Hostel admission procedures and room allotment."),
            _make_chunk("C2", "DOC2", text="Mess fee structure and dining hall timings."),
            _make_chunk("C3", "DOC3", text="Campus curfew and security check protocols."),
        ]
        context = self.assembler.assemble_context("hostel mess and curfew rules", chunks)
        self.assertEqual(len(context.document_ids), 3)
        self.assertCountEqual(context.document_ids, ["DOC1", "DOC2", "DOC3"])
        self.assertEqual(len(context.chunks), 3)

    def test_missing_document_metadata(self):
        """Handles missing title and section safely without generating 'None' strings."""
        chunk = _make_chunk("C1", "", title="", section=None, text="Bare content without metadata.")
        context = self.assembler.assemble_context("query", [chunk])
        self.assertEqual(len(context.chunks), 1)
        self.assertEqual(context.chunks[0].title, "Untitled Document")
        self.assertIsNone(context.chunks[0].section)
        self.assertIn("[C1] Untitled Document", context.formatted_sources)
        self.assertNotIn("None", context.formatted_sources)

    def test_duplicate_chunks_pruning(self):
        """Deduplicates identical chunks."""
        chunks = [
            _make_chunk("C1", "DOC1", text="Identical repeated policy clause text."),
            _make_chunk("C2", "DOC1", text="Identical repeated policy clause text."),
        ]
        context = self.assembler.assemble_context("query", chunks)
        self.assertEqual(len(context.chunks), 1)
        self.assertEqual(context.duplicate_chunks_pruned, 1)

    def test_overlapping_chunks_pruning(self):
        """Prunes chunks with token overlap exceeding redundancy threshold."""
        text1 = "Students must maintain at least 75 percent attendance to qualify for semester examinations."
        text2 = "Students must maintain at least 75 percent attendance to qualify for semester end exams."
        chunks = [
            _make_chunk("C1", "DOC1", text=text1),
            _make_chunk("C2", "DOC1", text=text2),
            _make_chunk("C3", "DOC2", text="Medical certificates must be submitted within 3 days of absence."),
        ]
        context = self.assembler.assemble_context("attendance and medical leave", chunks)
        selected_ids = [c.chunk_id for c in context.chunks]
        self.assertIn("C1", selected_ids)
        self.assertNotIn("C2", selected_ids)  # pruned as overlapping
        self.assertIn("C3", selected_ids)
        self.assertEqual(context.duplicate_chunks_pruned, 1)

    def test_context_token_budget_enforcement(self):
        """Respects the maximum token budget and stops selecting chunks when budget is full."""
        assembler = RAGContextAssembler(max_tokens=120, max_total_chunks=5)
        chunks = [
            _make_chunk("C1", "DOC1", text="Short chunk one", token_count=50),
            _make_chunk("C2", "DOC2", text="Short chunk two", token_count=50),
            _make_chunk("C3", "DOC3", text="Large chunk three", token_count=80),  # would exceed 120
        ]
        context = assembler.assemble_context("query", chunks)
        self.assertEqual(len(context.chunks), 2)
        self.assertLessEqual(context.total_tokens, 120)

    def test_citation_preservation_and_provenance(self):
        """Preserves chunk ID, document ID, title, section, and original rank."""
        chunk = _make_chunk(
            "C1",
            "ACAD-POL-001",
            title="Grading System",
            section="4. GPA CALCULATION",
            text="Cumulative GPA is calculated as sum of credit points divided by total credits.",
            score=0.95,
        )
        context = self.assembler.assemble_context("GPA calculation", [chunk])
        c = context.chunks[0]
        self.assertEqual(c.chunk_id, "C1")
        self.assertEqual(c.document_id, "ACAD-POL-001")
        self.assertEqual(c.title, "Grading System")
        self.assertEqual(c.section, "4. GPA CALCULATION")
        self.assertEqual(c.original_rank, 1)
        self.assertAlmostEqual(c.score, 0.95)

    def test_insufficient_evidence_handling(self):
        """Returns empty assembled context when no candidates are provided."""
        context = self.assembler.assemble_context("nonexistent topic", [])
        self.assertEqual(len(context.chunks), 0)
        self.assertEqual(context.total_tokens, 0)
        self.assertEqual(context.document_ids, [])
        self.assertEqual(context.formatted_sources, "")

    def test_single_document_query(self):
        """Correctly handles single-document focused queries without over-diversifying."""
        chunks = [
            _make_chunk("C1", "DOC1", text="Section 1 details course drop deadlines.", section="Drop Deadline"),
            _make_chunk("C2", "DOC1", text="Section 2 specifies fee forfeiture on drop.", section="Drop Fee"),
        ]
        context = self.assembler.assemble_context("course drop deadline and fee", chunks)
        self.assertEqual(len(context.document_ids), 1)
        self.assertEqual(context.document_ids[0], "DOC1")
        self.assertEqual(len(context.chunks), 2)

    def test_multi_document_query_aspect_coverage(self):
        """Assembles distinct evidence across documents to satisfy multiple query aspects."""
        chunks = [
            _make_chunk("C1", "DOC_SCHOLARSHIP", text="Merit scholarship requires minimum 8.5 CGPA.", section="Eligibility"),
            _make_chunk("C2", "DOC_ADMISSION", text="Application portal opens on July 1st annually.", section="Application Process"),
            _make_chunk("C3", "DOC_FINANCE", text="Income certificate and marksheets required.", section="Documents"),
        ]
        query = "Explain scholarship eligibility, application process, and required documents"
        context = self.assembler.assemble_context(query, chunks)
        self.assertEqual(len(context.document_ids), 3)
        self.assertIn("eligibility", " ".join(context.covered_aspects).lower())
        self.assertIn("documents", " ".join(context.covered_aspects).lower())


class TestAnswerGeneratorWithAssembler(unittest.TestCase):
    """Test AnswerGenerator integration with RAGContextAssembler."""

    def test_answer_generator_uses_assembler_and_records_cited_docs(self):
        llm = MagicMock(spec=LLMClient)
        assembler = RAGContextAssembler(max_total_chunks=3, max_chunks_per_doc=1)
        gen = AnswerGenerator(llm, context_assembler=assembler)

        chunks = [
            _make_chunk("C1", "DOC1", text="Library borrowing limit is 4 books."),
            _make_chunk("C2", "DOC1", text="Library fine is 10 rupees per day."),  # Same doc -> capped
            _make_chunk("C3", "DOC2", text="Hostel silence hours begin at 10 PM."),
        ]

        llm.generate.return_value = json.dumps({
            "answerable": True,
            "answer": "Borrowing limit is 4 books and silence hours begin at 10 PM.",
            "citations": ["C1", "C3"],
        })

        ans = gen.generate("borrowing limit and hostel silence hours", chunks)
        self.assertTrue(ans.answerable)
        self.assertCountEqual(ans.citations, ["C1", "C3"])
        self.assertCountEqual(ans.cited_documents, ["DOC1", "DOC2"])
        self.assertGreater(ans.context_token_count, 0)

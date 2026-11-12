"""Unit tests for Phase 2.4.1 Relevance-First Multi-Document RAG Context Assembly."""

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
    ce_score: float = 5.0,
) -> RetrievedChunk:
    metadata = {"ce_score": ce_score}
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
        metadata=metadata,
    )


class TestRAGContextAssembler(unittest.TestCase):
    """Comprehensive test suite for RAGContextAssembler adhering to Phase 2.4.1 requirements."""

    def setUp(self):
        self.assembler = RAGContextAssembler(
            max_tokens=500,
            max_total_chunks=6,
            max_chunks_per_doc=2,
            max_dominant_doc_chunks=4,
            redundancy_threshold=0.70,
            strategy="relevance_first",
        )
        self.capped_assembler = RAGContextAssembler(
            max_tokens=500,
            max_total_chunks=6,
            max_chunks_per_doc=2,
            redundancy_threshold=0.70,
            strategy="diversity_capped",
        )

    def test_single_document_query(self):
        """Single-document queries retain dominant chunks without forcing artificial diversity."""
        chunks = [
            _make_chunk("C1", "DOC1", text="Bus pass application procedure and form submission.", score=0.95, ce_score=8.0),
            _make_chunk("C2", "DOC1", text="Bus pass fee schedule and refund conditions.", score=0.90, ce_score=7.0),
            _make_chunk("C3", "DOC1", text="Lost pass penalty and re-issue guidelines.", score=0.85, ce_score=6.0),
            _make_chunk("C4", "DOC_IRRELEVANT", text="Library catalog search instructions.", score=0.10, ce_score=-4.0),
        ]
        context = self.assembler.assemble_context("bus pass fee and refund", chunks)
        # Should select all 3 relevant chunks from DOC1 and exclude DOC_IRRELEVANT
        self.assertEqual(context.document_ids, ["DOC1"])
        self.assertEqual(len(context.chunks), 3)
        self.assertEqual([c.chunk_id for c in context.chunks], ["C1", "C2", "C3"])

    def test_more_than_two_essential_chunks_from_one_document(self):
        """Permits more than two chunks from a document when justified by high relevance."""
        chunks = [
            _make_chunk("C1", "DOC1", text="Section 1 details course drop deadlines.", ce_score=8.0),
            _make_chunk("C2", "DOC1", text="Section 2 specifies refund percentages by week.", ce_score=7.5),
            _make_chunk("C3", "DOC1", text="Section 3 outlines appeal mechanisms for exceptional cases.", ce_score=7.0),
            _make_chunk("C4", "DOC1", text="Section 4 provides contact details for student finance.", ce_score=6.5),
        ]
        context = self.assembler.assemble_context("course drop deadline refund and appeals", chunks)
        doc1_chunks = [c for c in context.chunks if c.document_id == "DOC1"]
        self.assertGreater(len(doc1_chunks), 2)
        self.assertEqual(len(doc1_chunks), 4)

    def test_diversity_capped_strategy_enforces_two_chunks(self):
        """Verifies diversity_capped mode strictly enforces max_chunks_per_doc=2."""
        chunks = [
            _make_chunk("C1", "DOC1", text="Clause 1 of regulation."),
            _make_chunk("C2", "DOC1", text="Clause 2 of regulation."),
            _make_chunk("C3", "DOC1", text="Clause 3 of regulation."),
        ]
        context = self.capped_assembler.assemble_context("regulation query", chunks)
        self.assertEqual(len(context.chunks), 2)

    def test_multiple_relevant_documents(self):
        """Preserves top chunks across multiple distinct relevant documents."""
        chunks = [
            _make_chunk("C1", "DOC_HOSTEL", text="Hostel admission procedures and room allotment.", ce_score=6.5),
            _make_chunk("C2", "DOC_MESS", text="Mess fee structure and dining hall timings.", ce_score=6.0),
            _make_chunk("C3", "DOC_SECURITY", text="Campus curfew and gate security protocols.", ce_score=5.5),
        ]
        context = self.assembler.assemble_context("hostel admission mess fee and curfew rules", chunks)
        self.assertEqual(len(context.document_ids), 3)
        self.assertCountEqual(context.document_ids, ["DOC_HOSTEL", "DOC_MESS", "DOC_SECURITY"])
        self.assertEqual(len(context.chunks), 3)

    def test_several_documents_containing_necessary_evidence(self):
        """Reserves context slots for multi-document queries with competitive scores."""
        chunks = [
            _make_chunk("C1", "DOC1", text="Scholarship CGPA eligibility requirement.", ce_score=7.0),
            _make_chunk("C2", "DOC1", text="Scholarship renewal conditions.", ce_score=6.5),
            _make_chunk("C3", "DOC1", text="Scholarship committee review board.", ce_score=6.0),
            _make_chunk("C4", "DOC2", text="Application portal opening date and form.", ce_score=5.8),
            _make_chunk("C5", "DOC3", text="Income certificate and affidavit verification.", ce_score=5.2),
        ]
        context = self.assembler.assemble_context("scholarship eligibility application portal income documents", chunks)
        # DOC1 should not saturate all slots; DOC2 and DOC3 must be admitted
        self.assertIn("DOC1", context.document_ids)
        self.assertIn("DOC2", context.document_ids)
        self.assertIn("DOC3", context.document_ids)

    def test_redundant_documents_filtered_out(self):
        """Does not select chunks from irrelevant / redundant low-scoring documents."""
        chunks = [
            _make_chunk("C1", "DOC_EXP", text="Parking permit rules and fees.", ce_score=6.0),
            _make_chunk("C2", "DOC_EXP", text="Parking sticker application steps.", ce_score=5.5),
            _make_chunk("C3", "DOC_IRR1", text="Hostel laundry schedule.", ce_score=-5.0),
            _make_chunk("C4", "DOC_IRR2", text="Placement interview guidelines.", ce_score=-8.0),
        ]
        context = self.assembler.assemble_context("parking sticker rules", chunks)
        self.assertEqual(context.document_ids, ["DOC_EXP"])
        self.assertNotIn("DOC_IRR1", context.document_ids)
        self.assertNotIn("DOC_IRR2", context.document_ids)

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
        self.assertNotIn("C2", selected_ids)
        self.assertIn("C3", selected_ids)
        self.assertEqual(context.duplicate_chunks_pruned, 1)

    def test_duplicate_chunks_pruning(self):
        """Deduplicates exact identical chunks."""
        chunks = [
            _make_chunk("C1", "DOC1", text="Identical repeated policy clause text."),
            _make_chunk("C2", "DOC1", text="Identical repeated policy clause text."),
        ]
        context = self.assembler.assemble_context("query", chunks)
        self.assertEqual(len(context.chunks), 1)
        self.assertEqual(context.duplicate_chunks_pruned, 1)

    def test_missing_document_metadata(self):
        """Handles missing title and section safely without generating 'None' strings."""
        chunk = _make_chunk("C1", "", title="", section=None, text="Bare content without metadata.")
        context = self.assembler.assemble_context("query", [chunk])
        self.assertEqual(len(context.chunks), 1)
        self.assertEqual(context.chunks[0].title, "Untitled Document")
        self.assertIsNone(context.chunks[0].section)
        self.assertIn("[C1] Untitled Document", context.formatted_sources)
        self.assertNotIn("None", context.formatted_sources)

    def test_context_token_budget_enforcement(self):
        """Respects the maximum token budget and stops selecting chunks when budget is full."""
        assembler = RAGContextAssembler(max_tokens=120, max_total_chunks=5)
        chunks = [
            _make_chunk("C1", "DOC1", text="Short chunk one", token_count=50),
            _make_chunk("C2", "DOC2", text="Short chunk two", token_count=50),
            _make_chunk("C3", "DOC3", text="Large chunk three", token_count=80),
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

    def test_queries_with_missing_information_aspects(self):
        """Tracks covered vs missing aspects accurately."""
        chunks = [
            _make_chunk("C1", "DOC1", text="Scholarship eligibility requires 8.5 CGPA.", section="Eligibility"),
        ]
        query = "Explain scholarship eligibility and application deadline"
        context = self.assembler.assemble_context(query, chunks)
        self.assertIn("scholarship eligibility", context.covered_aspects)
        self.assertIn("application deadline", context.missing_aspects)


class TestAnswerGeneratorWithAssembler(unittest.TestCase):
    """Tests integration of RAGContextAssembler with AnswerGenerator."""

    def test_answer_generator_uses_assembler_and_records_cited_docs(self):
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.generate.return_value = json.dumps({
            "answerable": True,
            "answer": "GPA is calculated from total credits [C1].",
            "citations": ["C1"],
        })

        assembler = RAGContextAssembler(max_tokens=1000, max_total_chunks=3)
        gen = AnswerGenerator(llm_client=mock_llm, context_assembler=assembler)

        chunks = [
            _make_chunk("C1", "DOC_ACAD", title="Grading Policy", section="GPA", text="GPA formula text."),
        ]
        ans = gen.generate("How to calculate GPA?", chunks)
        self.assertTrue(ans.answerable)
        self.assertEqual(ans.citations, ["C1"])
        self.assertEqual(ans.cited_documents, ["DOC_ACAD"])
        self.assertGreater(ans.context_token_count, 0)


if __name__ == "__main__":
    unittest.main()

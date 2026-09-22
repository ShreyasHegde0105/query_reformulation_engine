from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from document_chunking.chunker import DocumentChunker
from document_chunking.cli import main
from document_chunking.config import ChunkingConfig
from document_chunking.loader import DocumentLoader
from document_chunking.models import Chunk
from document_chunking.normalizer import TextNormalizer
from document_chunking.section_parser import SectionParser
from document_chunking.token_counter import StandardTokenCounter
from document_chunking.validator import ChunkQualityValidator


def get_sample_doc(
    doc_id: str = "POL-ATTEND",
    title: str = "Mandatory Attendance Policy",
    content: str | None = None,
    doc_type: str = "Policy",
) -> Dict[str, Any]:
    if content is None:
        content = (
            "1. PURPOSE AND INSTITUTIONAL MANDATE\n"
            "Demo University maintains rigorous academic standards centered on active classroom instruction. "
            "Regular and punctual attendance in all scheduled instructional periods is fundamental.\n\n"
            "2. JURISDICTION AND SCOPE\n"
            "This policy governs all matriculated students enrolled in undergraduate and postgraduate coursework.\n\n"
            "3. MINIMUM ATTENDANCE THRESHOLD\n"
            "3.1 Mandatory Standard: Every enrolled student must maintain a minimum cumulative attendance of 75.0%.\n"
            "3.2 Computation Principle: Attendance percentages are computed per course code as the ratio of sessions attended.\n\n"
            "4. CONDONATION OF ATTENDANCE SHORTAGE\n"
            "The Academic Council authorizes discretionary condonation strictly between 65.0% and 74.9% on verified medical grounds."
        )
    return {
        "document_id": doc_id,
        "title": title,
        "department": "Academic Affairs",
        "department_id": "DEPT-ACAD",
        "document_type": doc_type,
        "version": "3.1",
        "effective_date": "2026-08-01",
        "status": "Active",
        "audience": ["All Students", "Faculty Advisors"],
        "keywords": ["attendance", "condonation", "75% requirement", "detention"],
        "retrieval_signals": ["how to get attendance condoned", "minimum attendance required"],
        "systems": ["AttendEdge", "UniCampus ERP"],
        "roles": ["Dean of Academic Affairs", "Faculty Advisor"],
        "related_documents": ["POL-CIE", "POL-SEE"],
        "content": content,
    }


class TestDocumentChunkingSuite(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ChunkingConfig(
            target_chunk_tokens=100,
            max_chunk_tokens=150,
            min_chunk_tokens=20,
            overlap_tokens=25,
            include_section_heading_in_text=True,
        )
        self.chunker = DocumentChunker(config=self.config)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # 1. Simple document with one section
    def test_01_simple_document_one_section(self) -> None:
        doc = get_sample_doc(content="1. PURPOSE\nThis is the single purpose section of the document.")
        chunks = self.chunker.chunk_document(doc)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].section, "1. PURPOSE")
        self.assertEqual(chunks[0].section_number, "1")

    # 2. Document with multiple sections
    def test_02_multiple_sections(self) -> None:
        doc = get_sample_doc()
        chunks = self.chunker.chunk_document(doc)
        sections = {c.section for c in chunks if c.section}
        self.assertGreaterEqual(len(sections), 3)

    # 3. Document with subsections
    def test_03_subsections(self) -> None:
        doc = get_sample_doc()
        chunks = self.chunker.chunk_document(doc)
        subsections = [c.subsection for c in chunks if c.subsection]
        self.assertTrue(any("3.1" in s for s in subsections))

    # 4. Small section remains intact
    def test_04_small_section_remains_intact(self) -> None:
        doc = get_sample_doc(content="1. SCOPE\nAll undergraduate students must comply.")
        chunks = self.chunker.chunk_document(doc)
        self.assertEqual(len(chunks), 1)
        self.assertIn("undergraduate students", chunks[0].text)

    # 5. Large section splits by paragraph
    def test_05_large_section_splits_by_paragraph(self) -> None:
        p1 = "Paragraph 1: " + ("word " * 60)
        p2 = "Paragraph 2: " + ("word " * 60)
        p3 = "Paragraph 3: " + ("word " * 60)
        content = f"1. REGULATIONS\n\n{p1}\n\n{p2}\n\n{p3}"
        doc = get_sample_doc(content=content)
        chunks = self.chunker.chunk_document(doc)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertEqual(c.section_number, "1")

    # 6. Large paragraph splits by sentence
    def test_06_large_paragraph_splits_by_sentence(self) -> None:
        s1 = "Sentence one contains extensive details on policy criteria. " * 8
        s2 = "Sentence two provides further explanations and requirements. " * 8
        s3 = "Sentence three summarizes the procedural steps for submission. " * 8
        content = f"1. PROCEDURE\n\n{s1} {s2} {s3}"
        doc = get_sample_doc(content=content)
        chunks = self.chunker.chunk_document(doc)
        self.assertGreater(len(chunks), 1)

    # 7. Very large sentence uses token fallback
    def test_07_very_large_sentence_token_fallback(self) -> None:
        giant_sentence = "This is an extremely long sentence containing " + ("continuous unbroken text " * 50) + "ending finally here."
        content = f"1. GUIDELINE\n\n{giant_sentence}"
        doc = get_sample_doc(content=content)
        chunks = self.chunker.chunk_document(doc)
        self.assertGreater(len(chunks), 1)

    # 8. Document without headings
    def test_08_document_without_headings(self) -> None:
        content = "This is a document that does not contain any numbered section headings.\n\nIt simply consists of narrative paragraphs discussing general campus information and student activities."
        doc = get_sample_doc(content=content)
        chunks = self.chunker.chunk_document(doc)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertIsNone(chunks[0].section_number)

    # 9. Empty content
    def test_09_empty_content(self) -> None:
        doc = get_sample_doc(content="   \n\t  ")
        chunks = self.chunker.chunk_document(doc)
        self.assertEqual(len(chunks), 0)

    # 10. Consecutive headings
    def test_10_consecutive_headings(self) -> None:
        content = "1. OVERVIEW\n\n2. PURPOSE\nActual text for the purpose section."
        doc = get_sample_doc(content=content)
        chunks = self.chunker.chunk_document(doc)
        self.assertGreaterEqual(len(chunks), 1)

    # 11. Metadata preservation
    def test_11_metadata_preservation(self) -> None:
        doc = get_sample_doc()
        chunks = self.chunker.chunk_document(doc)
        self.assertGreater(len(chunks), 0)
        c = chunks[0]
        self.assertEqual(c.document_id, doc["document_id"])
        self.assertEqual(c.department, doc["department"])
        self.assertEqual(c.department_id, doc["department_id"])
        self.assertEqual(c.document_type, doc["document_type"])
        self.assertEqual(c.version, doc["version"])
        self.assertEqual(c.status, doc["status"])

    # 12. Retrieval signals preservation
    def test_12_retrieval_signals_preservation(self) -> None:
        doc = get_sample_doc()
        chunks = self.chunker.chunk_document(doc)
        c = chunks[0]
        self.assertEqual(c.retrieval_signals, doc["retrieval_signals"])
        for sig in doc["retrieval_signals"]:
            self.assertNotIn(f"retrieval_signals: {sig}", c.text)

    # 13. Keyword preservation
    def test_13_keyword_preservation(self) -> None:
        doc = get_sample_doc()
        chunks = self.chunker.chunk_document(doc)
        self.assertEqual(chunks[0].keywords, doc["keywords"])

    # 14. Related document preservation
    def test_14_related_document_preservation(self) -> None:
        doc = get_sample_doc()
        chunks = self.chunker.chunk_document(doc)
        self.assertEqual(chunks[0].related_documents, doc["related_documents"])

    # 15. Deterministic chunk IDs
    def test_15_deterministic_chunk_ids(self) -> None:
        doc = get_sample_doc()
        chunks1 = self.chunker.chunk_document(doc)
        chunks2 = self.chunker.chunk_document(doc)
        ids1 = [c.chunk_id for c in chunks1]
        ids2 = [c.chunk_id for c in chunks2]
        self.assertEqual(ids1, ids2)
        self.assertEqual(ids1[0], "POL-ATTEND_CHUNK_001")

    # 16. Sequential chunk indexes
    def test_16_sequential_chunk_indexes(self) -> None:
        doc = get_sample_doc()
        chunks = self.chunker.chunk_document(doc)
        indexes = [c.chunk_index for c in chunks]
        self.assertEqual(indexes, list(range(len(chunks))))

    # 17. Correct total_chunks
    def test_17_correct_total_chunks(self) -> None:
        doc = get_sample_doc()
        chunks = self.chunker.chunk_document(doc)
        n = len(chunks)
        for c in chunks:
            self.assertEqual(c.total_chunks, n)

    # 18. No empty chunks
    def test_18_no_empty_chunks(self) -> None:
        doc = get_sample_doc()
        chunks = self.chunker.chunk_document(doc)
        for c in chunks:
            self.assertTrue(len(c.text.strip()) > 0)
            self.assertGreater(c.token_count, 0)

    # 19. Section heading inclusion
    def test_19_section_heading_inclusion(self) -> None:
        doc = get_sample_doc(content="1. PURPOSE AND MANDATE\nText body.")
        chunks = self.chunker.chunk_document(doc)
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].text.startswith("1. PURPOSE AND MANDATE"))

    # 20. Overlap behavior
    def test_20_overlap_behavior(self) -> None:
        p1 = "Paragraph Alpha with distinct tokens alpha1 alpha2 alpha3. " * 15
        p2 = "Paragraph Beta with distinct tokens beta1 beta2 beta3. " * 15
        content = f"1. REGULATIONS\n\n{p1}\n\n{p2}"
        doc = get_sample_doc(content=content)
        chunks = self.chunker.chunk_document(doc)
        if len(chunks) > 1:
            self.assertTrue(len(chunks[1].text) > 0)

    # 21. Chunk size configuration
    def test_21_chunk_size_configuration(self) -> None:
        custom_cfg = ChunkingConfig(target_chunk_tokens=50, max_chunk_tokens=80)
        custom_chunker = DocumentChunker(config=custom_cfg)
        doc = get_sample_doc()
        chunks = custom_chunker.chunk_document(doc)
        self.assertGreater(len(chunks), 1)

    # 22. Multiple documents
    def test_22_multiple_documents(self) -> None:
        doc1 = get_sample_doc("DOC-001", "Policy 1")
        doc2 = get_sample_doc("DOC-002", "Policy 2")
        chunks, stats = self.chunker.chunk_corpus([doc1, doc2])
        self.assertEqual(stats.total_documents, 2)
        self.assertGreater(stats.total_chunks, 1)

    # 23. Different document types
    def test_23_different_document_types(self) -> None:
        doc_pol = get_sample_doc("DOC-POL", "Policy", doc_type="Policy")
        doc_sop = get_sample_doc("DOC-SOP", "SOP", doc_type="Standard Operating Procedure")
        chunks, stats = self.chunker.chunk_corpus([doc_pol, doc_sop])
        types = stats.distributions["document_types"]
        self.assertIn("Policy", types)
        self.assertIn("Standard Operating Procedure", types)

    # 24. Large document
    def test_24_large_document(self) -> None:
        sections = [f"{i}. SECTION {i}\n" + ("Long detailed instruction paragraph for section. " * 20) for i in range(1, 10)]
        large_content = "\n\n".join(sections)
        doc = get_sample_doc("DOC-LARGE", "Large Manual", content=large_content)
        chunks = self.chunker.chunk_document(doc)
        self.assertGreaterEqual(len(chunks), 9)

    # 25. End-to-end chunking run via CLI
    def test_25_end_to_end_cli_run(self) -> None:
        val_file = Path(self.temp_dir) / "validated_docs.json"
        with open(val_file, "w", encoding="utf-8") as f:
            json.dump({"documents": [get_sample_doc()]}, f)

        out_file = Path(self.temp_dir) / "chunks.json"
        exit_code = main(["-i", str(val_file), "-o", str(out_file), "--log-level", "ERROR"])
        self.assertEqual(exit_code, 0)
        self.assertTrue(out_file.exists())
        with open(out_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertGreater(len(data), 0)

    # -------------------------------------------------------------
    # 8 Core Properties Verification
    # -------------------------------------------------------------
    def test_property_1_every_chunk_belongs_to_one_document(self) -> None:
        doc1 = get_sample_doc("DOC-A")
        doc2 = get_sample_doc("DOC-B")
        chunks, _ = self.chunker.chunk_corpus([doc1, doc2])
        for c in chunks:
            self.assertIn(c.document_id, {"DOC-A", "DOC-B"})

    def test_property_2_every_source_doc_traced_to_chunks(self) -> None:
        doc1 = get_sample_doc("DOC-A")
        doc2 = get_sample_doc("DOC-B")
        chunks, _ = self.chunker.chunk_corpus([doc1, doc2])
        doc_ids_in_chunks = {c.document_id for c in chunks}
        self.assertEqual(doc_ids_in_chunks, {"DOC-A", "DOC-B"})

    def test_property_3_chunk_ids_are_deterministic(self) -> None:
        doc = get_sample_doc("DOC-DET")
        chunks_run_1 = self.chunker.chunk_document(doc)
        chunks_run_2 = self.chunker.chunk_document(doc)
        self.assertEqual([c.chunk_id for c in chunks_run_1], [c.chunk_id for c in chunks_run_2])

    def test_property_4_document_order_does_not_affect_chunk_ids(self) -> None:
        doc1 = get_sample_doc("DOC-1")
        doc2 = get_sample_doc("DOC-2")
        chunks_forward, _ = self.chunker.chunk_corpus([doc1, doc2])
        chunks_reverse, _ = self.chunker.chunk_corpus([doc2, doc1])

        doc1_ids_forward = [c.chunk_id for c in chunks_forward if c.document_id == "DOC-1"]
        doc1_ids_reverse = [c.chunk_id for c in chunks_reverse if c.document_id == "DOC-1"]
        self.assertEqual(doc1_ids_forward, doc1_ids_reverse)

    def test_property_5_chunk_metadata_matches_parent(self) -> None:
        doc = get_sample_doc("DOC-META")
        chunks = self.chunker.chunk_document(doc)
        for c in chunks:
            self.assertEqual(c.title, doc["title"])
            self.assertEqual(c.department, doc["department"])
            self.assertEqual(c.keywords, doc["keywords"])

    def test_property_6_validated_corpus_is_unchanged(self) -> None:
        doc = get_sample_doc("DOC-IMMUTABLE")
        doc_copy = json.dumps(doc)
        _ = self.chunker.chunk_document(doc)
        self.assertEqual(json.dumps(doc), doc_copy)

    def test_property_7_chunks_preserve_source_content(self) -> None:
        doc = get_sample_doc()
        chunks = self.chunker.chunk_document(doc)
        for c in chunks:
            self.assertGreater(len(c.text), 10)

    def test_property_8_section_boundaries_respected(self) -> None:
        doc = get_sample_doc()
        chunks = self.chunker.chunk_document(doc)
        for c in chunks:
            if c.section:
                self.assertTrue(c.section[0].isdigit())

    # -------------------------------------------------------------
    # Stage-Specific and Edge-Case Tests
    # -------------------------------------------------------------
    def test_stage_1_document_loader(self) -> None:
        # Test loading valid json
        test_file = Path(self.temp_dir) / "test_docs.json"
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump({"documents": [get_sample_doc("DOC-LOAD")]}, f)

        docs, errs = DocumentLoader.load(test_file)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["document_id"], "DOC-LOAD")
        self.assertEqual(len(errs), 0)

        # Test loading invalid json gracefully
        bad_file = Path(self.temp_dir) / "bad.json"
        with open(bad_file, "w", encoding="utf-8") as f:
            f.write("{invalid_json}")
        docs_bad, errs_bad = DocumentLoader.load(bad_file)
        self.assertEqual(len(docs_bad), 0)
        self.assertGreater(len(errs_bad), 0)

    def test_stage_2_text_normalizer(self) -> None:
        raw_text = "Line 1\r\nLine 2\rLine 3   \n\n\n\nLine 4\u00a0with\u2005special spaces."
        normalized = TextNormalizer.normalize_for_chunking(raw_text)
        self.assertNotIn("\r", normalized)
        self.assertNotIn("\n\n\n", normalized)
        self.assertNotIn("\u00a0", normalized)
        self.assertIn("Line 1\nLine 2\nLine 3\n\nLine 4 with special spaces.", normalized)

    def test_mixed_case_and_title_case_sections(self) -> None:
        content = "1. Purpose and Scope\nDetailed content for scope.\n\n2. Eligibility Criteria\nDetailed eligibility rules."
        doc = get_sample_doc(content=content)
        chunks = self.chunker.chunk_document(doc)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].section, "1. Purpose and Scope")
        self.assertEqual(chunks[1].section, "2. Eligibility Criteria")

    def test_clean_subsection_title_extraction(self) -> None:
        content = (
            "3. REGULATIONS\n\n"
            "3.1 Mandatory Standard: Every student must attend at least 75 percent of all instructional periods across the entire semester.\n\n"
            "3.2 Computation Principle: Attendance is computed automatically from AttendEdge RFID card readers and classroom biometrics."
        )
        doc = get_sample_doc(content=content)
        custom_cfg = ChunkingConfig(target_chunk_tokens=20, max_chunk_tokens=25, min_chunk_tokens=5, overlap_tokens=0)
        custom_chunker = DocumentChunker(config=custom_cfg)
        chunks = custom_chunker.chunk_document(doc)
        subsecs = [c.subsection for c in chunks if c.subsection]
        self.assertTrue(any(s == "3.1 Mandatory Standard" for s in subsecs))
        self.assertTrue(any(s == "3.2 Computation Principle" for s in subsecs))

    def test_quality_validator_detects_errors(self) -> None:
        # Test validator detects missing document_id or duplicate IDs
        valid_doc = get_sample_doc("DOC-V1")
        bad_chunk1 = Chunk(
            chunk_id="DUP_CHUNK",
            document_id="DOC-V1",
            title="Title",
            department="Dept",
            department_id="D1",
            document_type="Policy",
            text="Valid text body.",
            chunk_index=0,
            total_chunks=2,
            token_count=10,
            character_count=15,
        )
        bad_chunk2 = Chunk(
            chunk_id="DUP_CHUNK",
            document_id="DOC-NONEXISTENT",
            title="Title",
            department="Dept",
            department_id="D1",
            document_type="Policy",
            text="Another text body.",
            chunk_index=1,
            total_chunks=2,
            token_count=10,
            character_count=17,
        )
        res = ChunkQualityValidator.validate_chunks([bad_chunk1, bad_chunk2], [valid_doc])
        self.assertFalse(res["is_valid"])
        self.assertGreater(res["error_count"], 0)
        # Check duplicate ID was caught
        self.assertTrue(any("Duplicate chunk ID" in e for e in res["errors"]))
        # Check non-existent doc_id was caught
        self.assertTrue(any("does not exist in source corpus" in e for e in res["errors"]))


if __name__ == "__main__":
    unittest.main()

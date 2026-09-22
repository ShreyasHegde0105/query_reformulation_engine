


from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

from corpus_validation.aggregator import ValidationAggregator
from corpus_validation.cli import main
from corpus_validation.config import ValidationConfig
from corpus_validation.exporter import CorpusExporter
from corpus_validation.loader import CorpusLoader
from corpus_validation.models import (
    Document,
    Severity,
    ValidationCategory,
    ValidationIssue,
)
from corpus_validation.reporter import ValidationReporter
from corpus_validation.validator import CorpusValidator
from corpus_validation.validators import (
    ContentValidator,
    CoverageValidator,
    DuplicateDetector,
    EntityValidator,
    GroundTruthValidator,
    IdValidator,
    MetadataValidator,
    QualityValidator,
    SchemaValidator,
)


def get_sample_valid_doc_raw(doc_id: str = "POL-TEST-001") -> Dict[str, Any]:
    return {
        "document_id": doc_id,
        "title": "Institutional Examination and Grading Policy",
        "department": "Academic Affairs",
        "department_id": "DEPT-ACAD",
        "document_type": "Policy",
        "version": "1.0",
        "effective_date": "2026-08-01",
        "status": "Active",
        "audience": ["All Undergraduate Students", "Faculty Members"],
        "keywords": ["examination", "grading scheme", "SGPA", "CGPA calculation", "re-evaluation"],
        "retrieval_signals": [
            "how is SGPA calculated",
            "grading policy for B.Tech students",
            "what is the passing grade in exam",
            "how to apply for grade re-check"
        ],
        "systems": ["UniCampus ERP", "ExamShield Portal"],
        "roles": ["Dean of Academic Affairs", "Controller of Examinations (COE)"],
        "related_documents": ["POL-CIE", "POL-SEE"],
        "content": (
            "1. PURPOSE AND INSTITUTIONAL MANDATE\n"
            "This policy establishes the formal grading metrics, continuous internal evaluation, and semester-end assessment framework for all academic programs.\n\n"
            "2. JURISDICTION AND SCOPE\n"
            "The regulations apply uniformly to all matriculated undergraduate and postgraduate students enrolled at Demo University.\n\n"
            "3. GRADING SCHEME AND SGPA CALCULATION\n"
            "Grades are awarded on a 10-point scale. A minimum of 40% in Continuous Internal Evaluation (CIE) and Semester End Examination (SEE) is mandatory for passing."
        ),
    }


class TestCorpusValidationSuite(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ValidationConfig(
            min_content_length=100,
            max_content_length=10000,
            min_word_count=15,
            near_duplicate_threshold=0.85,
            world_model_path=None,  # Handled per test
        )
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # 1. Valid document test
    def test_01_valid_document(self) -> None:
        doc = Document(raw=get_sample_valid_doc_raw())
        validator = SchemaValidator(self.config)
        issues = validator.validate([doc])
        errors = [i for i in issues if i.severity == Severity.ERROR]
        self.assertEqual(len(errors), 0, f"Expected 0 errors for valid document, got: {errors}")

    # 2. Missing required field
    def test_02_missing_required_field(self) -> None:
        raw = get_sample_valid_doc_raw()
        del raw["content"]
        doc = Document(raw=raw)
        validator = SchemaValidator(self.config)
        issues = validator.validate([doc])
        errors = [i for i in issues if i.severity == Severity.ERROR and i.field == "content"]
        self.assertGreaterEqual(len(errors), 1)
        self.assertEqual(errors[0].category, ValidationCategory.SCHEMA)

    # 3. Invalid field type
    def test_03_invalid_field_type(self) -> None:
        raw = get_sample_valid_doc_raw()
        raw["audience"] = "Not a list, but a string"
        doc = Document(raw=raw)
        validator = SchemaValidator(self.config)
        issues = validator.validate([doc])
        errors = [i for i in issues if i.severity == Severity.ERROR and i.field == "audience"]
        self.assertGreaterEqual(len(errors), 1)

    # 4. Duplicate document ID
    def test_04_duplicate_document_id(self) -> None:
        doc1 = Document(raw=get_sample_valid_doc_raw("POL-DUP-001"))
        doc2 = Document(raw=get_sample_valid_doc_raw("POL-DUP-001"))
        validator = IdValidator(self.config)
        issues = validator.validate([doc1, doc2])
        errors = [i for i in issues if i.severity == Severity.ERROR and i.category == ValidationCategory.ID]
        self.assertGreaterEqual(len(errors), 1)
        self.assertIn("Duplicate document ID detected", errors[0].message)

    # 5. Empty content
    def test_05_empty_content(self) -> None:
        raw = get_sample_valid_doc_raw()
        raw["content"] = "   \n\t  "
        doc = Document(raw=raw)
        validator = ContentValidator(self.config)
        issues = validator.validate([doc])
        errors = [i for i in issues if i.severity == Severity.ERROR and i.field == "content"]
        self.assertGreaterEqual(len(errors), 1)
        self.assertIn("empty", errors[0].message.lower())

    # 6. Very short document
    def test_06_very_short_document(self) -> None:
        raw = get_sample_valid_doc_raw()
        raw["content"] = "Short note text."
        doc = Document(raw=raw)
        validator = ContentValidator(self.config)
        issues = validator.validate([doc])
        short_issues = [i for i in issues if i.field == "content" and "short" in i.message.lower()]
        self.assertGreaterEqual(len(short_issues), 1)

    # 7. Exact duplicate detection
    def test_07_exact_duplicate(self) -> None:
        doc1 = Document(raw=get_sample_valid_doc_raw("DOC-EXACT-001"))
        doc2 = Document(raw=get_sample_valid_doc_raw("DOC-EXACT-002"))
        # Same content, different ID
        validator = DuplicateDetector(self.config)
        issues = validator.validate([doc1, doc2])
        self.assertEqual(len(validator.exact_duplicate_groups), 1)
        group = validator.exact_duplicate_groups[0]
        self.assertEqual(group.canonical_document_id, "DOC-EXACT-001")
        self.assertIn("DOC-EXACT-002", group.documents)

    # 8. Near duplicate detection
    def test_08_near_duplicate(self) -> None:
        raw1 = get_sample_valid_doc_raw("DOC-NEAR-001")
        raw2 = get_sample_valid_doc_raw("DOC-NEAR-002")
        # Slightly alter the content with 1-2 words
        raw2["content"] = raw1["content"] + " All faculty advisors shall maintain continuous office hours."
        doc1 = Document(raw=raw1)
        doc2 = Document(raw=raw2)
        validator = DuplicateDetector(self.config)
        issues = validator.validate([doc1, doc2])
        near_pairs = validator.near_duplicate_pairs
        self.assertGreaterEqual(len(near_pairs), 1)
        self.assertGreaterEqual(near_pairs[0].similarity, 0.85)

    # 9. Invalid metadata
    def test_09_invalid_metadata(self) -> None:
        raw = get_sample_valid_doc_raw()
        raw["status"] = "NonExistentStatus"
        doc = Document(raw=raw)
        validator = MetadataValidator(self.config)
        issues = validator.validate([doc])
        errors = [i for i in issues if i.severity == Severity.ERROR and i.field == "status"]
        self.assertGreaterEqual(len(errors), 1)

    # 10. Valid metadata
    def test_10_valid_metadata(self) -> None:
        doc = Document(raw=get_sample_valid_doc_raw())
        validator = MetadataValidator(self.config)
        issues = validator.validate([doc])
        errors = [i for i in issues if i.severity == Severity.ERROR]
        self.assertEqual(len(errors), 0)

    # 11. Entity inconsistency
    def test_11_entity_inconsistency(self) -> None:
        raw1 = get_sample_valid_doc_raw("DOC-ENT-001")
        raw1["department_id"] = "DEPT-CSE"
        raw1["department"] = "Department of Computer Science"

        raw2 = get_sample_valid_doc_raw("DOC-ENT-002")
        raw2["department_id"] = "DEPT-CSE"
        raw2["department"] = "Department of Finance and Accounts"  # Conflict!

        validator = EntityValidator(self.config)
        issues = validator.validate([Document(raw=raw1), Document(raw=raw2)])
        warnings = [i for i in issues if i.category == ValidationCategory.ENTITY and i.field == "department"]
        self.assertGreaterEqual(len(warnings), 1)
        self.assertIn("inconsistently", warnings[0].message)

    # 12. Invalid ground-truth reference
    def test_12_invalid_ground_truth_reference(self) -> None:
        wm_file = Path(self.temp_dir) / "world_model.json"
        with open(wm_file, "w", encoding="utf-8") as f:
            json.dump({
                "departments": [{"department_id": "DEPT-ACAD", "name": "Academic Affairs"}],
                "systems": [{"name": "UniCampus ERP"}],
            }, f)

        cfg = ValidationConfig(world_model_path=str(wm_file))
        raw = get_sample_valid_doc_raw("DOC-GT-001")
        raw["department_id"] = "DEPT-NONEXISTENT"

        validator = GroundTruthValidator(cfg)
        issues = validator.validate([Document(raw=raw)])
        gt_issues = [i for i in issues if i.category == ValidationCategory.GROUND_TRUTH and i.field == "department_id"]
        self.assertGreaterEqual(len(gt_issues), 1)

    # 13. Valid ground-truth reference
    def test_13_valid_ground_truth_reference(self) -> None:
        wm_file = Path(self.temp_dir) / "world_model.json"
        with open(wm_file, "w", encoding="utf-8") as f:
            json.dump({
                "departments": [{"department_id": "DEPT-ACAD", "name": "Academic Affairs"}],
                "systems": [{"name": "UniCampus ERP"}, {"name": "ExamShield Portal"}],
                "roles": [{"name": "Dean of Academic Affairs"}],
            }, f)

        cfg = ValidationConfig(world_model_path=str(wm_file))
        raw = get_sample_valid_doc_raw("DOC-GT-002")
        validator = GroundTruthValidator(cfg)
        issues = validator.validate([Document(raw=raw)])
        gt_errors = [i for i in issues if i.severity == Severity.ERROR]
        self.assertEqual(len(gt_errors), 0)

    # 14. Corpus distribution calculation
    def test_14_corpus_distribution_calculation(self) -> None:
        raw1 = get_sample_valid_doc_raw("DOC-DIST-001")
        raw1["department"] = "Academic Affairs"
        raw2 = get_sample_valid_doc_raw("DOC-DIST-002")
        raw2["department"] = "Computer Science"

        validator = CoverageValidator(self.config)
        validator.validate([Document(raw=raw1), Document(raw=raw2)])
        depts = validator.distributions["departments"]
        self.assertEqual(depts.get("Academic Affairs"), 1)
        self.assertEqual(depts.get("Computer Science"), 1)
        self.assertGreater(validator.quality_metrics.average_word_count, 0)

    # 15. Validation report generation
    def test_15_validation_report_generation(self) -> None:
        doc = Document(raw=get_sample_valid_doc_raw("DOC-REP-001"))
        aggregator = ValidationAggregator(self.config)
        valid, rejected, issues, meta = aggregator.run([doc])
        report_json = ValidationReporter.generate_json_report("test-run-123", issues, meta)
        report_md = ValidationReporter.generate_markdown_report("test-run-123", issues, meta)

        self.assertEqual(report_json["run_id"], "test-run-123")
        self.assertIn("summary", report_json)
        self.assertIn("# Enterprise Corpus Validation Report", report_md)

    # 16. Rejected-document generation
    def test_16_rejected_document_generation(self) -> None:
        raw_bad = get_sample_valid_doc_raw("DOC-REJ-001")
        del raw_bad["content"]  # Fatal error
        doc_bad = Document(raw=raw_bad)
        doc_good = Document(raw=get_sample_valid_doc_raw("DOC-REJ-002"))

        aggregator = ValidationAggregator(self.config)
        valid_docs, rejected_docs, issues, meta = aggregator.run([doc_bad, doc_good])

        self.assertEqual(len(valid_docs), 1)
        self.assertEqual(valid_docs[0].document_id, "DOC-REJ-002")
        self.assertEqual(len(rejected_docs), 1)
        self.assertEqual(rejected_docs[0].document_id, "DOC-REJ-001")

    # 17. CLI execution
    def test_17_cli_execution(self) -> None:
        sample_corpus = Path(self.temp_dir) / "sample_corpus.json"
        with open(sample_corpus, "w", encoding="utf-8") as f:
            json.dump({"domain": "Test", "documents": [get_sample_valid_doc_raw()]}, f)

        out_dir = Path(self.temp_dir) / "out_validated"
        ret_code = main(["-i", str(sample_corpus), "-o", str(out_dir), "--log-level", "ERROR"])
        self.assertEqual(ret_code, 0)
        self.assertTrue((out_dir / "validated_documents.json").exists())
        self.assertTrue((out_dir / "validation_report.json").exists())

    # 18. Empty corpus
    def test_18_empty_corpus(self) -> None:
        empty_dir = Path(self.temp_dir) / "empty_dir"
        empty_dir.mkdir()
        docs, issues = CorpusLoader.load(empty_dir)
        self.assertEqual(len(docs), 0)
        self.assertTrue(any("No .json or .jsonl" in i.message for i in issues))

    # 19. Malformed input
    def test_19_malformed_input(self) -> None:
        malformed_file = Path(self.temp_dir) / "malformed.json"
        with open(malformed_file, "w", encoding="utf-8") as f:
            f.write("{ invalid json formatting")

        docs, issues = CorpusLoader.load(malformed_file)
        self.assertEqual(len(docs), 0)
        self.assertTrue(any("Malformed JSON" in i.message for i in issues))

    # 20. Mixed valid and invalid corpus
    def test_20_mixed_valid_invalid_corpus(self) -> None:
        raw2 = get_sample_valid_doc_raw("DOC-MIX-002")
        raw2["content"] = "1. PURPOSE AND SCOPE\nThis is a distinct valid second policy document providing university guidelines on student credentials and academic evaluations.\n\n2. REGULATIONS\nAll departments must enforce these regulations uniformly."
        good_doc1 = Document(raw=get_sample_valid_doc_raw("DOC-MIX-001"))
        good_doc2 = Document(raw=raw2)

        bad_doc1 = Document(raw={"document_id": "DOC-MIX-003"})  # missing title, content, dept
        bad_doc2 = Document(raw=get_sample_valid_doc_raw("DOC-MIX-004"))
        bad_doc2.raw["status"] = "INVALID_STATUS"

        aggregator = ValidationAggregator(self.config)
        valid_docs, rejected_docs, issues, meta = aggregator.run(
            [good_doc1, good_doc2, bad_doc1, bad_doc2]
        )

        self.assertEqual(len(valid_docs), 2)
        self.assertEqual({d.document_id for d in valid_docs}, {"DOC-MIX-001", "DOC-MIX-002"})
        self.assertEqual(len(rejected_docs), 2)
        self.assertEqual({d.document_id for d in rejected_docs}, {"DOC-MIX-003", "DOC-MIX-004"})


if __name__ == "__main__":
    unittest.main()

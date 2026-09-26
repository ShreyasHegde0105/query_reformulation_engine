"""Tests for the reformulation foundation (validation & normalization).

These tests run offline and use only the local implementation.  They verify
that the public API behaves exactly as specified in the task description.

The normalization tests are split into two classes:
- ``TestReformulationFoundationLegacy``: the original whitespace / punctuation
  tests, updated to account for the new sentence-case behaviour.
- ``TestNormalizationExtended``: the 14 new test cases covering casing, typos,
  abbreviations, entity preservation, punctuation deduplication, and edge cases.
"""

import os
import unittest
from typing import Any

# Import the public API from the package
from reformulation import validate_query, normalize_query


class TestReformulationFoundation(unittest.TestCase):
    # ----- Validation tests -----
    def test_valid_normal_query(self):
        # Simple non‑empty query should pass without exception
        try:
            validate_query("What is the capital of France?")
        except Exception as e:
            self.fail(f"validate_query raised unexpectedly: {e}")

    def test_empty_string(self):
        with self.assertRaises(ValueError) as cm:
            validate_query("")
        self.assertIn("empty", str(cm.exception).lower())

    def test_whitespace_only(self):
        with self.assertRaises(ValueError):
            validate_query("   \t\n   ")

    def test_none_input(self):
        with self.assertRaises(ValueError):
            validate_query(None)

    def test_integer_input(self):
        with self.assertRaises(ValueError):
            validate_query(123)

    def test_list_input(self):
        with self.assertRaises(ValueError):
            validate_query(["a", "b"])

    def test_dict_input(self):
        with self.assertRaises(ValueError):
            validate_query({"q": "test"})

    def test_maximum_allowed_length(self):
        max_len = int(os.getenv("MAX_QUERY_LENGTH", "256"))
        query = "x" * max_len
        # Should be accepted (no exception)
        try:
            validate_query(query)
        except Exception as e:
            self.fail(f"validate_query rejected max‑length query: {e}")

    def test_exceeding_maximum_length(self):
        max_len = int(os.getenv("MAX_QUERY_LENGTH", "256"))
        query = "x" * (max_len + 1)
        with self.assertRaises(ValueError) as cm:
            validate_query(query)
        self.assertIn("exceeds", str(cm.exception).lower())

    # ----- Normalization tests (whitespace / punctuation – updated for sentence case) -----
    def test_trim_whitespace(self):
        # Leading/trailing whitespace stripped; sentence case applied.
        result = normalize_query("   hello world   ")
        self.assertEqual(result.lower(), "hello world")
        # First word capitalised
        self.assertTrue(result[0].isupper(), f"Expected first char uppercase, got: {result!r}")

    def test_collapse_multiple_spaces(self):
        result = normalize_query("hello   world")
        self.assertEqual(result.lower(), "hello world")

    def test_tabs_and_newlines(self):
        result = normalize_query("hello\tworld\nnew")
        self.assertEqual(result.lower(), "hello world new")

    def test_punctuation_spacing(self):
        # Space before punctuation removed; sentence case applied.
        result = normalize_query("hello , world !")
        self.assertNotIn(" ,", result)
        self.assertNotIn(" !", result)
        self.assertIn(",", result)

    def test_semantic_unchanged(self):
        raw = "Find AI/ML papers from 2022."
        norm = normalize_query(raw)
        # Should keep numbers, slash and period
        self.assertIn("2022", norm)
        self.assertIn("/", norm)

    def test_preserve_entities(self):
        raw = "Explain the theory of relativity by Einstein."
        self.assertIn("Einstein", normalize_query(raw))

    def test_preserve_technical_terms(self):
        raw = "What is a transformer model in NLP?"
        self.assertIn("transformer", normalize_query(raw).lower())

    def test_preserve_numbers(self):
        raw = "Show me stats for 2020 vs 2021."
        self.assertIn("2020", normalize_query(raw))
        self.assertIn("2021", normalize_query(raw))

    def test_preserve_acronyms(self):
        raw = "Explain CPU vs GPU performance."
        self.assertIn("CPU", normalize_query(raw))
        self.assertIn("GPU", normalize_query(raw))

    def test_normalization_empty_result(self):
        # Normalising an input that becomes empty should raise via validation
        with self.assertRaises(ValueError):
            # The validation will catch the empty after stripping
            validate_query(normalize_query("   "))

    # ----- Integration tests -----
    def test_valid_raw_query_flow(self):
        raw = "   Retrieve   documents about   AI.   "
        # Validation first
        validate_query(raw)
        # Normalization second
        norm = normalize_query(raw)
        # Content preserved; whitespace collapsed; AI acronym kept
        self.assertIn("AI", norm)
        self.assertIn("Retrieve", norm)
        self.assertIn("documents", norm)

    def test_invalid_query_never_normalized(self):
        raw = "   "
        with self.assertRaises(ValueError):
            validate_query(raw)
        # Ensure that normalization is *not* called – we simply verify that no
        # exception from normalize_query would be raised because we never call it.
        # (The test passes if the above block raises as expected.)
        self.assertTrue(True)


# ---------------------------------------------------------------------------
# Extended normalization tests (14 new cases from the task specification)
# ---------------------------------------------------------------------------

class TestNormalizationExtended(unittest.TestCase):
    """14 normalization test cases specified in the task description."""

    # Test 1 – Already clean query
    def test_01_already_clean_query(self):
        inp = "How do I check my grades?"
        out = normalize_query(inp)
        self.assertEqual(out, "How do I check my grades?")

    # Test 2 – Excessive whitespace
    def test_02_excessive_whitespace(self):
        inp = "   How   do   I   check   my   grades?   "
        out = normalize_query(inp)
        self.assertEqual(out, "How do I check my grades?")

    # Test 3 – Mixed capitalization
    def test_03_mixed_capitalization(self):
        inp = "HOW TO CHeck my Grades FROm last SEM?"
        out = normalize_query(inp)
        self.assertEqual(out, "How to check my grades from last semester?")

    # Test 4 – Typo only
    def test_04_typo_correction(self):
        inp = "How to check my grades from laast semester?"
        out = normalize_query(inp)
        self.assertEqual(out, "How to check my grades from last semester?")

    # Test 5 – Typo + abbreviation (the main motivating example)
    def test_05_typo_and_abbreviation(self):
        inp = "HOW TO CHeck my Grades FROm laast sem?"
        out = normalize_query(inp)
        self.assertEqual(out, "How to check my grades from last semester?")

    # Test 6 – Multiple punctuation
    def test_06_multiple_punctuation(self):
        inp = "What are the university's academic integrity rules???"
        out = normalize_query(inp)
        self.assertEqual(out, "What are the university's academic integrity rules?")

    # Test 7 – Preserve proper noun (GitHub)
    def test_07_preserve_proper_noun_github(self):
        inp = "How do I access GitHub?"
        out = normalize_query(inp)
        self.assertEqual(out, "How do I access GitHub?")

    # Test 8 – Preserve course code (CSE 301)
    def test_08_preserve_course_code(self):
        inp = "Where can I find the CSE 301 syllabus?"
        out = normalize_query(inp)
        self.assertEqual(out, "Where can I find the CSE 301 syllabus?")

    # Test 9 – Common abbreviation expansion
    def test_09_abbreviation_sem(self):
        inp = "How do I check my grades from last sem?"
        out = normalize_query(inp)
        self.assertEqual(out, "How do I check my grades from last semester?")

    # Test 10 – All-caps academic policy query
    def test_10_all_caps_policy(self):
        inp = "WHAT ARE THE UNIVERSITY'S RULES REGARDING ACADEMIC INTEGRITY?"
        out = normalize_query(inp)
        self.assertEqual(out, "What are the university's rules regarding academic integrity?")

    # Test 11 – Preserve meaning (no semantic transformation)
    def test_11_preserve_meaning(self):
        inp = "Can I retake my exam if I failed?"
        out = normalize_query(inp)
        self.assertEqual(out, "Can I retake my exam if I failed?")

    # Test 12 – Empty/whitespace input
    def test_12_empty_whitespace_input(self):
        # normalize_query returns "" for whitespace-only input;
        # validate_query rejects it with ValueError.
        self.assertEqual(normalize_query("     "), "")
        with self.assertRaises((ValueError, TypeError)):
            validate_query(normalize_query("     "))

    # Test 13 – Preserve numbers and dates
    def test_13_preserve_numbers_dates(self):
        inp = "What was my GPA in 2025?"
        out = normalize_query(inp)
        self.assertEqual(out, "What was my GPA in 2025?")

    # Test 14 – Preserve mixed technical terms (CSE, GPA, ERP)
    def test_14_preserve_mixed_technical_terms(self):
        inp = "How do I access my CSE GPA through the ERP portal?"
        out = normalize_query(inp)
        self.assertEqual(out, "How do I access my CSE GPA through the ERP portal?")


# ---------------------------------------------------------------------------
# Regression tests: verify the separator between normalization and reformulation
# ---------------------------------------------------------------------------

class TestNormalizationVsReformulation(unittest.TestCase):
    """Ensure normalization does NOT perform semantic reformulation."""

    def test_no_synonym_generation(self):
        inp = "How do I submit my assignment?"
        out = normalize_query(inp)
        # Should not introduce synonyms like "upload", "turn in", etc.
        self.assertNotIn("upload", out.lower())
        self.assertNotIn("turn in", out.lower())

    def test_no_answer_generation(self):
        inp = "What is the grading policy?"
        out = normalize_query(inp)
        # Must remain a question, not an answer
        self.assertIn("grading", out.lower())
        self.assertIn("policy", out.lower())

    def test_clean_query_unchanged(self):
        """A perfectly clean query should be returned unchanged."""
        inp = "Where can I find the hostel rules?"
        out = normalize_query(inp)
        self.assertEqual(out, inp)

    def test_deterministic(self):
        """Calling normalize_query twice with the same input must give the same output."""
        inp = "HOW TO CHeck my Grades FROm laast sem?"
        self.assertEqual(normalize_query(inp), normalize_query(inp))

    def test_original_query_not_mutated(self):
        """The original query string object must be unchanged after normalization."""
        original = "HOW TO CHeck my Grades FROm laast sem?"
        original_copy = original  # same object reference
        _ = normalize_query(original)
        self.assertEqual(original, original_copy)


# ---------------------------------------------------------------------------
# Comprehensive Robustness Matrix: 30 Problem Classes & Adversarial Inputs
# ---------------------------------------------------------------------------

class TestRobustNormalizationMatrix(unittest.TestCase):
    """Systematic coverage for all 30 problem classes and adversarial user inputs."""

    # 1. Clean query
    def test_cat01_clean_query(self):
        self.assertEqual(normalize_query("How do I check my grades?"), "How do I check my grades?")
        self.assertEqual(normalize_query("Where can I find the hostel rules?"), "Where can I find the hostel rules?")

    # 2. Leading/trailing whitespace
    def test_cat02_leading_trailing_whitespace(self):
        self.assertEqual(normalize_query("   How do I check my grades?   "), "How do I check my grades?")

    # 3. Repeated internal whitespace
    def test_cat03_repeated_whitespace(self):
        self.assertEqual(normalize_query("How   do   I   check   my grades?"), "How do I check my grades?")

    # 4. Tabs
    def test_cat04_tabs(self):
        self.assertEqual(normalize_query("How\tdo\tI\tcheck\tmy grades?"), "How do I check my grades?")

    # 5. Newlines
    def test_cat05_newlines(self):
        self.assertEqual(normalize_query("How do I\ncheck my grades?"), "How do I check my grades?")

    # 6. ALL CAPS
    def test_cat06_all_caps(self):
        self.assertEqual(normalize_query("HOW TO CHECK MY GRADES"), "How to check my grades")

    # 7. Mixed casing
    def test_cat07_mixed_casing(self):
        self.assertEqual(normalize_query("HoW To ChEcK My GrAdEs"), "How to check my grades")

    # 8. Lowercase
    def test_cat08_lowercase(self):
        self.assertEqual(normalize_query("how to check my grades"), "How to check my grades")

    # 9. Obvious typos
    def test_cat09_obvious_typos(self):
        self.assertEqual(normalize_query("Where is the universty polcy?"), "Where is the university policy?")
        self.assertEqual(normalize_query("acadmic report card"), "Academic report card")

    # 10. Multiple typos & character elongation
    def test_cat10_multiple_typos_elongation(self):
        self.assertEqual(normalize_query("HOWWWW TO CHECK MY GRDAES???"), "How to check my grades?")
        self.assertEqual(normalize_query("pleaaase check my semster grdaes"), "Please check my semester grades")

    # 11. Controlled abbreviation expansion
    def test_cat11_abbreviations(self):
        self.assertEqual(normalize_query("contact the dept prof"), "Contact the department professor")
        self.assertEqual(normalize_query("last sem acad calendar"), "Last semester academic calendar")

    # 12. Ambiguous abbreviations preserved
    def test_cat12_ambiguous_abbreviations_preserved(self):
        self.assertEqual(normalize_query("Can I apply for CS or IT?"), "Can I apply for CS or IT?")
        self.assertEqual(normalize_query("What are the rules for UG and PG?"), "What are the rules for UG and PG?")

    # 13. Repeated punctuation
    def test_cat13_repeated_punctuation(self):
        self.assertEqual(normalize_query("What are the rules???"), "What are the rules?")
        self.assertEqual(normalize_query("Academic integrity!!!"), "Academic integrity!")

    # 14. Malformed punctuation
    def test_cat14_malformed_punctuation(self):
        self.assertEqual(normalize_query(")How do I check my grades?"), "How do I check my grades?")
        self.assertEqual(normalize_query("How )do I check my grades?"), "How do I check my grades?")
        self.assertEqual(normalize_query("How do I check my grades?!!!"), "How do I check my grades?")
        self.assertEqual(normalize_query("How do I check my grades..."), "How do I check my grades?")
        self.assertEqual(normalize_query("What are the rules ?!"), "What are the rules?")

    # 15. Punctuation attached to tokens
    def test_cat15_punctuation_attached_to_tokens(self):
        self.assertEqual(normalize_query("How to ,check my grades"), "How to check my grades")
        self.assertEqual(normalize_query("((grades))"), "Grades")
        self.assertEqual(normalize_query('"grades"'), '"Grades"')

    # 16. Possessives
    def test_cat16_possessives(self):
        self.assertEqual(normalize_query("student's grade card"), "Student's grade card")
        self.assertEqual(normalize_query("WHAT ARE THE UNIVERSITY'S RULES?"), "What are the university's rules?")

    # 17. Contractions
    def test_cat17_contractions(self):
        self.assertEqual(normalize_query("I'M looking for my grades"), "I'm looking for my grades")
        self.assertEqual(normalize_query("WHAT'S the attendance policy?"), "What's the attendance policy?")
        self.assertEqual(normalize_query("CAN'T find my syllabus"), "Can't find my syllabus")

    # 18. Proper nouns
    def test_cat18_proper_nouns(self):
        self.assertEqual(normalize_query("How do I access GitHub?"), "How do I access GitHub?")
        self.assertEqual(normalize_query("Connect with alumni on LinkedIn and YouTube"), "Connect with alumni on LinkedIn and YouTube")
        self.assertEqual(normalize_query("Explain the theory of relativity by Einstein."), "Explain the theory of relativity by Einstein.")

    # 19. Acronyms
    def test_cat19_acronyms(self):
        self.assertEqual(normalize_query("access ERP portal for GPA"), "Access ERP portal for GPA")
        self.assertEqual(normalize_query("What is a transformer model in NLP?"), "What is a transformer model in NLP?")
        self.assertEqual(normalize_query("Explain CPU vs GPU performance."), "Explain CPU vs GPU performance.")

    # 20. Course codes
    def test_cat20_course_codes(self):
        self.assertEqual(normalize_query("Where can I find the CSE 301 syllabus?"), "Where can I find the CSE 301 syllabus?")
        self.assertEqual(normalize_query("what is the CSE-301 policy?"), "What is the CSE-301 policy?")
        self.assertEqual(normalize_query("rules for cse301 and cs101"), "Rules for CSE301 and CS101")

    # 21. Document IDs
    def test_cat21_document_ids(self):
        self.assertEqual(normalize_query("Where is ACAD-REG-001?"), "Where is ACAD-REG-001?")
        self.assertEqual(normalize_query("view CSE-CRS-304 lab manual"), "View CSE-CRS-304 lab manual")

    # 22. Dates
    def test_cat22_dates(self):
        self.assertEqual(normalize_query("exam schedule for 2025-09-20"), "Exam schedule for 2025-09-20")
        self.assertEqual(normalize_query("deadline on 20/09/2025"), "Deadline on 20/09/2025")
        self.assertEqual(normalize_query("calendar for September 2025"), "Calendar for September 2025")

    # 23. Numbers & academic years
    def test_cat23_numbers_and_years(self):
        self.assertEqual(normalize_query("What was my GPA in 2025?"), "What was my GPA in 2025?")
        self.assertEqual(normalize_query("what is CSE301 policy for 2025/26?"), "What is CSE301 policy for 2025/26?")
        self.assertEqual(normalize_query("Show me stats for 2020 vs 2021."), "Show me stats for 2020 vs 2021.")

    # 24. Short queries
    def test_cat24_short_queries(self):
        self.assertEqual(normalize_query("GPA"), "GPA")
        self.assertEqual(normalize_query("CSE 301"), "CSE 301")
        self.assertEqual(normalize_query("last sem"), "Last semester")

    # 25. Empty & whitespace queries
    def test_cat25_empty_whitespace_queries(self):
        self.assertEqual(normalize_query(""), "")
        self.assertEqual(normalize_query("   "), "")

    # 26. Unknown terms preserved
    def test_cat26_unknown_terms_preserved(self):
        self.assertEqual(normalize_query("Where is ABCD portal?"), "Where is ABCD portal?")
        self.assertEqual(normalize_query("Where is the xyz policy?"), "Where is the xyz policy?")
        self.assertEqual(normalize_query("abcXYZ123"), "abcXYZ123")
        self.assertEqual(normalize_query("rules for XJ-301"), "Rules for XJ-301")
        self.assertEqual(normalize_query("guide to Kubernetes in cloud"), "Guide to Kubernetes in cloud")

    # 27. Technical terminology
    def test_cat27_technical_terminology(self):
        self.assertEqual(normalize_query("Find AI/ML papers from 2022."), "Find AI/ML papers from 2022.")

    # 28. Mixed entity + typo adversarial queries
    def test_cat28_mixed_entity_typo_adversarial(self):
        inp = "HoW  )To Check my las   year grade card or last sem marks card"
        expected = "How to check my last year grade card or last semester marks card"
        self.assertEqual(normalize_query(inp), expected)

        inp2 = "   where   can i   find my CSE 301   syllabus ???   "
        expected2 = "Where can I find my CSE 301 syllabus?"
        self.assertEqual(normalize_query(inp2), expected2)

    # 29. Idempotence across diverse queries
    def test_cat29_idempotence(self):
        queries = [
            "HoW  )To Check my las   year grade card or last sem marks card",
            "HOWWWW TO CHECK MY GRDAES???",
            "   where   can i   find my CSE 301   syllabus ???",
            "can i get my GPA for 2025???",
            "WHAT ARE THE UNIVERSITY'S RULES!!!",
            "how do i access github???",
            "How )do I check my grades?",
            "what is the CSE-301 policy?",
            "what is CSE301 policy for 2025/26?",
            "GPA",
            "CSE 301",
            "last sem",
            "abcXYZ123",
            "Where is ABCD portal?",
            "Can I retake my exam if I failed???",
        ]
        for q in queries:
            once = normalize_query(q)
            twice = normalize_query(once)
            self.assertEqual(once, twice, f"Idempotence failed for query: {q!r} -> {once!r} vs {twice!r}")

    # 30. Meaning preservation
    def test_cat30_meaning_preservation(self):
        inp = "Can I retake my exam if I failed???"
        out = normalize_query(inp)
        self.assertEqual(out, "Can I retake my exam if I failed?")
        self.assertNotIn("improve", out.lower())


class TestGeneralNoiseSuite(unittest.TestCase):
    """Test suite covering robust handling of noisy human input (whitespace, casing,
    leetspeak/character substitution, accidental surrounding characters, and entity preservation).
    """

    def test_repeated_whitespace(self):
        self.assertEqual(
            normalize_query("How   many years do I have?"),
            "How many years do I have?"
        )

    def test_mixed_casing(self):
        self.assertEqual(
            normalize_query("HOW MANY YEARS I HAVe?"),
            "How many years I have?"
        )

    def test_surrounding_special_characters(self):
        self.assertEqual(
            normalize_query("#Engineering@"),
            "Engineering"
        )

    def test_combined_noise(self):
        self.assertEqual(
            normalize_query("How  many years I HAVe before C0mpleting #Engineering@"),
            "How many years I have before completing Engineering"
        )

    def test_meaningful_hyphen_preserved(self):
        out = normalize_query("What is the CSE-301 policy?")
        self.assertEqual(out, "What is the CSE-301 policy?")
        self.assertIn("CSE-301", out)

    def test_meaningful_slash_preserved(self):
        out = normalize_query("What are the rules for 2025/26?")
        self.assertEqual(out, "What are the rules for 2025/26?")
        self.assertIn("2025/26", out)

    def test_meaningful_technical_punctuation_cpp(self):
        out = normalize_query("How do I use C++?")
        self.assertEqual(out, "How do I use C++?")
        self.assertIn("C++", out)

    def test_email_like_entity_preserved(self):
        out = normalize_query("Contact admin@example.com")
        self.assertEqual(out, "Contact admin@example.com")
        self.assertNotIn("admin example com", out)

    def test_identifier_preserved(self):
        out = normalize_query("Find document ABC-123@2025")
        self.assertEqual(out, "Find document ABC-123@2025")
        self.assertIn("ABC-123@2025", out)

    def test_hashtag_entity_preserved(self):
        out = normalize_query("Where is #CSE lab?")
        self.assertEqual(out, "Where is #CSE lab?")
        self.assertIn("#CSE", out)

    def test_accidental_surrounding_tags(self):
        self.assertEqual(
            normalize_query("Check #semester @grades"),
            "Check semester grades"
        )

    def test_enclosing_brackets_stripped(self):
        self.assertEqual(
            normalize_query("Where can I see ((grades)) and [courses]?"),
            "Where can I see grades and courses?"
        )

    def test_exclamation_noise_collapsed(self):
        self.assertEqual(
            normalize_query("I need my grades!!!"),
            "I need my grades!"
        )

    def test_leetspeak_negative_cases_not_corrupted(self):
        # Course codes and numbers must NOT have digits converted to letters
        self.assertIn("CSE301", normalize_query("Where is CSE301 syllabus?"))
        self.assertIn("CS101", normalize_query("What is CS101 prerequisite?"))
        self.assertIn("2025", normalize_query("Rules for 2025 admissions"))
        self.assertIn("ABC123", normalize_query("Document ABC123 status"))


class TestNormalizerTargetedCorrections(unittest.TestCase):
    """Regression tests for targeted normalizer corrections."""

    # 1. Acronyms with repeated characters preserved before elongation collapse
    def test_ieee_acronym_preserved(self):
        self.assertEqual(normalize_query("IEEE conference papers"), "IEEE conference papers")
        self.assertEqual(normalize_query("IEEE standards"), "IEEE standards")

    def test_iiit_acronym_preserved(self):
        self.assertEqual(normalize_query("IIIT admissions"), "IIIT admissions")
        self.assertEqual(normalize_query("IIIT cutoff"), "IIIT cutoff")

    # 2. Uppercase acronyms protected from typo map
    def test_las_acronym_protected(self):
        self.assertEqual(normalize_query("LAS department courses"), "LAS department courses")
        self.assertEqual(normalize_query("Dean of LAS"), "Dean of LAS")

    def test_las_lowercase_typo_still_corrected(self):
        self.assertEqual(normalize_query("las sem marks"), "Last semester marks")

    # 3. Dangerous typo correction hospitel -> hospital (not hostel)
    def test_hospitel_corrected_to_hospital(self):
        self.assertEqual(normalize_query("university hospitel bills"), "University hospital bills")
        self.assertEqual(normalize_query("leave for hospitel visit"), "Leave for hospital visit")

    # 4. Dangerous std abbreviation removed (not expanded to student)
    def test_std_abbreviation_not_expanded_to_student(self):
        self.assertEqual(normalize_query("std deviation for grade calculation"), "Std deviation for grade calculation")
        self.assertEqual(normalize_query("std format for thesis"), "Std format for thesis")
        self.assertEqual(normalize_query("std operating procedure"), "Std operating procedure")

    # 5. Sentence casing on leading ordinals and numbers
    def test_leading_ordinal_and_number_casing(self):
        self.assertEqual(normalize_query("2nd semester fees"), "2nd semester fees")
        self.assertEqual(normalize_query("3rd semester marks"), "3rd semester marks")
        self.assertEqual(normalize_query("1st year fees"), "1st year fees")
        self.assertEqual(normalize_query("75 percent attendance"), "75 percent attendance")
        self.assertEqual(normalize_query("10 credits"), "10 credits")

    # 6. Missing common university typos
    def test_missing_university_typos_corrected(self):
        self.assertEqual(normalize_query("attendance reqmnt"), "Attendance requirement")
        self.assertEqual(normalize_query("hostel allotmnt process"), "Hostel allotment process")
        self.assertEqual(normalize_query("revalution policy"), "Revaluation policy")
        self.assertEqual(normalize_query("attendence shortage"), "Attendance shortage")

    # 7. Course code canonicalization
    def test_course_code_canonicalization(self):
        self.assertEqual(normalize_query("cse301 prerequisites"), "CSE301 prerequisites")
        self.assertEqual(normalize_query("Cse301 prerequisites"), "CSE301 prerequisites")
        self.assertEqual(normalize_query("CSE301 prerequisites"), "CSE301 prerequisites")
        self.assertEqual(normalize_query("CSE 301 prerequisites"), "CSE 301 prerequisites")
        self.assertEqual(normalize_query("CSE-301 prerequisites"), "CSE-301 prerequisites")
        self.assertEqual(normalize_query("Cs101 syllabus"), "CS101 syllabus")

    # 8. Idempotence across all regression queries
    def test_regression_queries_idempotence(self):
        queries = [
            "IEEE conference papers",
            "IIIT admissions",
            "LAS department courses",
            "Dean of LAS",
            "university hospitel bills",
            "leave for hospitel visit",
            "std deviation for grade calculation",
            "2nd semester fees",
            "3rd semester marks",
            "1st year fees",
            "75 percent attendance",
            "10 credits",
            "attendance reqmnt",
            "hostel allotmnt process",
            "revalution policy",
            "attendence shortage",
            "cse301 prerequisites",
            "Cse301 prerequisites",
            "Cs101 syllabus",
        ]
        for q in queries:
            once = normalize_query(q)
            twice = normalize_query(once)
            self.assertEqual(once, twice, f"Idempotence failed for {q!r}")


if __name__ == "__main__":
    unittest.main()



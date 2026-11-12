"""Tests for intent-gated adaptive query routing and regression protections."""

import unittest
from unittest.mock import MagicMock

from reformulation.intent_router import IntentRouter, QueryIntent, RoutingAction
from reformulation.normalizer import normalize_query
from reformulation.search_pipeline import SearchPipeline
from reformulation.spelling import CorpusSpellCorrector
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.models import RetrievedChunk


class TestIntentRouterClassifications(unittest.TestCase):
    """Verify deterministic intent classification across all 9 intent classes."""

    @classmethod
    def setUpClass(cls):
        # Initialize populated router from corpus chunks
        cls.router = IntentRouter.from_chunks_file()

    def test_course_code_intent(self):
        res1 = self.router.route("MEC204")
        self.assertEqual(res1.intent, QueryIntent.COURSE_CODE)
        self.assertEqual(res1.action, RoutingAction.DIRECT)
        self.assertEqual(res1.matched_entity, "MEC204")

        res2 = self.router.route("ECE302 VLSI EDA tools")
        self.assertEqual(res2.intent, QueryIntent.COURSE_CODE)
        self.assertEqual(res2.action, RoutingAction.DIRECT)
        self.assertEqual(res2.matched_entity, "ECE302")

    def test_document_id_intent(self):
        res = self.router.route("HSTL-REG-003")
        self.assertEqual(res.intent, QueryIntent.DOCUMENT_ID)
        self.assertEqual(res.action, RoutingAction.DIRECT)
        self.assertEqual(res.matched_entity, "HSTL-REG-003")

    def test_exact_identifier_intent(self):
        res = self.router.route("STU-FRM-003")
        self.assertEqual(res.intent, QueryIntent.EXACT_IDENTIFIER)
        self.assertEqual(res.action, RoutingAction.DIRECT)

    def test_exact_title_intent(self):
        res1 = self.router.route("hostel silence hours")
        self.assertEqual(res1.intent, QueryIntent.EXACT_TITLE)
        self.assertEqual(res1.action, RoutingAction.DIRECT)

        res2 = self.router.route("rules about silence hours in hostel")
        self.assertEqual(res2.intent, QueryIntent.EXACT_TITLE)
        self.assertEqual(res2.action, RoutingAction.DIRECT)

        res3 = self.router.route("hostel curfew policy")
        self.assertEqual(res3.intent, QueryIntent.EXACT_TITLE)
        self.assertEqual(res3.action, RoutingAction.DIRECT)

    def test_abbreviation_intent(self):
        res = self.router.route("intl students FRRO reg")
        self.assertEqual(res.intent, QueryIntent.ABBREVIATION)
        self.assertEqual(res.action, RoutingAction.REFORMULATE)

    def test_procedural_query_intent(self):
        res = self.router.route("how do I obtain a migration certificate")
        self.assertEqual(res.intent, QueryIntent.PROCEDURAL_QUERY)
        self.assertEqual(res.action, RoutingAction.REFORMULATE)

    def test_keyword_lookup_intent(self):
        res1 = self.router.route("student academic report card")
        self.assertEqual(res1.intent, QueryIntent.KEYWORD_LOOKUP)
        self.assertEqual(res1.action, RoutingAction.REFORMULATE)

        res2 = self.router.route("fee refund")
        self.assertEqual(res2.intent, QueryIntent.KEYWORD_LOOKUP)
        self.assertEqual(res2.action, RoutingAction.REFORMULATE)

    def test_natural_language_intent(self):
        res = self.router.route("what are the eligibility requirements for scholarships")
        self.assertEqual(res.intent, QueryIntent.NATURAL_LANGUAGE)
        self.assertEqual(res.action, RoutingAction.REFORMULATE)

    def test_ambiguous_intent(self):
        res = self.router.route("query")
        self.assertEqual(res.intent, QueryIntent.AMBIGUOUS)
        self.assertEqual(res.action, RoutingAction.SAFE_FALLBACK)


class TestNormalizerRegressionProtection(unittest.TestCase):
    """Verify normalizer does not destroy domain abbreviations or convert 'intl' to 'Into'."""

    def test_intl_students_frro_reg_not_into(self):
        normalized = normalize_query("intl students FRRO reg")
        self.assertNotIn("into", normalized.lower().split())
        self.assertIn("international", normalized.lower())
        self.assertIn("frro", normalized.lower())
        self.assertIn("reg", normalized.lower())

    def test_spell_corrector_preserves_intl_and_stopwords(self):
        vocab = {"into": 100, "international": 50, "students": 80, "regulations": 40}
        corrector = CorpusSpellCorrector(vocabulary=vocab)
        # 'intl' should never be corrected to stopword 'into'
        corrected = corrector.correct("intl")
        self.assertNotEqual(corrected, "into")


class TestSearchPipelineRoutingIntegration(unittest.TestCase):
    """Verify SearchPipeline honors IntentRouter decisions and records observability trace."""

    def setUp(self):
        self.reformulator = MagicMock()
        self.reformulator.generate_variants.return_value = ["variant 1", "variant 2"]
        self.sem = MagicMock()
        self.sem.filter_redundant.side_effect = lambda qs, original_query=None: qs
        self.hybrid = MagicMock(spec=HybridRetriever)
        self.hybrid.retrieve.return_value = [
            RetrievedChunk(
                chunk_id="c1",
                document_id="d1",
                title="T",
                domain="D",
                chunk_index=0,
                chunk_text="text",
                section=None,
                token_count=10,
                score=1.0,
            )
        ]
        self.pipeline = SearchPipeline(
            self.reformulator,
            self.sem,
            self.hybrid,
            max_workers=1,
            intent_router=IntentRouter.from_chunks_file(),
        )

    def test_exact_identifier_bypasses_reformulation(self):
        res = self.pipeline.search_detailed("MEC204")
        self.assertEqual(res.detected_intent, QueryIntent.COURSE_CODE.value)
        self.assertEqual(res.routing_decision, RoutingAction.DIRECT.value)
        self.assertFalse(res.reformulation_invoked)
        self.assertEqual(res.reformulations, [])
        self.reformulator.generate_variants.assert_not_called()
        self.sem.filter_redundant.assert_not_called()
        # Direct retrieval was invoked on normalized query
        self.assertEqual(self.hybrid.retrieve.call_count, 1)

    def test_natural_language_triggers_reformulation(self):
        res = self.pipeline.search_detailed("what are the eligibility requirements for scholarships")
        self.assertEqual(res.detected_intent, QueryIntent.NATURAL_LANGUAGE.value)
        self.assertEqual(res.routing_decision, RoutingAction.REFORMULATE.value)
        self.assertTrue(res.reformulation_invoked)
        self.reformulator.generate_variants.assert_called_once()
        self.sem.filter_redundant.assert_called_once()

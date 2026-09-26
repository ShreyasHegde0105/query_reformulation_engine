import unittest
from unittest.mock import Mock
from reformulation.semantic_similarity import SemanticSimilarity, FilterDecision

class TestSemanticSimilarity(unittest.TestCase):
    def setUp(self):
        # Simple deterministic fake embedding function
        # Map specific strings to vectors for controlled similarity tests
        self.fake_embeddings = {
            "identical": [1.0, 0.0, 0.0],
            "orthogonal": [0.0, 1.0, 0.0],
            "opposite": [-1.0, 0.0, 0.0],
            "similar_a": [1.0, 0.0, 0.0],
            "similar_b": [0.99, 0.01, 0.0],
            "different": [0.0, 0.0, 1.0],
        }
        self.embed_mock = Mock(side_effect=lambda q: self.fake_embeddings[q])
        self.similarity = SemanticSimilarity(threshold=0.85, embed_fn=self.embed_mock)

    # 1. Identical embeddings -> similarity ~1.0
    def test_identical_similarity(self):
        sim = self.similarity.similarity("identical", "identical")
        self.assertAlmostEqual(sim, 1.0, places=5)

    # 2. Orthogonal embeddings -> similarity 0.0
    def test_orthogonal_similarity(self):
        sim = self.similarity.similarity("identical", "orthogonal")
        self.assertAlmostEqual(sim, 0.0, places=5)

    # 3. Opposite embeddings -> similarity -1.0
    def test_opposite_similarity(self):
        sim = self.similarity.similarity("identical", "opposite")
        self.assertAlmostEqual(sim, -1.0, places=5)

    # 4. Threshold behavior
    def test_threshold_behavior(self):
        # similar_a and similar_b have high similarity (>0.85)
        self.assertTrue(self.similarity.are_similar("similar_a", "similar_b"))
        # similar_a vs different is low similarity
        self.assertFalse(self.similarity.are_similar("similar_a", "different"))

    # 5. Invalid threshold values
    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):
            SemanticSimilarity(threshold=-1.5)
        with self.assertRaises(ValueError):
            SemanticSimilarity(threshold=1.2)

    # 6. Empty query list handling
    def test_empty_list(self):
        self.assertEqual(self.similarity.filter_redundant([]), [])

    # 7. Single query handling
    def test_single_query(self):
        result = self.similarity.filter_redundant(["identical"])
        self.assertEqual(result, ["identical"])

    # 8. Duplicate semantic queries filtered
    def test_duplicate_semantic_filter(self):
        queries = ["similar_a", "similar_b", "different"]
        result = self.similarity.filter_redundant(queries)
        # similar_b should be considered redundant with similar_a
        self.assertEqual(result, ["similar_a", "different"])

    # 9. Deterministic order-independent selection between redundant queries
    def test_order_independent_selection(self):
        queries = ["similar_b", "similar_a", "different"]
        result = self.similarity.filter_redundant(queries)
        # Content-based deterministic selection chooses similar_a regardless of input order
        self.assertEqual(result, ["similar_a", "different"])
        self.assertEqual(
            set(result),
            set(self.similarity.filter_redundant(["similar_a", "similar_b", "different"]))
        )

    # 10. Transitive-like case (B similar to C, A distinct)
    def test_transitive_like(self):
        # Define vectors where B ~ C but A distinct from both
        self.fake_embeddings.update({
            "A": [1.0, 0.0, 0.0],
            "B": [0.0, 1.0, 0.0],
            "C": [0.0, 0.99, 0.01],
        })
        similarity = SemanticSimilarity(threshold=0.9, embed_fn=self.embed_mock)
        queries = ["A", "B", "C"]
        result = similarity.filter_redundant(queries)
        # B accepted, C redundant (similar to B). A distinct, kept.
        self.assertEqual(result, ["A", "B"])

    # 11. Embedding reuse – ensure embed_fn called only once per unique query
    def test_embedding_reuse(self):
        embed_counter = Mock(side_effect=lambda q: [1.0, 0.0, 0.0])
        sim = SemanticSimilarity(threshold=0.9, embed_fn=embed_counter)
        queries = ["q1", "q2", "q1"]
        sim.filter_redundant(queries)
        # Expected calls: q1 (first), q2, then q1 again should be cached -> no extra call
        self.assertEqual(embed_counter.call_count, 2)

    # 12. Invalid embedding dimensions raise error
    def test_invalid_embedding_dimension(self):
        bad_embed = Mock(return_value=[1.0, 2.0])  # length 2 vs other length 3
        sim = SemanticSimilarity(threshold=0.5, embed_fn=bad_embed)
        with self.assertRaises(ValueError):
            sim.similarity("identical", "different")

    # 13. Zero norm vector handling in static cosine_similarity
    def test_static_cosine_zero_norm(self):
        with self.assertRaises(ValueError):
            SemanticSimilarity.cosine_similarity([0, 0, 0], [1, 0, 0])

    # 14. Static cosine similarity correct calculation
    def test_static_cosine_correct(self):
        sim = SemanticSimilarity.cosine_similarity([1, 0, 0], [0, 1, 0])
        self.assertAlmostEqual(sim, 0.0, places=5)
        sim = SemanticSimilarity.cosine_similarity([1, 0, 0], [1, 0, 0])
        self.assertAlmostEqual(sim, 1.0, places=5)
        sim = SemanticSimilarity.cosine_similarity([1, 0, 0], [-1, 0, 0])
        self.assertAlmostEqual(sim, -1.0, places=5)


class TestSemanticFilteringCategories(unittest.TestCase):
    """Dedicated regression tests verifying the 7 required semantic filtering categories
    using the real production BGE embedding encoder.
    """
    @classmethod
    def setUpClass(cls):
        # Initialize production filter once to avoid repeated model reloads
        cls.filter = SemanticSimilarity()

    # Test 1 — Exact duplicate
    def test_category_1_exact_duplicates(self):
        queries = ["semester fees", "semester fees"]
        filtered = self.filter.filter_redundant(queries)
        self.assertEqual(filtered, ["semester fees"])

        # Also test casing and whitespace variations
        queries_ws = ["semester fees", "  Semester Fees  ", "semester fees"]
        filtered_ws = self.filter.filter_redundant(queries_ws)
        self.assertEqual(filtered_ws, ["Semester Fees"])

    # Test 2 — Near-identical paraphrases
    def test_category_2_near_identical_paraphrases(self):
        queries = [
            "semester withdrawal refund policy",
            "refund policy for semester withdrawal",
        ]
        filtered = self.filter.filter_redundant(queries)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0], "semester withdrawal refund policy")

    # Test 3 — Semantic similarity but lexical novelty
    def test_category_3_semantic_similarity_lexical_novelty(self):
        queries = [
            "attendance shortage below 75%",
            "attendance shortage rules",
        ]
        filtered = self.filter.filter_redundant(queries)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered, queries)

    # Test 4 — Refund reformulations
    def test_category_4_refund_reformulations(self):
        queries = [
            "Can I get a refund if I withdraw from the semester?",
            "semester withdrawal refund policy",
            "refund eligibility after semester withdrawal",
            "tuition refund for semester withdrawal",
            "semester withdrawal fee refund",
        ]
        filtered = self.filter.filter_redundant(queries)
        # Original query must be preserved unconditionally at index 0
        self.assertEqual(filtered[0], queries[0])
        # Diverse reformulations must survive (not collapsed to only original)
        self.assertGreaterEqual(len(filtered), 4)
        for expected_topic in ["tuition refund for semester withdrawal", "semester withdrawal refund policy"]:
            self.assertIn(expected_topic, filtered)

    # Test 5 — Unrelated queries
    def test_category_5_unrelated_queries(self):
        queries = [
            "semester withdrawal refund",
            "hostel allotment process",
            "CSE301 prerequisites",
        ]
        filtered = self.filter.filter_redundant(queries)
        self.assertEqual(filtered, queries)

    # Test 6 — Same intent, different terminology
    def test_category_6_same_intent_different_terminology(self):
        pairs = [
            ("attendance shortage", "attendance deficiency"),
            ("course registration", "course enrollment"),
            ("semester withdrawal", "semester cancellation"),
            ("academic transcript", "grade report"),
        ]
        for q1, q2 in pairs:
            filtered = self.filter.filter_redundant([q1, q2])
            self.assertEqual(len(filtered), 2, f"Failed to preserve terminology variant pair: {(q1, q2)}")
            self.assertEqual(filtered, [q1, q2])

    # Test 7 — Order independence for diverse candidate sets
    def test_category_7_order_independence(self):
        p1 = [
            "attendance shortage below 75%",
            "attendance shortage rules",
            "attendance condonation process",
        ]
        p2 = [
            "attendance shortage rules",
            "attendance condonation process",
            "attendance shortage below 75%",
        ]
        p3 = [
            "attendance condonation process",
            "attendance shortage below 75%",
            "attendance shortage rules",
        ]
        res1 = self.filter.filter_redundant(p1)
        res2 = self.filter.filter_redundant(p2)
        res3 = self.filter.filter_redundant(p3)
        self.assertEqual(set(res1), set(res2))
        self.assertEqual(set(res2), set(res3))

    # Additional robustness and idempotence tests
    def test_idempotence(self):
        queries = [
            "Can I get a refund if I withdraw from the semester?",
            "semester withdrawal refund policy",
            "refund policy for semester withdrawal",
            "tuition refund for semester withdrawal",
        ]
        first_pass = self.filter.filter_redundant(queries)
        second_pass = self.filter.filter_redundant(first_pass)
        self.assertEqual(first_pass, second_pass)

    def test_whitespace_and_invalid_elements(self):
        self.assertEqual(self.filter.filter_redundant(["   ", ""]), [])
        self.assertEqual(self.filter.filter_redundant(["fees", None, 123]), ["fees"])


class TestGlobalOrderIndependence(unittest.TestCase):
    """Exhaustive adversarial permutation tests verifying global order independence."""
    @classmethod
    def setUpClass(cls):
        cls.filter = SemanticSimilarity()

    # Test 1 — Refund paraphrase permutations (all 6 permutations)
    def test_adversarial_1_refund_paraphrases(self):
        from itertools import permutations
        queries = [
            "semester withdrawal refund policy",
            "refund policy for semester withdrawal",
            "tuition refund after semester withdrawal"
        ]
        results = set()
        for p in permutations(queries):
            res = self.filter.filter_redundant(list(p))
            results.add(frozenset(res))
        self.assertEqual(len(results), 1, f"Expected 1 unique selected set, got {len(results)}: {results}")

    # Test 2 — Attendance paraphrases (all 6 permutations)
    def test_adversarial_2_attendance_paraphrases(self):
        from itertools import permutations
        queries = [
            "attendance shortage below 75%",
            "attendance shortage under 75%",
            "attendance condonation procedure"
        ]
        results = set()
        for p in permutations(queries):
            res = self.filter.filter_redundant(list(p))
            results.add(frozenset(res))
        self.assertEqual(len(results), 1, f"Expected 1 unique selected set, got {len(results)}: {results}")

    # Test 3 — Course registration paraphrases (all 6 permutations)
    def test_adversarial_3_course_registration_paraphrases(self):
        from itertools import permutations
        queries = [
            "course registration process",
            "process for course registration",
            "course enrollment procedure"
        ]
        results = set()
        for p in permutations(queries):
            res = self.filter.filter_redundant(list(p))
            results.add(frozenset(res))
        self.assertEqual(len(results), 1, f"Expected 1 unique selected set, got {len(results)}: {results}")

    # Test 4 — General vs specific ([A, B] and [B, A])
    def test_adversarial_4_general_vs_specific(self):
        a = "semester fees"
        b = "semester tuition and fees"
        res_ab = self.filter.filter_redundant([a, b])
        res_ba = self.filter.filter_redundant([b, a])
        self.assertEqual(set(res_ab), set(res_ba))
        self.assertEqual(res_ab, ["semester tuition and fees"])

    # Test 5 — Exact duplicates permutations
    def test_adversarial_5_exact_duplicates(self):
        from itertools import permutations
        dups = [
            "semester fees",
            "semester fees",
            "Semester fees",
            " semester fees "
        ]
        results = set()
        for p in permutations(dups):
            res = self.filter.filter_redundant(list(p))
            results.add(frozenset(res))
        self.assertEqual(len(results), 1, f"Expected 1 unique selected set, got {len(results)}: {results}")

    # Test 6 — Useful lexical diversity (all 6 permutations)
    def test_adversarial_6_useful_lexical_diversity(self):
        from itertools import permutations
        queries = [
            "attendance shortage below 75%",
            "attendance shortage rules",
            "attendance condonation procedure"
        ]
        results = set()
        for p in permutations(queries):
            res = self.filter.filter_redundant(list(p))
            results.add(frozenset(res))
            self.assertEqual(len(res), 3)
        self.assertEqual(len(results), 1)

    # Test 7 — Five refund reformulations (all 120 permutations)
    def test_adversarial_7_five_refund_reformulations(self):
        from itertools import permutations
        queries = [
            "Can I get a refund if I withdraw from the semester?",
            "semester withdrawal refund policy",
            "tuition refund after semester withdrawal",
            "refund eligibility for withdrawing from semester",
            "semester withdrawal fee refund"
        ]
        results = set()
        for p in permutations(queries):
            res = self.filter.filter_redundant(list(p), original_query=queries[0])
            results.add(frozenset(res))
        self.assertEqual(len(results), 1, f"Expected 1 unique selected set across 120 permutations, got {len(results)}")


class TestSemanticFilterDiagnostics(unittest.TestCase):
    def setUp(self):
        self.fake_embeddings = {
            "query orig": [1.0, 0.0, 0.0],
            "query distinct": [0.0, 1.0, 0.0],
            "query redundant": [0.99, 0.01, 0.0],
            "query overlap": [0.92, 0.08, 0.0],
            "query low sim": [0.1, 0.9, 0.0],
        }
        self.embed_mock = Mock(side_effect=lambda q: self.fake_embeddings.get(q.strip().lower(), [0.5, 0.5, 0.0]))
        self.sim = SemanticSimilarity(threshold=0.85, embed_fn=self.embed_mock)

    def test_filter_with_diagnostics_structure(self):
        candidates = ["query orig", "query distinct", "query redundant"]
        retained, diagnostics = self.sim.filter_redundant_with_diagnostics(
            candidates, original_query="query orig"
        )
        self.assertEqual(len(retained), 2)
        self.assertEqual(len(diagnostics), 3)

        # Check diagnostics types
        for d in diagnostics:
            self.assertIsInstance(d, FilterDecision)
            self.assertIn("query", d.to_dict())
            self.assertIn("similarity", d.to_dict())
            self.assertIn("kept", d.to_dict())
            self.assertIn("reason", d.to_dict())
            self.assertIn("compared_with", d.to_dict())

        # Original query diagnostic
        d0 = diagnostics[0]
        self.assertEqual(d0.query, "query orig")
        self.assertTrue(d0.kept)
        self.assertEqual(d0.reason, "Original normalized query")
        self.assertEqual(d0.similarity, 1.0)
        self.assertIsNone(d0.compared_with)

        # Distinct query diagnostic
        d1 = diagnostics[1]
        self.assertEqual(d1.query, "query distinct")
        self.assertTrue(d1.kept)
        self.assertIn("diversity", d1.reason.lower())

        # Redundant query diagnostic
        d2 = diagnostics[2]
        self.assertEqual(d2.query, "query redundant")
        self.assertFalse(d2.kept)
        self.assertIn("redundancy", d2.reason.lower())
        self.assertEqual(d2.compared_with, "query orig")
        self.assertGreater(d2.similarity, 0.95)

    def test_duplicate_input_diagnostics(self):
        candidates = ["query orig", "query orig", "query distinct"]
        retained, diagnostics = self.sim.filter_redundant_with_diagnostics(
            candidates, original_query="query orig"
        )
        self.assertEqual(len(retained), 2)
        self.assertEqual(len(diagnostics), 3)

        # Duplicate item should have kept=False, reason containing 'duplicate'
        d1 = diagnostics[1]
        self.assertEqual(d1.query, "query orig")
        self.assertFalse(d1.kept)
        self.assertIn("duplicate", d1.reason.lower())
        self.assertEqual(d1.compared_with, "query orig")

    def test_empty_and_single_query_diagnostics(self):
        retained_empty, diag_empty = self.sim.filter_redundant_with_diagnostics([])
        self.assertEqual(retained_empty, [])
        self.assertEqual(diag_empty, [])

        retained_single, diag_single = self.sim.filter_redundant_with_diagnostics(
            ["query orig"], original_query="query orig"
        )
        self.assertEqual(retained_single, ["query orig"])
        self.assertEqual(len(diag_single), 1)
        self.assertTrue(diag_single[0].kept)
        self.assertEqual(diag_single[0].similarity, 1.0)


if __name__ == "__main__":
    unittest.main()



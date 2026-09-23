import unittest
from unittest.mock import Mock
from reformulation.semantic_similarity import SemanticSimilarity

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

    # 9. First occurrence wins
    def test_first_occurrence_wins(self):
        queries = ["similar_b", "similar_a", "different"]
        result = self.similarity.filter_redundant(queries)
        # Now similar_a is redundant because similar_b came first (they are similar)
        self.assertEqual(result, ["similar_b", "different"])

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

if __name__ == "__main__":
    unittest.main()

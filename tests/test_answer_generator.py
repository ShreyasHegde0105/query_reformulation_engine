import json
import unittest
from unittest.mock import MagicMock

from reformulation.answer_generator import AnswerGenerator
from reformulation.exceptions import ReformulationParseError
from reformulation.llm.client import LLMClient
from retrieval.models import RetrievedChunk


def _chunk(cid, text="Revaluation costs 500 rupees per paper within 7 days."):
    return RetrievedChunk(chunk_id=cid, document_id=cid.split("_")[0], title="Revaluation Guidelines",
                          domain="Exams", chunk_index=0, chunk_text=text, section="3. FEES", token_count=10, score=0.1)


class TestAnswerGenerator(unittest.TestCase):
    def setUp(self):
        self.llm = MagicMock(spec=LLMClient)
        self.gen = AnswerGenerator(self.llm, max_chunks=2)
        self.chunks = [_chunk("EXAM_CHUNK_001"), _chunk("EXAM_CHUNK_002"), _chunk("EXAM_CHUNK_003")]

    def _reply(self, **payload):
        self.llm.generate.return_value = json.dumps(payload)

    def test_grounded_answer_with_citations(self):
        self._reply(answerable=True, answer="500 rupees per paper, within 7 days.", citations=["EXAM_CHUNK_001"])
        ans = self.gen.generate("revaluation fee?", self.chunks)
        self.assertTrue(ans.answerable)
        self.assertEqual(ans.citations, ["EXAM_CHUNK_001"])
        self.assertEqual(ans.text, "500 rupees per paper, within 7 days.")

    def test_prompt_contains_only_top_chunks(self):
        self._reply(answerable=True, answer="x", citations=["EXAM_CHUNK_001"])
        self.gen.generate("revaluation fee?", self.chunks)
        prompt = self.llm.generate.call_args.args[0]
        self.assertIn("[EXAM_CHUNK_001] Revaluation Guidelines — 3. FEES", prompt)
        self.assertIn("[EXAM_CHUNK_002]", prompt)
        self.assertNotIn("EXAM_CHUNK_003", prompt)
        self.assertIn("Question: revaluation fee?", prompt)

    def test_citations_filtered_to_provided_chunks_and_deduplicated(self):
        self._reply(answerable=True, answer="x",
                    citations=["EXAM_CHUNK_002", "EXAM_CHUNK_003", "MADE_UP", "EXAM_CHUNK_002", 7])
        self.assertEqual(self.gen.generate("q", self.chunks).citations, ["EXAM_CHUNK_002"])

    def test_answer_without_valid_citation_is_not_grounded(self):
        self._reply(answerable=True, answer="Probably 1000 rupees.", citations=["MADE_UP"])
        ans = self.gen.generate("q", self.chunks)
        self.assertFalse(ans.answerable)
        self.assertEqual(ans.citations, [])
        self.assertNotIn("1000", ans.text)

    def test_unanswerable(self):
        self._reply(answerable=False, answer="The sources do not mention hostel wifi.", citations=["EXAM_CHUNK_001"])
        ans = self.gen.generate("hostel wifi password?", self.chunks)
        self.assertFalse(ans.answerable)
        self.assertEqual(ans.citations, [])
        self.assertEqual(ans.text, "The sources do not mention hostel wifi.")

    def test_no_chunks_skips_llm(self):
        ans = self.gen.generate("q", [])
        self.assertFalse(ans.answerable)
        self.llm.generate.assert_not_called()

    def test_markdown_wrapped_json(self):
        self.llm.generate.return_value = '```json\n{"answerable": true, "answer": "x", "citations": ["EXAM_CHUNK_001"]}\n```'
        self.assertTrue(self.gen.generate("q", self.chunks).answerable)

    def test_malformed_responses_raise(self):
        for raw in ("not json", "[1, 2]", json.dumps({"answer": "x"}),
                    json.dumps({"answerable": "yes", "answer": "x", "citations": []})):
            self.llm.generate.return_value = raw
            with self.assertRaises(ReformulationParseError, msg=raw):
                self.gen.generate("q", self.chunks)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            self.gen.generate("  ", self.chunks)
        with self.assertRaises(TypeError):
            AnswerGenerator(object())
        with self.assertRaises(ValueError):
            AnswerGenerator(self.llm, max_chunks=0)


if __name__ == "__main__":
    unittest.main()

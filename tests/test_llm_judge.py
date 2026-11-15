"""Tests for the evaluation-only LLM relevance judge (no network: the LLM is scripted)."""

import copy
import json
import unittest

from evaluation.judge import (
    PROMPT_VERSION, RUBRIC_VERSION, JudgeConfig, JudgmentValidationError, LLMRelevanceJudge,
    answerability_from, validate_judge_payload,
)
from evaluation.judge.llm_judge import JudgePassage, PassageJudgment
from retrieval.models import RetrievedChunk


def chunk(cid, text, title="Policy", section="1. RULES"):
    return RetrievedChunk(chunk_id=cid, document_id=cid.split("_")[0], title=title, domain="D", chunk_index=0,
                          chunk_text=text, section=section, token_count=10, score=0.5)


class ScriptedLLM:
    """Answers each prompt by calling ``responder(prompt)``; records prompts."""

    def __init__(self, responder):
        self.responder = responder
        self.prompts = []

        class C:
            provider, model = "fake", "fake-judge-1"
        self.config = C()

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.responder(prompt)


def passage_ids(prompt):
    return [line[1:-1] for line in prompt.splitlines() if line.startswith("[P") and line.endswith("]")]


def grade_all(grade, intents=("curfew time",), evidence_from=None):
    def respond(prompt):
        js = []
        for pid in passage_ids(prompt):
            if grade == 0:
                js.append({"passage_id": pid, "grade": 0, "intents_addressed": [], "evidence": [], "rationale": "no"})
            else:
                body = prompt.split(f"[{pid}]\n", 1)[1].split("\n\n[P", 1)[0].split("\n\nReturn ONLY", 1)[0]
                span = evidence_from or body.splitlines()[-1][:30]
                js.append({"passage_id": pid, "grade": grade, "intents_addressed": [0], "evidence": [span],
                           "rationale": "yes"})
        return json.dumps({"query_intents": list(intents), "judgments": js})
    return respond


class TestSchema(unittest.TestCase):
    P = {"P1": "Curfew is 9:30 PM on weekdays.", "P2": "Library fines are Rs 5 per day."}

    def ok(self, **over):
        p = {"query_intents": ["curfew time"], "judgments": [
            {"passage_id": "P1", "grade": 3, "intents_addressed": [0], "evidence": ["Curfew is 9:30 PM"], "rationale": "r"},
            {"passage_id": "P2", "grade": 0, "intents_addressed": [], "evidence": [], "rationale": "r"}]}
        p.update(over)
        return p

    def test_valid_payload_passes(self):
        out = validate_judge_payload(self.ok(), self.P)
        self.assertEqual([j["grade"] for j in out["judgments"]], [3, 0])

    def test_evidence_must_be_quoted_from_passage(self):
        bad = self.ok()
        bad["judgments"][0]["evidence"] = ["Curfew is 11 PM"]
        with self.assertRaises(JudgmentValidationError):
            validate_judge_payload(bad, self.P)

    def test_evidence_match_ignores_whitespace_and_case(self):
        p = self.ok()
        p["judgments"][0]["evidence"] = ["curfew   IS 9:30 pm"]
        validate_judge_payload(p, self.P)

    def test_positive_grade_requires_evidence(self):
        bad = self.ok()
        bad["judgments"][0]["evidence"] = []
        with self.assertRaises(JudgmentValidationError):
            validate_judge_payload(bad, self.P)

    def test_grade_zero_must_not_carry_evidence(self):
        bad = self.ok()
        bad["judgments"][1]["evidence"] = ["Library fines"]
        with self.assertRaises(JudgmentValidationError):
            validate_judge_payload(bad, self.P)

    def test_grade_out_of_range_rejected(self):
        for g in (4, -1, 2.5, True, "3"):
            bad = self.ok()
            bad["judgments"][0]["grade"] = g
            with self.assertRaises(JudgmentValidationError, msg=str(g)):
                validate_judge_payload(bad, self.P)

    def test_every_passage_judged_exactly_once(self):
        bad = self.ok()
        bad["judgments"] = bad["judgments"][:1]
        with self.assertRaises(JudgmentValidationError):
            validate_judge_payload(bad, self.P)
        dup = self.ok()
        dup["judgments"].append(dict(dup["judgments"][1]))
        with self.assertRaises(JudgmentValidationError):
            validate_judge_payload(dup, self.P)

    def test_unknown_keys_rejected(self):
        bad = self.ok(confidence=0.9)
        with self.assertRaises(JudgmentValidationError):
            validate_judge_payload(bad, self.P)

    def test_intent_index_out_of_range_rejected(self):
        bad = self.ok()
        bad["judgments"][0]["intents_addressed"] = [3]
        with self.assertRaises(JudgmentValidationError):
            validate_judge_payload(bad, self.P)

    def test_fixed_intents_must_be_repeated(self):
        with self.assertRaises(JudgmentValidationError):
            validate_judge_payload(self.ok(), self.P, fixed_intents=["something else"])


class TestAnswerability(unittest.TestCase):
    def j(self, grade, intents):
        return PassageJudgment("c", grade, intents, ["x"] if grade else [], "")

    def test_levels(self):
        intents = ["refund", "fine"]
        self.assertEqual(answerability_from(intents, [self.j(1, [0]), self.j(0, [])]), "unanswerable")
        self.assertEqual(answerability_from(intents, [self.j(3, [0])]), "partially_answerable")
        self.assertEqual(answerability_from(intents, [self.j(3, [0]), self.j(2, [1])]), "fully_answerable")


class TestJudge(unittest.TestCase):
    def setUp(self):
        self.chunks = [chunk("A_CHUNK_001", "Curfew is 9:30 PM on weekdays."),
                       chunk("B_CHUNK_001", "Library fines are Rs 5 per day."),
                       chunk("A_CHUNK_001", "Curfew is 9:30 PM on weekdays.")]  # duplicate

    def test_records_provenance_and_dedups(self):
        llm = ScriptedLLM(grade_all(2))
        res = LLMRelevanceJudge(llm).judge("Q1", "what is the curfew", self.chunks)
        self.assertEqual(res.status, "ok")
        self.assertEqual(sorted(res.grades()), ["A_CHUNK_001", "B_CHUNK_001"])
        self.assertEqual((res.rubric_version, res.prompt_version), (RUBRIC_VERSION, PROMPT_VERSION))
        self.assertEqual((res.provider, res.model, res.query_id), ("fake", "fake-judge-1", "Q1"))
        self.assertTrue(res.timestamp)
        self.assertEqual(res.answerability, "fully_answerable")

    def test_blind_prompt_reveals_no_ids_systems_ranks_or_scores(self):
        llm = ScriptedLLM(grade_all(0))
        LLMRelevanceJudge(llm).judge("Q1", "what is the curfew", self.chunks)
        prompt = llm.prompts[0]
        for leak in ("A_CHUNK_001", "B_CHUNK_001", "baseline", "reformulat", "rerank", "score", "rank 1"):
            self.assertNotIn(leak, prompt)

    def test_blind_order_independent_of_input_order(self):
        a = LLMRelevanceJudge.blind_order("Q1", [JudgePassage.from_any(c) for c in self.chunks])
        b = LLMRelevanceJudge.blind_order("Q1", [JudgePassage.from_any(c) for c in reversed(self.chunks)])
        self.assertEqual([p.chunk_id for p in a], [p.chunk_id for p in b])

    def test_judge_does_not_modify_inputs(self):
        before = copy.deepcopy(self.chunks)
        LLMRelevanceJudge(ScriptedLLM(grade_all(3))).judge("Q1", "curfew", self.chunks)
        self.assertEqual(self.chunks, before)

    def test_invalid_output_is_rejected_not_repaired(self):
        llm = ScriptedLLM(lambda p: '{"query_intents": ["x"], "judgments": []}')
        res = LLMRelevanceJudge(llm, JudgeConfig(max_attempts=2)).judge("Q1", "curfew", self.chunks)
        self.assertEqual(res.status, "rejected")
        self.assertEqual(res.judgments, [])
        self.assertIsNone(res.answerability)
        self.assertEqual(len(llm.prompts), 2)  # one retry, then give up

    def test_hallucinated_evidence_rejected(self):
        llm = ScriptedLLM(grade_all(3, evidence_from="Curfew is midnight"))
        res = LLMRelevanceJudge(llm).judge("Q1", "curfew", self.chunks)
        self.assertEqual(res.status, "rejected")

    def test_transport_error_is_reported(self):
        def boom(_):
            raise RuntimeError("quota")
        res = LLMRelevanceJudge(ScriptedLLM(boom)).judge("Q1", "curfew", self.chunks)
        self.assertEqual(res.status, "rejected")
        self.assertIn("quota", res.error)

    def test_batches_share_fixed_intents(self):
        many = [chunk(f"D{i}_CHUNK_001", f"Text number {i} about curfew rules.") for i in range(5)]
        llm = ScriptedLLM(grade_all(1))
        res = LLMRelevanceJudge(llm, JudgeConfig(batch_size=2)).judge("Q1", "curfew", many)
        self.assertEqual(res.status, "ok")
        self.assertEqual(len(llm.prompts), 3)
        self.assertNotIn("already been identified", llm.prompts[0])
        self.assertIn("already been identified", llm.prompts[1])
        self.assertEqual(res.answerability, "unanswerable")  # only grade-1 passages

    def test_empty_passages(self):
        res = LLMRelevanceJudge(ScriptedLLM(grade_all(0))).judge("Q1", "curfew", [])
        self.assertEqual((res.status, res.judgments), ("ok", []))


if __name__ == "__main__":
    unittest.main()

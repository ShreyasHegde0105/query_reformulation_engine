"""Architecture rules: the evaluation judge and the production reranker stay independent.

* Production code (api/, reformulation/, retrieval/, embedding/) never imports the judge
  or anything from evaluation/.
* The judge never imports the reranker, the search pipeline or retrieval code.
* The reranker never imports evaluation code.
* Neither component consumes the other's scores; the reranker exposes no relevance labels,
  probabilities or thresholds.

Static checks parse each module's own import statements (package ``__init__`` side effects
of the evaluation package are not counted: they belong to the evaluation runner, not the judge).
"""

import ast
import copy
import dataclasses
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIRS = ["api", "reformulation", "retrieval", "embedding"]
JUDGE_DIR = ROOT / "evaluation" / "judge"
RERANKER_FILE = ROOT / "reformulation" / "production_reranker.py"


def imported_modules(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module if node.level == 0 else f"(relative){node.module}")
    return mods


class TestStaticImports(unittest.TestCase):
    def test_production_code_never_imports_evaluation_or_judge(self):
        offenders = []
        for d in PRODUCTION_DIRS:
            for f in (ROOT / d).rglob("*.py"):
                for m in imported_modules(f):
                    if m == "evaluation" or m.startswith("evaluation.") or "judge" in m:
                        offenders.append(f"{f.relative_to(ROOT)} imports {m}")
        self.assertEqual(offenders, [])

    def test_api_does_not_reference_the_judge(self):
        src = (ROOT / "api" / "app.py").read_text(encoding="utf-8")
        for name in ("LLMRelevanceJudge", "evaluation.judge", "JudgeConfig"):
            self.assertNotIn(name, src)

    def test_judge_does_not_import_ranking_or_retrieval(self):
        forbidden = ("reformulation.production_reranker", "reformulation.cross_encoder",
                     "reformulation.search_pipeline", "reformulation.reformulator", "retrieval", "api")
        offenders = []
        for f in JUDGE_DIR.glob("*.py"):
            for m in imported_modules(f):
                if any(m == x or m.startswith(x + ".") for x in forbidden):
                    offenders.append(f"{f.name} imports {m}")
        self.assertEqual(offenders, [])

    def test_reranker_does_not_import_evaluation(self):
        mods = imported_modules(RERANKER_FILE)
        self.assertFalse([m for m in mods if m.startswith("evaluation") or "judge" in m])


class TestRuntimeSeparation(unittest.TestCase):
    def test_reranker_has_no_label_probability_or_threshold_surface(self):
        from reformulation.production_reranker import ProductionReranker, RerankerConfig, RerankOutcome
        names = {f.name for f in dataclasses.fields(RerankerConfig)} | {f.name for f in dataclasses.fields(RerankOutcome)}
        names |= {n for n in dir(ProductionReranker) if not n.startswith("__")}
        for bad in ("threshold", "probability", "prob", "relevance", "grade", "label", "judge", "calibrat"):
            self.assertFalse([n for n in names if bad in n.lower()], bad)

    def test_judge_api_does_not_accept_reranker_scores(self):
        import inspect
        from evaluation.judge import LLMRelevanceJudge
        params = set(inspect.signature(LLMRelevanceJudge.judge).parameters)
        self.assertEqual(params, {"self", "query_id", "query", "passages"})

    def test_judge_prompt_never_contains_reranker_or_fusion_scores(self):
        from evaluation.judge import LLMRelevanceJudge
        from retrieval.models import RetrievedChunk

        seen = []

        class LLM:
            class config:
                provider, model = "fake", "fake"

            def generate(self, prompt):
                seen.append(prompt)
                return '{"query_intents": ["x"], "judgments": []}'

        c = RetrievedChunk(chunk_id="Z_CHUNK_001", document_id="Z", title="T", domain="D", chunk_index=0,
                           chunk_text="body text here", section=None, token_count=3, score=0.0123456,
                           metadata={"reranker_logit": -7.654321})
        before = copy.deepcopy(c)
        LLMRelevanceJudge(LLM()).judge("Q", "query", [c])
        self.assertTrue(seen)
        for prompt in seen:
            self.assertNotIn("7.654", prompt)
            self.assertNotIn("0.0123", prompt)
            self.assertNotIn("Z_CHUNK_001", prompt)
        self.assertEqual(c, before)


if __name__ == "__main__":
    unittest.main()

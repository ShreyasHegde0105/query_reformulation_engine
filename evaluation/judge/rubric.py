"""Version-controlled relevance rubric and prompt for the evaluation-only LLM judge.

Changing the rubric or the prompt text MUST bump the matching version string, so
judgments made under different instructions are never mixed. Both versions are
recorded with every judgment.
"""

from __future__ import annotations

RUBRIC_VERSION = "relevance-rubric-v1"
PROMPT_VERSION = "judge-prompt-v1"

GRADE_SCALE = {
    0: "Irrelevant: does not address any intent of the query.",
    1: "Marginally relevant: on the topic or background, but does not give the information asked for.",
    2: "Relevant: gives part of the information asked for (one intent, or part of one).",
    3: "Highly relevant: directly and substantially answers at least one intent of the query.",
}

RUBRIC_TEXT = """\
RELEVANCE RUBRIC (relevance-rubric-v1)

You grade how relevant each passage is to the user's query. Grade each passage on its own;
do not compare passages with each other and do not reward a passage for its position.

Grades:
  0 = Irrelevant: does not address any intent of the query.
  1 = Marginally relevant: on the same topic or useful background, but it does not contain the
      information the user asked for.
  2 = Relevant: contains part of the requested information (for example, one intent of a
      multi-part query, or an incomplete answer to one intent).
  3 = Highly relevant: directly and substantially answers at least one intent of the query.

Rules:
  - Judge only from the passage text shown. Do not use outside knowledge about the organisation.
  - A query may contain several intents (for example "hostel fee refund and library fine payment"
    has two). List them, and for each passage say which intents (by index) it addresses.
  - For grades 1, 2 and 3 you MUST quote evidence: one or more short spans copied EXACTLY,
    character for character, from the passage. For grade 0 give no evidence.
  - A passage that merely repeats words from the query without giving the requested information
    is at most grade 1.
  - If the query is misspelled, judge against the evident intended meaning.
"""

PROMPT_TEMPLATE = """\
You are an evaluation judge for an enterprise document search system. You only grade relevance;
you never rewrite queries or rank results.

{rubric}

Query: "{query}"
{intents_block}
Passages (identifiers are random; their order carries no meaning):
{passages}

Return ONLY a JSON object with exactly this structure:
{{
  "query_intents": ["intent 1", "intent 2"],
  "judgments": [
    {{
      "passage_id": "P1",
      "grade": 0,
      "intents_addressed": [],
      "evidence": [],
      "rationale": "one short sentence"
    }}
  ]
}}
Requirements: one judgment per passage id listed above, no others; "grade" is an integer 0-3;
"intents_addressed" holds 0-based indexes into "query_intents"; every "evidence" string is an
exact substring of that passage; grade 0 has empty "evidence" and "intents_addressed".
"""

FIXED_INTENTS_BLOCK = """\
The query's intents have already been identified; use exactly these, in this order, as "query_intents":
{intents}
"""

# AI-Powered Enterprise Query Reformulation & Search Engine

A search engine for an organisation's internal documents (here: the policies, procedures and forms of a
fictional university, "Demo University"). Before searching, it **cleans the user's query and asks an LLM to
rewrite it in several ways**, searches with every version, and merges the results. The project then
measures, on queries it was never tuned on, whether this actually finds better documents than plain search.

**Short answer: it does.** On 94 held-out test queries, on average 94.4% of the relevant chunks appear in
the top 10 results instead of 87.5%, and the gain is statistically significant.

---

## Contents

1. [Results at a glance](#1-results-at-a-glance)
2. [How it works](#2-how-it-works)
3. [Each stage explained, with formulas and worked examples](#3-each-stage-explained)
4. [How it is evaluated](#4-how-it-is-evaluated)
5. [Results in detail](#5-results-in-detail)
6. [Demo: one query, end to end](#6-demo-one-query-end-to-end)
7. [Ideas that were tested and rejected](#7-ideas-that-were-tested-and-rejected)
8. [Setup and usage](#8-setup-and-usage)
9. [Project structure](#9-project-structure)
10. [Known limitations](#10-known-limitations)

---

## 1. Results at a glance

94 held-out test queries, written without looking at the documents and never used for tuning
(`evaluation/heldout_queries.json`, full report in `evaluation/results/heldout_report.md`).
Every row uses the same indexes; **only the handling of the query changes**.

| System | Recall@10 | MRR | NDCG@10 |
|:---|---:|---:|---:|
| **Before:** hybrid search on the raw query | 0.875 | 0.691 | 0.691 |
| + normalization and spelling correction | 0.893 | 0.678 | 0.696 |
| **After:** + LLM reformulation (full pipeline) | **0.944** | **0.727** | **0.739** |

- The full pipeline improves **Recall@10 by +0.068** and **NDCG@10 by +0.048** over plain hybrid search.
  Both are statistically significant after correcting for 9 tests (Holm, p < 0.05).
- The **MRR** gain (+0.036) is **not** significant.
- Normalization and spelling on their own are **not** significant; the LLM rewrites do most of the work.
- The biggest gain is on queries worded differently from the documents
  ("lexical diversity": NDCG@10 **0.43 → 0.62**).

---

## 2. How it works

```
                       OFFLINE (run once)
 ┌──────────────┐   ┌──────────────┐   ┌─────────────┐   ┌───────────────────────────┐
 │ 158 JSON     │ → │ Validation   │ → │ Chunking    │ → │ Embedding (bge-small-en)  │
 │ documents    │   │ 11 checks    │   │ 702 chunks  │   │ → Supabase pgvector       │
 └──────────────┘   └──────────────┘   └─────────────┘   └───────────────────────────┘

                       ONLINE (every search)
 user query
    │
    ▼
 1. Validate         non-empty string, at most 256 characters
 2. Normalize        fix typos, expand abbreviations, keep codes and acronyms
 3. Reformulate      LLM writes up to 5 alternative search queries
 4. Filter           drop rewrites that are near-duplicates of each other
 5. Hybrid search    for EACH query: vector search + BM25 keyword search, merged with RRF
 6. Multi-query RRF  merge the result lists of all queries into one ranking
 7. (optional)       LLM answer grounded in the top 5 chunks, with checked citations
    │
    ▼
 ranked chunks (+ answer)
```

If the LLM is unavailable (no key, quota used up, timeout), step 3 is skipped and the search still runs
on the original query. The original query is always searched, and always counts first.

---

## 3. Each stage explained

### 3.1 Corpus validation (`corpus_validation/`)

**What:** before anything is indexed, every document goes through 11 checks: schema, ID format and
uniqueness, content length and quality, exact and near duplicates, metadata, entity consistency (one
department ID = one department name), internal consistency, search quality, coverage, and a check
against the university's "world model" (`Data/demo_university_world_model.json`).

**Why:** a broken document breaks search silently. A duplicate ID, for example, makes two documents
claim the same chunk IDs.

**Actual output** (`python validate_corpus.py -i Data/documents`):

```
[INFO] Loaded 158 documents from Data/documents (0 load issues)
[INFO] Running Stage 1: Schema Validation
   ...
[INFO] Running Stage 11: Ground Truth Reconciliation
[INFO] Validation complete: 158 valid, 0 rejected, 0 errors, 0 warnings, 187 infos
[INFO] Corpus successfully validated. Downstream chunking pipeline can proceed.
```

The first run on the raw corpus failed with 1 error and 6 warnings. These were fixed in the data:

| Problem found | Fix |
|:---|:---|
| Two different documents used the ID `POL-MEDICAL-LEAVE` (both were rejected, so the medical leave policy was unsearchable) | Kept the full v2.2 policy, removed the short v3.0 copy |
| Placeholder `+91-XXXX-EMERG` in `SEC-SOP-001` | Filled in a phone number |
| Example code `GRV-2026-XXXXX` in `STU-GUI-001` flagged as a placeholder | Replaced with a concrete example `GRV-2026-00417` |
| `DEPT-ACAD`, `DEPT-ADM`, `DEPT-EXAM` each had several department names | One official name per ID; governance/HR documents moved to a new `DEPT-REG` |
| `DEPT-IRO` missing from the world model | Added it |

### 3.2 Document chunking (`document_chunking/`)

**What:** each document is split into chunks, following its own structure: first by numbered section
(`1. PURPOSE`, `2. SCOPE`, ...), then by paragraph, then by sentence, and only as a last resort by token
count. Settings (`config/chunking_config.json`): target 500 tokens, maximum 600, minimum 100, 75 tokens of
overlap when a section has to be split. Overlap never crosses a section boundary.

**Why:** an embedding of a whole 3-page policy blurs specific facts (a 75% threshold, a Rs 50 fine). A
chunk that is one section keeps them sharp, and a section boundary never cuts a rule in half.

**Result** (`python chunk_documents.py`): 158 documents → **702 chunks**, average 106.5 tokens (min 21,
max 552). Every chunk keeps its parent's metadata (title, department, version, ...) and gets a
deterministic ID, `{document_id}_CHUNK_{nnn}`:

```json
{
  "chunk_id": "POL-ATTEND_CHUNK_001",
  "document_id": "POL-ATTEND",
  "title": "Mandatory Attendance and Condonation Policy",
  "section": "1. PURPOSE AND INSTITUTIONAL MANDATE",
  "text": "1. PURPOSE AND INSTITUTIONAL MANDATE\n\nDemo University maintains rigorous academic standards ...",
  "chunk_index": 0, "total_chunks": 7, "token_count": 69,
  "department": "Academic Affairs", "document_type": "Policy", "version": "3.1", "...": "..."
}
```

### 3.3 Embedding (`embedding/`)

**What:** every chunk is turned into a 384-number vector by `BAAI/bge-small-en-v1.5` and stored in
Supabase (PostgreSQL + pgvector). The text that is embedded is the **contextual passage**
`title + section heading + chunk text`, so a chunk deep inside a document still "knows" what the document
is about. Queries get the BGE instruction prefix
`"Represent this sentence for searching relevant passages: "`.

**Similarity** between a query vector $q$ and a chunk vector $d$ is cosine similarity. The vectors are
normalized to length 1, so it is just the dot product:

$$\cos(q, d) = \frac{q \cdot d}{\lVert q\rVert\,\lVert d\rVert} = \sum_{i=1}^{384} q_i\, d_i \quad(\text{since } \lVert q\rVert = \lVert d\rVert = 1)$$

1 means "same meaning", around 0 means "unrelated".

### 3.4 Query validation and normalization (`reformulation/validator.py`, `normalizer.py`)

**Validation:** the query must be a string, not blank, and at most 256 characters (`MAX_QUERY_LENGTH`).

**Normalization** is rule-based and conservative: it never removes meaning, and leaves anything ambiguous
alone. Real examples:

| Input | Normalized | What happened |
|:---|:---|:---|
| `attendence shortage` | `Attendance shortage` | typo fixed |
| `sem fees and dept regs` | `Semester fees and department regulations` | safe abbreviations expanded |
| `hostle curfew timings` | `Hostel curfew timings` | swapped letters fixed |
| `HOWWW to get bonafide cert???` | `How to get bonafide cert?` | repeated letters and punctuation collapsed |

Course codes (`CSE301`), document codes (`ACAD-REG-001`), acronyms (`GPA`, `ERP`) and ambiguous
abbreviations (`CS`, `UG`, `PG`) are kept exactly as typed.

### 3.5 Corpus spelling correction (`reformulation/spelling.py`)

**What:** a misspelled word is corrected to a word that **actually occurs in the indexed documents**, so a
correction can only ever point at something search can find.

**Edit distance** (Damerau–Levenshtein): the number of single-letter insertions, deletions,
substitutions or swaps of two neighbouring letters needed to turn one word into another.
`hostle → hostel` is 1 (swap `l` and `e`).

**Rules** (a word is left alone unless all hold):
1. it is at least 4 letters, purely alphabetic, and **not** already a word of the corpus;
2. some corpus word is **exactly 1 edit** away and starts with the same letter;
3. if two candidates tie, the winner must be **at least twice as frequent**; otherwise it is a guess and
   nothing changes.

| Typo | Correction | Edits | Corpus frequency |
|:---|:---|---:|---:|
| `attendence` | `attendance` | 1 | 140 |
| `hostle` | `hostel` | 1 | 100 |
| `librery` | `library` | 1 | 93 |
| `allotmnt` | `allotment` | 1 | 27 |
| `revalution` | *(unchanged)* | — | — |

`revalution` is two edits from `revaluation`, so it is left alone; the LLM rewrites and the vector search
still cope with it. Allowing 2 edits was tried and rejected: it turned `revalution` into `resolution`.

### 3.6 LLM reformulation (`reformulation/reformulator.py`, `reformulation/llm/`)

**What:** the normalized query goes to an LLM with a prompt that asks for up to 5 short, search-style
rewrites that keep the meaning, identifiers and numbers, without answering the question or inventing
facts. The reply must be JSON: `{"reformulations": ["...", "..."]}`. Rewrites identical to the original
or to each other are dropped.

**Why:** users and documents use different words. A student asks *"who has the final say in running the
university"*; the document is titled *"Constitution and Powers of the Executive Council"*. The rewrite
*"University governance final authority"* bridges that gap.

**Reliability:**
- **Key fail-over** (`llm/fallback.py`): several keys (Gemini, Groq, OpenRouter, Z.ai, OpenAI) are
  tried in order. A key that hits its quota or times out is skipped for 60 s; a rejected key is dropped.
- **Cache:** the rewrites of the last 256 queries are kept in memory, so a repeated query costs no LLM call.
- **Fallback:** if every key fails, the search runs on the original query and reports why.

### 3.7 Redundancy filter (`reformulation/semantic_similarity.py`)

**What:** each rewrite costs a full search, so rewrites that add nothing are removed. A rewrite is
**redundant** with an already-kept query when any of these is true:

1. it is the same text (ignoring case);
2. cosine similarity $\ge 0.97$ (practically the same meaning);
3. cosine similarity $\ge 0.85$ **and** it adds no new keyword, i.e. its content words are the same, a
   subset, or overlap with Jaccard $\ge 0.75$.

**Jaccard overlap** of the two sets of content words (stop words removed):

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

Rule 3 is the important one: a rewrite that **means the same but uses a new word** is **kept**, because
that new word can match a document that the original words miss (the keyword branch needs exact words).

**Worked examples** (real similarities from the model):

| Query A (kept) | Query B (candidate) | cos | Content words B adds | Jaccard | Result |
|:---|:---|---:|:---|---:|:---|
| How can I check my attendance? | How do I check my attendance? | 0.996 | none | 1.00 | **dropped** (rule 2: cos ≥ 0.97) |
| Hostel fee payment | Hostel fee payment deadline | 0.903 | `deadline` | 0.75 | **dropped** (rule 3: 3/4 = 0.75 ≥ 0.75) |
| Rules about silence hours in hostel | Rules for silence times in hostel | 0.967 | `times` | 0.50 | **kept** (new word, overlap 0.50 < 0.75) |
| How can I check my attendance? | What is my current attendance percentage? | 0.767 | `current`, `percentage` | 0.25 | **kept** (cos 0.767 < 0.85: related, but asks something different) |

Calculation for row 2: A = {fee, hostel, payment}, B = {deadline, fee, hostel, payment};
$|A \cap B| = 3$, $|A \cup B| = 4$, $J = 3/4 = 0.75$.

The result does not depend on the order in which the LLM returned the rewrites.

### 3.8 Hybrid retrieval (`retrieval/`)

Each query is searched two ways, and the two result lists are merged.

**(a) Vector search** (`vector_retriever.py`): the 20 chunks with the highest cosine similarity to the
query vector, from Supabase (`match_chunks_vector`, HNSW index). Good at **meaning** ("final say" ≈
"authority").

**(b) BM25 keyword search** (`bm25_retriever.py`): an in-process Okapi BM25 index over title, section
and chunk text. Good at **exact words**, codes and numbers (`CSE301`, `75%`). For a query with words
$t$ and a chunk $d$:

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t)\cdot\frac{f(t,d)\,(k_1+1)}{f(t,d) + k_1\left(1 - b + b\,\frac{|d|}{\text{avgdl}}\right)}$$

$$\text{IDF}(t) = \ln\!\left(1 + \frac{N - n(t) + 0.5}{n(t) + 0.5}\right)$$

- $f(t,d)$ = how often word $t$ occurs in chunk $d$; $|d|$ = chunk length in words;
  avgdl = average chunk length (79.24); $N$ = number of chunks (702); $n(t)$ = chunks containing $t$.
- $k_1 = 1.2$ limits how much repeating a word helps; $b = 0.75$ stops long chunks from winning just by
  being long.
- Words are lowercased, stop words removed, and lightly stemmed (`hours → hour`, `fees → fee`).

**IDF makes rare words count more** (real values from this corpus):

| Word | Chunks containing it, $n(t)$ | IDF |
|:---|---:|---:|
| `student` | 367 | 0.65 |
| `semester` | 150 | 1.54 |
| `hostel` | 89 | 2.06 |
| `condonation` | 28 | 3.21 |
| `silence` | 6 | 4.68 |

**Worked example:** query *"rules about silence hours in hostel"* → words `rule, about, silence, hour,
hostel`. For chunk `HSTL-REG-003_CHUNK_002` (64 words; `silence` 6 times, `hour` 6 times):

- length factor: $k_1(1 - b + b \cdot 64/79.24) = 1.2 \times (0.25 + 0.6058) = 1.0269$
- `silence`: $4.6836 \times \dfrac{6 \times 2.2}{6 + 1.0269} = 4.6836 \times 1.8785 = 8.798$
- `hour`: $2.0281 \times 1.8785 = 3.810$
- the other words do not occur in this chunk → **BM25 = 8.798 + 3.810 = 12.608**, the top keyword result.

**(c) Merging the two lists: Reciprocal Rank Fusion (RRF).** Scores from the two searches are on
different scales (cosine 0–1, BM25 0–20+), so RRF uses only the **rank** in each list:

$$\text{RRF}(d) = \sum_{\text{lists } L} \frac{1}{k + \text{rank}_L(d)}, \qquad k = 60$$

A chunk missing from a list gets nothing from it. **Worked example:** for the query
*"who has the final say in running the university"*, chunk `EXAM-NOT-001_CHUNK_001` is rank 15 in the
vector list and rank 3 in the BM25 list:

$$\text{RRF} = \frac{1}{60+15} + \frac{1}{60+3} = 0.01333 + 0.01587 = 0.02921$$

A chunk at rank 1 in only one list scores $1/61 = 0.01639$, so a chunk that **both** searches find
reasonably high beats a chunk that only one search ranks first.

### 3.9 Multi-query fusion (`reformulation/search_pipeline.py`)

Every query that survived the filter (the original plus the kept rewrites) gets its own hybrid list
(top 50). The lists are merged with RRF again:

$$\text{score}(d) = \sum_{j=1}^{m} \frac{w_j}{k + \text{rank}_j(d)}, \qquad k = 60,\ w_j = 1$$

A chunk that **several** versions of the query rank highly rises to the top; a chunk that only one odd
rewrite likes does not. Worked example in [section 6](#6-demo-one-query-end-to-end).

### 3.10 Grounded answers (`reformulation/answer_generator.py`)

With `answer: true`, the LLM receives **only** the top 5 chunks, each labelled with its chunk ID, and must
answer from them in JSON with the IDs it used. Then:

- any cited ID that was **not** among the 5 chunks is thrown away;
- an "answer" that cites **none** of the given chunks is replaced by
  *"The available documents do not answer this question."* (`answerable: false`).

So the model cannot cite a document it was not shown. It is not checked whether each sentence is
supported by the chunk it cites.

---

## 4. How it is evaluated

### 4.1 Query sets

| Set | File | Queries | Used for |
|:---|:---|---:|:---|
| Tuning | `evaluation/golden_queries.json` | 50 | Designing and tuning the pipeline. **Not** a fair test any more. |
| Held-out test | `evaluation/heldout_queries.json` | 100 (94 answerable) | **All reported results.** |

The held-out queries were written by someone who saw only department names and document titles, never
the document text: 10 queries in each of 10 types (keyword, natural-language question, abbreviation,
typo, course code, numeric, policy, eligibility, procedure, lexical diversity). Relevance was judged
separately, by reading the documents (not by using the search engine), on a 0–3 scale:

- **3** = answers the query; **2** = needed for a complete answer; **1** = useful background.
- Grade 2–3 chunks count as "relevant" for Recall and MRR; grade 1 only adds partial credit in NDCG.
- 6 queries the documents cannot answer are excluded from the metrics.

The LLM rewrites for every query are generated once and saved (`evaluation/cache/`), so the evaluation
is deterministic, repeatable and needs no LLM.

For reference, the full pipeline scores Recall@10 0.86, MRR 0.659, NDCG@10 0.638 on the tuning set
(`evaluation/results/tune_set_report.md`). Do not quote these as test results: the pipeline was tuned on
these queries.

### 4.2 Metrics, with a worked example

Let $R$ be the set of relevant chunks for a query, and the system's ranked list be $d_1, d_2, \dots$

**Recall@K:** what fraction of the relevant chunks appear in the top K?

$$\text{Recall@}K = \frac{|\{d_1,\dots,d_K\} \cap R|}{|R|}$$

**MRR (Mean Reciprocal Rank):** how high is the **first** relevant chunk? Averaged over all queries.

$$\text{RR} = \frac{1}{\text{rank of first relevant chunk}}, \qquad \text{MRR} = \frac{1}{|Q|}\sum_{q \in Q} \text{RR}_q$$

**NDCG@K:** rewards putting the **most** relevant chunks **highest**, using the 0–3 grades:

$$\text{DCG@}K = \sum_{i=1}^{K} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \qquad \text{NDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}$$

IDCG is the DCG of the perfect ordering, so NDCG is between 0 (nothing useful) and 1 (perfect).

**Worked example**: query H096, *"who has the final say in running the university"*.
Relevant chunks: `GOV-ACT-001_CHUNK_001` (grade 3), `_CHUNK_003` (2), `_CHUNK_004` (2); background:
`GOV-REG-002_CHUNK_001` (1).

| | Before (raw query) | After (full pipeline) |
|:---|:---|:---|
| Ranks of graded chunks | `CHUNK_001` at 14, the rest not in top 10 | `CHUNK_001` at 2, `GOV-REG-002` at 4, `CHUNK_004` at 7 |
| Recall@10 | $0/3 = 0.00$ | $2/3 = 0.67$ |
| Reciprocal rank | $1/14 = 0.071$ | $1/2 = 0.50$ |
| NDCG@10 | $0.00$ | $0.540$ |

NDCG@10 after, step by step:

$$\text{DCG} = \frac{2^3-1}{\log_2 3} + \frac{2^1-1}{\log_2 5} + \frac{2^2-1}{\log_2 8} = \frac{7}{1.585} + \frac{1}{2.322} + \frac{3}{3} = 4.417 + 0.431 + 1.000 = 5.847$$

$$\text{IDCG} = \frac{7}{\log_2 2} + \frac{3}{\log_2 3} + \frac{3}{\log_2 4} + \frac{1}{\log_2 5} = 7 + 1.893 + 1.500 + 0.431 = 10.824$$

$$\text{NDCG@10} = 5.847 / 10.824 = 0.540$$

### 4.3 Is a difference real? Significance tests (`evaluation/significance.py`)

With ~100 queries, one query changing moves an average by ~0.01, so a difference must be tested before
it is believed.

**Paired randomization test.** For each query, take the difference $\Delta_q = \text{after}_q - \text{before}_q$.
If the pipeline made no real difference, each $\Delta_q$ would be equally likely to be positive or
negative. So: flip the sign of each $\Delta_q$ at random, recompute the mean, and repeat 10,000 times.
The **p-value** is the fraction of those random means at least as far from 0 as the real one. A small p
(< 0.05) means "this gain is very unlikely to be luck".

**95% confidence interval.** Resample the queries with replacement 10,000 times (bootstrap) and take the
middle 95% of the mean differences.

**Holm correction.** 9 tests are reported together, so by chance alone one might look significant. Holm
fixes this: sort the $m = 9$ p-values from smallest to largest, multiply the $i$-th smallest by
$(m - i + 1)$, and never let an adjusted value be smaller than the one before it. Real values:

| Test | p | × | Holm p | Significant? |
|:---|---:|---:|---:|:---:|
| Whole query layer, Recall@10 | 0.0003 | 9 | 0.0027 | yes |
| LLM rewrites, Recall@10 | 0.0017 | 8 | 0.0136 | yes |
| Whole query layer, NDCG@10 | 0.0051 | 7 | 0.0357 | yes |
| LLM rewrites, NDCG@10 | 0.0089 | 6 | 0.0534 | no (just over 0.05) |
| LLM rewrites, MRR | 0.0651 | 5 | 0.3255 | no |
| ... 4 more | ≥ 0.186 | | ≥ 0.74 | no |

---

## 5. Results in detail

All numbers: held-out test set, 94 queries (`evaluation/results/heldout_report.md`).

### 5.1 All systems

| System | R@5 | R@10 | R@20 | MRR | NDCG@10 |
|:---|---:|---:|---:|---:|---:|
| BM25 only | 0.653 | 0.792 | 0.864 | 0.609 | 0.611 |
| Vector only | 0.780 | 0.857 | 0.928 | 0.751 | 0.707 |
| Hybrid, raw query (**before**) | 0.770 | 0.875 | 0.933 | 0.691 | 0.691 |
| + normalization + spelling | 0.787 | 0.893 | 0.945 | 0.678 | 0.696 |
| + normalization + LLM, no spelling | 0.836 | 0.937 | 0.965 | 0.738 | 0.743 |
| **Full pipeline (after)** | **0.845** | **0.944** | **0.965** | **0.727** | **0.739** |

### 5.2 What each layer adds

| Layer | Metric | Before | After | Change | 95% CI | Holm p | Significant |
|:---|:---|---:|---:|---:|:---:|---:|:---:|
| Normalization + spelling | Recall@10 | 0.875 | 0.893 | +0.018 | [+0.000, +0.041] | 0.745 | no |
| | NDCG@10 | 0.691 | 0.696 | +0.005 | [−0.008, +0.019] | 0.882 | no |
| LLM rewrites | Recall@10 | 0.893 | 0.944 | +0.051 | [+0.020, +0.084] | 0.014 | **yes** |
| | MRR | 0.678 | 0.727 | +0.049 | [−0.002, +0.100] | 0.326 | no |
| | NDCG@10 | 0.696 | 0.739 | +0.044 | [+0.012, +0.077] | 0.053 | no (borderline) |
| **Whole query layer** | Recall@10 | 0.875 | 0.944 | **+0.068** | [+0.031, +0.109] | 0.003 | **yes** |
| | MRR | 0.691 | 0.727 | +0.036 | [−0.018, +0.091] | 0.745 | no |
| | NDCG@10 | 0.691 | 0.739 | **+0.048** | [+0.015, +0.083] | 0.036 | **yes** |

### 5.3 By query type (NDCG@10, before → after)

| Query type | n | Before | After | Change |
|:---|---:|---:|---:|---:|
| Lexical diversity (user's words ≠ document's words) | 10 | 0.430 | 0.621 | **+0.192** |
| Natural-language question | 10 | 0.591 | 0.749 | +0.157 |
| Numeric | 8 | 0.751 | 0.840 | +0.089 |
| Abbreviation | 10 | 0.652 | 0.731 | +0.079 |
| Procedure | 9 | 0.713 | 0.768 | +0.055 |
| Eligibility | 9 | 0.694 | 0.710 | +0.017 |
| Keyword | 9 | 0.814 | 0.829 | +0.015 |
| Typo | 10 | 0.830 | 0.801 | −0.029 |
| Policy | 10 | 0.725 | 0.693 | −0.031 |
| Course code | 9 | 0.745 | 0.675 | **−0.070** |

**Why this pattern:** rewriting helps most when the user's words are not the document's words, because
the rewrites add the document's vocabulary. It slightly hurts queries that already contain the exact term
(a course code, a policy name): the original query alone would rank the right chunk first, and the
rewrites bring in neighbouring chunks that dilute it. With 8–10 queries per type, these per-type numbers
are indicative only.

### 5.4 Per query

Of 94 queries, the full pipeline improved NDCG@10 on **53**, made it worse on **32**, and left **9**
unchanged (vs. hybrid on the raw query). Gains are larger than losses on average, which is what the
significant mean improvement reflects. It does not help every query.

---

## 6. Demo: one query, end to end

Held-out query **H096**: *"who has the final say in running the university"* (type: lexical diversity).
The answer is in *"Constitution and Powers of the Executive Council"* (`GOV-ACT-001`), which never uses
the phrase "final say" or the word "running". Produced with the same code and settings as the evaluation
(local vector index, saved LLM rewrites).

**Step 1–2, normalize:** `Who has the final say in running the university` (only capitalised; nothing to fix).

**Step 3, LLM rewrites:**
1. University governance final authority
2. Who manages the university
3. University chief executive decision maker
4. Highest authority in university administration

**Step 4, filter:** all 4 kept. Their similarity to the original is 0.72–0.80, below the 0.85 threshold,
so each one counts as a genuinely different search.

**Before: hybrid search on the raw query (top 5)**

| # | Chunk | Document | Correct? |
|---:|:---|:---|:---:|
| 1 | `EXAM-NOT-001_CHUNK_001` | Fall Semester End Examination Timetable | ✗ |
| 2 | `IT-POL-001_CHUNK_003` | Acceptable IT Use Policy | ✗ |
| 3 | `STU-POL-006_CHUNK_003` | Grievance Redressal: Appellate Procedure and Finality | ✗ |
| 4 | `PLC-POL-001_CHUNK_001` | Campus Recruitment Policy | ✗ |
| 5 | `POL-GRADUATION_CHUNK_001` | Degree Award and Graduation Requirements | ✗ |

The correct chunk is at **rank 14**. The raw words only match on the surface: the keyword search finds
"the **final** schedule" in the exam timetable and "**running** unauthorized file-sharing hubs" in the IT
policy; the vector search finds chunks about "finality" of appeals and degree awards.

**After: full pipeline (top 5)**

| # | Chunk | Document | Rank in each of the 5 query lists | Score | Correct? |
|---:|:---|:---|:---|---:|:---:|
| 1 | `ACAD-POL-002_CHUNK_001` | Department Academic Governance and HOD Decision Authorities | 12, 1, 8, 1, 1 | 0.0778 | ✗ (related) |
| 2 | `GOV-ACT-001_CHUNK_001` | **Constitution and Powers of the Executive Council** | 14, 7, 2, 2, 4 | 0.0763 | **✓ grade 3** |
| 3 | `ACAD-POL-002_CHUNK_002` | Department Academic Governance: Hierarchy | 6, 9, 5, 4, 12 | 0.0745 | ✗ (related) |
| 4 | `GOV-REG-002_CHUNK_001` | Academic Council Mandate and Standing Committee Powers | 21, 8, 3, 29, 3 | 0.0700 | background (grade 1) |
| 5 | `OPS-POL-001_CHUNK_001` | Campus Operations Estate Maintenance | 34, 21, 7, 18, 17 | 0.0637 | ✗ |

The correct chunk moves **from rank 14 to rank 2**. Its score:

$$\frac{1}{60+14} + \frac{1}{60+7} + \frac{1}{60+2} + \frac{1}{60+2} + \frac{1}{60+4} = 0.01351 + 0.01493 + 0.01613 + 0.01613 + 0.01563 = 0.0763$$

| Metric | Before | After |
|:---|---:|---:|
| Recall@10 | 0.00 | 0.67 |
| Reciprocal rank | 0.071 | 0.50 |
| NDCG@10 | 0.00 | 0.54 |

**Justification:**
- The original query alone ranks the correct chunk 14th: its words ("final", "say", "running") are not the
  document's words.
- Each rewrite uses governance vocabulary (`authority`, `governance`, `chief executive`, `highest
  authority`), and three of the four put the chunk in their top 4 (ranks 2, 2, 4).
- Multi-query RRF rewards a chunk that **several** versions of the query agree on. The off-topic results
  of the raw query (the exam timetable, the IT policy) appear in only one list, so they drop out of the
  top 5.
- The new #1 is not a wrong answer by accident: it is the department-level governance policy, a
  reasonable but less complete answer (not graded relevant). The pipeline found the right *area*; the
  ranking within that area is what limits MRR.

### Where it gets worse, and why

Held-out query **H063**, *"rules about silence hours in hostel"* (type: policy):

| | Before | After |
|:---|---:|---:|
| Rank of the correct chunk (`HSTL-REG-003_CHUNK_002`, "Silence Hours") | **1** | 3 |
| NDCG@10 | 0.94 | 0.47 |

The raw query already contains the document's exact words ("silence hours", which is in the section
title), so both searches put the right chunk first. The rewrites say "quiet hours" and "noise
regulations", which also match the **curfew** policy and the contraband rules. In the 5 query lists the
correct chunk ranks 1, 14, 1, 1, 9, while the curfew chunk ranks 6, 1, 8, 3, 1: steadier, so it wins the
fusion. This is the typical failure: **the query was already precise, and the rewrites spread it out.**
It is why policy and course-code queries lose slightly in section 5.3.

---

## 7. Ideas that were tested and rejected

These were measured on the 50-query tuning set (with earlier versions of the pipeline) and dropped
because they did not help. Reports are kept in `evaluation/results/`.

| Idea | Result | Why it was rejected |
|:---|:---|:---|
| **Cross-encoder reranker** (`ms-marco-MiniLM-L-6-v2`) on the top 50 (`phase4_report.md`) | Recall@10 0.69 → 0.62 (pure reranker); 0.69 → 0.67 (mixed with RRF) | The model is trained on web search, and here it often preferred a chunk whose title matched the query over the chunk with the answer (e.g. a course outline's header chunk over its lab chunk, query Q024). Code kept, off by default (`RERANKER_ENABLED=false`). |
| **Organisation glossary in the LLM prompt** (department, system and document names) (`tune_glossary_report.md`) | Recall@10 0.87 → 0.78, NDCG@10 0.646 → 0.562 | The LLM copied document titles instead of restating the user's need; a title matches every chunk of a document and pushes the answering chunk down. |
| **Extra weight for the original query** in fusion, $w_1 = 1.5$ or $2$ (`rrf_weight_analysis.md`) | changes ≤ 0.01 on every metric | No measurable effect. |
| **Two-edit spelling correction** | `revalution` → `resolution` | Too many wrong corrections; 1 edit only. |

Things that **were** adopted on the tuning set (`phase5_report.md`): embedding the contextual passage
(title + section + text) instead of the bare text, and fusing the top 50 of each query instead of the
top 20.

---

## 8. Setup and usage

### Install

```bash
pip install -e ".[chunking,api,dev]"
cp .env.example .env          # fill in SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY and at least one LLM key
```

LLM keys: paste any of `GEMINI_API_KEY`, `GEMINI_API_KEY_2`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`,
`ZAI_API_KEY`, `OPENAI_API_KEY`; they are tried in order. Without any key the search still works, without
rewrites or answers.

### Build the index (offline pipeline)

```bash
python validate_corpus.py -i Data/documents      # → Data/validated/
python chunk_documents.py                        # → Data/chunks/chunks.jsonl
python ingest_chunks.py --dry-run                # embed and check everything, upload nothing
python ingest_chunks.py                          # embed and upsert into Supabase
```

Database tables and search functions: `supabase/migrations/` (apply in order).

### Search API

```bash
uvicorn api.app:app --port 8000
```

```bash
curl -X POST localhost:8000/search -H "Content-Type: application/json" \
  -d '{"query": "revaluation fee and deadline", "top_k": 5, "answer": true, "debug": true}'
```

| Field | Meaning |
|:---|:---|
| `query` | the user's query (1–256 characters) |
| `top_k` | number of results, 1–50 (default 10) |
| `domain` | optional: only search one department, e.g. `"Academic Affairs"` |
| `max_variants` | maximum LLM rewrites, 1–10 (default 5) |
| `answer` | `true` adds an LLM answer grounded in the top 5 results, with cited chunk IDs |
| `debug` | `true` adds the trace: normalized query, rewrites, queries used, timings per stage |

The response lists the ranked chunks (`chunk_id`, `document_id`, `title`, `section`, `score`, `text`);
`degraded: true` means part of retrieval failed and the results came from what remained.
`GET /health` shows which components are active. Each new query uses one LLM call for rewrites (plus one
for `answer: true`); free-tier quotas run out quickly, repeated queries use the cache.

### Demos

```bash
python jury_demo.py          # terminal: every stage printed, before vs after, rank movement
python demo_server.py        # browser demo at http://localhost:8001
```

### Evaluation

```bash
python evaluation/run_heldout_evaluation.py      # held-out test set → evaluation/results/heldout_report.md
python evaluation/run_heldout_evaluation.py --golden evaluation/golden_queries.json \
    --reformulations evaluation/cache/reformulations.json --tag tune_set   # tuning set
```

Both run offline and deterministically from the saved rewrites. To create rewrites for a new query set:
`python evaluation/build_reformulation_cache.py --queries <file> --model <llm>`.
`evaluation/run_beir_evaluation.py` runs the same comparison on public BEIR datasets (FiQA, an MS MARCO
sample) against a non-LLM expansion baseline (RM3); see the script's docstring for the data download.

### Tests

```bash
python -m pytest             # 468 offline tests; Supabase and the LLM are mocked
```

CI runs them on every push (`.github/workflows/tests.yml`).

---

## 9. Project structure

```
Data/
  documents/            raw corpus: 158 documents in 12 domain files
  validated/            validated corpus + validation reports
  chunks/               702 chunks (chunks.jsonl) + chunking report
  demo_university_world_model.json   departments, systems, roles (ground truth for validation)
corpus_validation/      11-stage document validation
document_chunking/      structure-aware recursive chunking
embedding/              BGE embeddings, Supabase upload and checks
retrieval/              vector search, BM25, hybrid RRF, RM3 baseline
reformulation/          validation, normalization, spelling, LLM rewrites, filter, pipeline, answers
  llm/                  provider-neutral LLM client with key fail-over
api/                    FastAPI search service
evaluation/             query sets, metrics, significance tests, evaluation scripts, reports
supabase/migrations/    database schema, search functions, access lock-down
web_demo/               browser demo page
tests/                  test suite
```

---

## 10. Known limitations

- **Synthetic corpus.** The 158 documents describe a fictional university; results on real
  organisational documents may differ.
- **Small test set.** 94 queries: differences below about 0.02 are noise, and per-type results (8–10
  queries each) are only indicative.
- **Ranking within the right area.** MRR does not improve significantly: the pipeline usually finds the
  right documents, but not always the best chunk first.
- **Precise queries can get worse** (course codes, exact policy names; see the H063 example).
- **Spelling correction treats every word missing from the corpus as a typo:** "fake" becomes "fare"
  because "fake" never appears in the documents.
- **No link to the full document.** Results and citations give the chunk and its document ID, but the API
  has no endpoint that returns a whole document.
- **LLM quota.** Each new query needs an LLM call; on free tiers the rewrites are often skipped, which
  falls back to plain hybrid search.

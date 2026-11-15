# Evaluation datasets and split boundaries

`manifest.json` defines three splits. `splits.load_split()` enforces them: every query file is
hash-checked, and the blind test refuses to load until its human labels are frozen.

| Split | File | Status | May be used for |
|:---|:---|:---|:---|
| `development` | `evaluation/golden_queries.json` (50) | Tuning set; label provenance unrecorded | Anything: tuning, prompts, thresholds, debugging |
| `validation_contaminated` | `evaluation/heldout_queries.json` (100) | Used while developing the intent router and contextual reranking | Validation, **reported under this name only**, never as an independent blind test |
| `blind_test` | `evaluation/datasets/blind_test_v1.json` (66) | Queries written; **not labelled** | Final evaluation **only after** human labels are frozen; never for tuning |

## Blind test v1

- **Composition:**
  - 34 proposed-unanswerable queries. Each passed a corpus absence check (`check_unanswerable.py` → `unanswerable_absence_check.json`).
  - 10 typo-heavy queries.
  - 10 multi-intent queries.
  - 12 precise queries: 4 course codes, 4 document IDs, 4 named systems.
- **Authorship:** the queries were written by an AI assistant that had read the corpus, so they are **not blind-authored**. Annotators should review them, and replace or extend them, before freezing.
- **"Proposed unanswerable" is not a label.** An annotator must confirm each one, because the term check can miss synonyms.

## Annotation workflow

1. Run `python evaluation/datasets/build_annotation_pool.py`. It pools the top-10 results of configurations A–D per query (in hash order, blind to system and rank) into:
   - `blind_test_v1_annotation_pool.csv`: grade each (query, chunk) row 0–3;
   - `blind_test_v1_answerability.csv`: mark each query fully / partially / unanswerable.

   Grades, annotator IDs and timestamps are blank. Annotators may add rows for relevant chunks the pool missed.
2. Grade with the rubric in `evaluation/judge/rubric.py` (`relevance-rubric-v1`). The rubric is the same one the LLM judge uses, so judge and human grades are comparable. Ideally two annotators grade every row; disagreements resolve to the median grade.
3. Describe annotators in `annotators.json`, then run `python evaluation/datasets/labels_from_csv.py --annotators evaluation/datasets/annotators.json`. It refuses to write labels while any row is incomplete.
4. Run `python evaluation/datasets/freeze_blind_test.py` once. It records the label hash in `manifest.json`. Later edits to the labels make the blind split fail to load.
5. Run `python evaluation/run_ad_benchmark.py --split blind_test`. Report that result once, without retuning afterwards.

Until step 4 is done, every blind-test metric is **NOT EVALUATED**.

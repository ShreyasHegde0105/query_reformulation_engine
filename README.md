# AI-Powered Enterprise Query Reformulation & Search Engine

## Enterprise Pipeline Overview

```
Document Generation (Modules 1–12)          ✅ COMPLETED (157 enterprise documents)
        ↓
Corpus Validation                           ✅ COMPLETED (Data/validated/)
        ↓
Document Chunking                           ✅ COMPLETED (Data/chunks/)
        ↓
Embedding Generation                        ⏳ Next Phase
        ↓
Vector / Hybrid Indexing                    ⏳ Upcoming
        ↓
Retrieval (Dense + BM25)                    ⏳ Upcoming
        ↓
Query Reformulation Engine                  ⏳ Upcoming
        ↓
Benchmark Evaluation                        ⏳ Upcoming
```

---

# Document Chunking Module (`document_chunking`)

The **Document Chunking Module** is a production-grade, deterministic, structure-aware chunking pipeline designed to convert validated enterprise JSON documents into retrieval-friendly chunks ready for downstream dense embedding and hybrid search.

---

## 1. Why Chunking is Required
Embedding models and downstream neural rerankers operate with finite context windows (e.g. 512, 1024, or 8192 tokens). If an entire university policy or standard operating procedure (often 1,500 to 4,000 words) is embedded as a single monolithic vector:
- Fine-grained details (such as a 75.0% mandatory attendance threshold, a 65.0%–74.9% condonation band, a 48-hour hardware error escalation window, or specific fine amounts) are diluted across the averaged dense representation.
- Dense retrieval loses precision when queries ask targeted procedural questions.
- Generative LLMs receive excessive irrelevant context, increasing latency and the risk of hallucination.

## 2. Why Structure-Aware Chunking is Used
Traditional naive chunking (such as blind splitting every $N$ characters or fixed token windows) breaks sentences mid-thought, separates figures from their units, and severs numbered regulatory conditions. 

In contrast, **Structure-Aware Recursive Chunking** respects the natural hierarchical boundaries of enterprise documents:

```
DOCUMENT
   ↓
SECTION (e.g. 1. PURPOSE, 2. SCOPE, 3. REGULATIONS, 4. PROCEDURE...)
   ↓
PARAGRAPH (Double newlines & structured list items)
   ↓
SENTENCE (Grammatical boundaries ending in [.!?])
   ↓
TOKEN FALLBACK (Graceful final fallback for giant sentences)
   ↓
CHUNK (Self-contained, retrieval-optimized unit)
```

1. **Priority 1: Section Boundaries:** Attempts to keep cohesive numbered institutional sections intact.
2. **Priority 2: Paragraph Boundaries:** Splits oversized sections at logical paragraph breaks.
3. **Priority 3: Sentence Boundaries:** Splits oversized paragraphs at grammatical punctuation boundaries.
4. **Priority 4: Token Fallback:** Token-level splitting is strictly a graceful fallback for giant single sentences.

## 3. Why JSON is Not Itself Being Chunked
Enterprise documents in this corpus are stored in structured JSON format (`document_id`, `title`, `metadata`, `content`). The JSON brackets, quotation marks, and key names are structural database containers, not natural language prose. 

Chunking the raw JSON syntax would:
- Sever valid JSON formatting across chunk boundaries.
- Pollute embedding vector space with repeated metadata keys (`"retrieval_signals": [...]`, `"systems": [...]`).
- Harm dense retrieval semantics.

Instead, the prose body is cleanly extracted from `content`, parsed structurally, and enriched with parent document metadata in each resulting chunk.

## 4. How the `content` Field is Extracted and Normalized
1. The validated corpus is loaded from `Data/validated/validated_documents.json` using [`DocumentLoader`](file:///C:/Users/dell/Desktop/Query_Reformulation_Engine/document_chunking/loader.py).
2. The `content` string is extracted without mutating the source document.
3. Prior to chunking, [`TextNormalizer`](file:///C:/Users/dell/Desktop/Query_Reformulation_Engine/document_chunking/normalizer.py) applies non-destructive normalization:
   - Normalizes CRLF (`\r\n`) and CR (`\r`) to standard LF (`\n`).
   - Replaces non-standard Unicode whitespace (non-breaking spaces, thin spaces) with standard spaces.
   - Strips trailing whitespace per line to avoid false regex mismatches.
   - Collapses excessive blank lines (3+ newlines to `\n\n`) to preserve distinct paragraph boundaries.
   - **Data Integrity Rule:** Words, numbers, section headings, and punctuation are strictly preserved without rewriting or summarizing.

## 5. Section and Subsection Detection
Section detection is performed by [`SectionParser`](file:///C:/Users/dell/Desktop/Query_Reformulation_Engine/document_chunking/section_parser.py). It discovers section boundaries from the actual document content:
- **Numbered Section Headings:** Discovers headings such as `1. PURPOSE`, `2. SCOPE`, `3. REGULATIONS`, `4. PROCEDURE`, as well as title-case variants like `1. Purpose and Mandate`.
- **Configurable Patterns:** Uses `section_heading_pattern` from configuration.
- **Subsection Detection:** Discovers subsections such as `3.1 Mandatory Standard: ...`, `3.2 Computation Principle: ...`, extracting the clean subsection title (`3.1 Mandatory Standard`) and associating it with the generated chunk.
- **Graceful Unsectioned Fallback (CASE 1):** For documents without numbered section headings (such as `LIB-FAQ-001`), the parser sets `section = None` and `section_number = None`, falling back to paragraph, sentence, and token chunking.
- **No Artificial Headings:** The parser NEVER invents artificial sections such as "Introduction", "Summary", or "Conclusion" unless they actually exist in the source document.

## 6. Recursive Splitting
The recursive splitting engine in [`RecursiveSplitter`](file:///C:/Users/dell/Desktop/Query_Reformulation_Engine/document_chunking/splitter.py) follows a strict hierarchy:
1. **Section Level:** If the entire section fits within `max_chunk_tokens` (default: 600 tokens), it is kept intact as a single chunk.
2. **Paragraph Level:** If a section exceeds `max_chunk_tokens`, it is split by paragraphs and combined up to `target_chunk_tokens` (default: 500 tokens).
3. **Sentence Level:** If an individual paragraph exceeds `max_chunk_tokens`, it is split into sentences at grammatical boundaries.
4. **Token Fallback:** If a single sentence exceeds `max_chunk_tokens`, token-level boundary splitting is invoked as a final fallback.

## 7. Chunk Size Configuration
All sizing parameters are fully configurable via `config/chunking_config.json`:
- `target_chunk_tokens`: 500 (target size when combining units)
- `max_chunk_tokens`: 600 (hard threshold triggering recursive splitting)
- `min_chunk_tokens`: 100 (threshold below which trailing fragments are merged if possible)
- `overlap_tokens`: 75 (overlap size when a section must be split)

## 8. Overlap Behavior
When a section is split into multiple chunks, configurable overlap (default 75 tokens) is applied:
- **Strict Section-Internal Overlap:** Overlap is maintained strictly within the same section.
- **Boundary Isolation:** Text from `3. REGULATIONS` is never carried over into `4. PROCEDURE`, preserving semantic purity across sections.

## 9. Metadata Propagation & Lineage
Every generated chunk inherits the full parent document context:
- Document identity: `document_id`, `title`
- Classification: `department`, `department_id`, `document_type`
- Versioning: `version`, `effective_date`, `status`
- Targeted discovery: `audience`, `keywords`, `systems`, `roles`, `related_documents`
- Structural coordinates: `chunk_index`, `total_chunks`, `token_count`, `character_count`, `start_char`, `end_char`
- Section context: `section`, `section_number`, `subsection`

## 10. Retrieval Signals Preservation
The document schema contains `retrieval_signals` (natural-language search queries, colloquial questions, alternative phrasings, and common typos).
- **Critical Rule:** `retrieval_signals` are preserved strictly in chunk metadata.
- **Never Injected in Text:** They are NEVER appended to `chunk["text"]`, preventing contamination of formal policy prose during vector embedding.

## 11. Deterministic Chunk Identifiers
Chunk IDs are deterministically generated using the pattern:
```
{document_id}_CHUNK_{index:03d}
```
Examples:
- `POL-ATTEND_CHUNK_001`
- `POL-ATTEND_CHUNK_002`
- `POL-ATTEND_CHUNK_003`

Deterministic IDs guarantee full reproducibility across multiple runs, deterministic caching, and reliable document lineage tracing.

## 12. Output Format
The pipeline produces two output formats in `Data/chunks/`:
- `Data/chunks/chunks.json`: Standard JSON array containing all generated chunk objects.
- `Data/chunks/chunks.jsonl`: JSON Lines format for streaming and large-scale vector indexing.
- `Data/chunks/chunking_report.json`: Machine-readable execution statistics and quality audit.
- `Data/chunks/chunking_report.md`: Human-readable summary report.

---

## 13. Configuration (`config/chunking_config.json`)

```json
{
  "target_chunk_tokens": 500,
  "max_chunk_tokens": 600,
  "min_chunk_tokens": 100,
  "overlap_tokens": 75,
  "include_section_heading_in_text": true,
  "detect_numbered_sections": true,
  "detect_subsections": true,
  "enable_paragraph_split": true,
  "enable_sentence_split": true,
  "enable_token_fallback": true,
  "section_heading_pattern": "^(?:#{1,6}\\s+)?\\s*(\\d+)\\.\\s+([^\\n\\r]+?)$",
  "subsection_heading_pattern": "^(?:#{1,6}\\s+)?\\s*(\\d+\\.\\d+)\\s+([^:\\n\\r]+)(?::|\\s*-|\\n|$)"
}
```

---

## 14. CLI Usage

The module can be executed directly via Python module syntax or via the top-level runner:

```powershell
# Standard execution with default paths and report
python -m document_chunking --input Data/validated/validated_documents.json --output Data/chunks/chunks.json --report

# Using the root convenience script
python chunk_documents.py --input Data/validated/validated_documents.json --output Data/chunks/chunks.json

# Custom chunk sizing and overlap
python -m document_chunking --target-tokens 400 --max-tokens 500 --overlap 60 --min-tokens 80

# Specifying a custom configuration file
python -m document_chunking --config config/chunking_config.json
```

---

## 15. Example Input and Output

### Input Document (`Data/validated/validated_documents.json` snippet):
```json
{
  "document_id": "POL-ATTEND",
  "title": "Mandatory Attendance and Condonation Policy",
  "department": "Academic Affairs",
  "department_id": "DEPT-ACAD",
  "document_type": "Policy",
  "version": "3.1",
  "effective_date": "2026-08-01",
  "status": "Active",
  "audience": ["All Undergraduate Students", "Course Instructors"],
  "keywords": ["attendance percentage", "75% requirement", "condonation"],
  "retrieval_signals": ["how to get attendance condoned", "minimum attendance required"],
  "systems": ["AttendEdge", "UniCampus ERP"],
  "roles": ["Dean of Academic Affairs", "Course Instructor"],
  "related_documents": ["POL-CIE", "POL-SEE"],
  "content": "1. PURPOSE AND INSTITUTIONAL MANDATE\nDemo University maintains rigorous academic standards centered on active classroom instruction...\n\n2. JURISDICTION AND SCOPE\nThis policy governs all matriculated students...\n\n3. MINIMUM PRESCRIBED ATTENDANCE THRESHOLD\n3.1 Mandatory Standard: Every enrolled student must maintain a minimum cumulative attendance of 75.0%..."
}
```

### Generated Chunk (`Data/chunks/chunks.json` snippet):
```json
{
  "chunk_id": "POL-ATTEND_CHUNK_001",
  "document_id": "POL-ATTEND",
  "title": "Mandatory Attendance and Condonation Policy",
  "section": "1. PURPOSE AND INSTITUTIONAL MANDATE",
  "section_number": "1",
  "subsection": null,
  "text": "1. PURPOSE AND INSTITUTIONAL MANDATE\n\nDemo University maintains rigorous academic standards centered on active classroom instruction, collaborative studio inquiry, and continuous laboratory experimentation. Regular and punctual attendance in all scheduled instructional periods is fundamental to the learning lifecycle. This policy establishes the minimum quantifiable attendance standards, criteria for condonation of attendance shortages on authenticated grounds, and statutory consequences of academic detention.",
  "chunk_index": 0,
  "total_chunks": 7,
  "token_count": 69,
  "character_count": 515,
  "start_char": 0,
  "end_char": 477,
  "department": "Academic Affairs",
  "department_id": "DEPT-ACAD",
  "document_type": "Policy",
  "version": "3.1",
  "effective_date": "2026-08-01",
  "status": "Active",
  "audience": ["All Undergraduate Students", "Course Instructors"],
  "keywords": ["attendance percentage", "75% requirement", "condonation"],
  "retrieval_signals": ["how to get attendance condoned", "minimum attendance required"],
  "systems": ["AttendEdge", "UniCampus ERP"],
  "roles": ["Dean of Academic Affairs", "Course Instructor"],
  "related_documents": ["POL-CIE", "POL-SEE"]
}
```

---

## 16. Chunking Statistics (Validated Corpus Run)

Results from processing all 157 validated enterprise documents:

| Metric | Value | Description |
|---|---|---|
| **Total Documents Processed** | 157 | Validated input enterprise documents |
| **Total Chunks Generated** | 696 | Retrieval-optimized semantic chunks |
| **Average Chunks / Document** | 4.43 | Mean chunk granularity |
| **Average Chunk Tokens** | 106.7 | Mean token size across corpus |
| **Median Chunk Tokens** | 91.5 | Median token size |
| **Min Chunk Tokens** | 21 | Concise terminal/scope sections |
| **Max Chunk Tokens** | 552 | Largest intact section within boundary limits |
| **Sections Detected** | 694 | Numbered structural section headings identified |
| **Docs Without Headings** | 1 | Fallback paragraph-chunked document (`LIB-FAQ-001`) |
| **Recursive Section Splits** | 1 | Sections split recursively |
| **Sentence Fallbacks** | 0 | Oversized paragraphs requiring sentence splits |
| **Token Fallbacks** | 0 | Giant sentences requiring token fallback |
| **Quality Check Status** | **PASSED (0 Errors)** | All 10 structural integrity checks passed |

---

## 17. How the Output Connects to Embedding Generation

The output file [`Data/chunks/chunks.json`](file:///C:/Users/dell/Desktop/Query_Reformulation_Engine/Data/chunks/chunks.json) is completely decoupled from the upstream document generation and validation logic.

The upcoming **Embedding Generation** module can consume chunks directly with a single loop:

```python
import json

# 1. Load chunked corpus
with open("Data/chunks/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# 2. Dense embedding consumption pattern
for chunk in chunks:
    # chunk["text"] contains contextual section heading + section prose
    embedding_vector = embedding_model.encode(chunk["text"])
    
    # chunk["chunk_id"], chunk["document_id"], and metadata are ready for vector database insertion
    vector_db.upsert(
        id=chunk["chunk_id"],
        vector=embedding_vector,
        metadata={
            "document_id": chunk["document_id"],
            "title": chunk["title"],
            "department": chunk["department"],
            "department_id": chunk["department_id"],
            "document_type": chunk["document_type"],
            "section": chunk["section"],
            "section_number": chunk["section_number"],
            "subsection": chunk["subsection"],
            "keywords": chunk["keywords"],
            "retrieval_signals": chunk["retrieval_signals"],
            "systems": chunk["systems"],
            "roles": chunk["roles"],
            "related_documents": chunk["related_documents"],
        }
    )
```

---

## 18. Automated Quality Checks & Test Suite

The test suite contains **58 unit and integration tests** verifying all 25 specific test scenarios and 8 core invariant properties:

```powershell
# Run the complete test suite
python -m unittest discover -s tests -v
```

### Verified Core Invariants:
- **Property 1:** Every chunk belongs to exactly one source document.
- **Property 2:** Every source document is traced to its generated chunks.
- **Property 3:** Chunk IDs are strictly deterministic across runs.
- **Property 4:** Document ordering does not change chunk IDs.
- **Property 5:** Chunk metadata matches parent document metadata.
- **Property 6:** The validated source corpus is immutable and unchanged.
- **Property 7:** Chunks preserve source content rather than generating new text.
- **Property 8:** Section boundaries are respected whenever present.

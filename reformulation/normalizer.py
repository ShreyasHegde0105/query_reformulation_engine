"""Normalization utilities for query reformulation.

The public function :func:`normalize_query` prepares a raw query string for search
retrieval and LLM reformulation prompts. Normalization performs the following steps:

1. Type-guard – reject non-string inputs early.
2. Strip leading/trailing whitespace and collapse internal whitespace runs.
3. Clean and normalize punctuation:
   - Remove spaces before punctuation marks (e.g. "hello ," → "hello,").
   - Collapse repeated identical marks (e.g. "???" → "?", "!!!" → "!").
   - Collapse mixed emphasis sequences (e.g. "?!!!" or "?!" → "?").
   - Strip redundant/excessive enclosing brackets (e.g. "((grades))" → "grades").
   - Normalize question queries ending in trailing hesitation dots to question marks.
4. Token-level decomposition & cleaning:
   - Strip stray/malformed leading punctuation (e.g. ")To" → "To", ",check" → "check").
   - Strip unmatched brackets while preserving legitimate balanced brackets.
   - Decompose possessives ('s/'S/’s/’S) for independent root analysis.
5. Letter elongation / keystroke repeat normalization:
   - Normalize repeated character runs (e.g. "HOWWWW" → "HOW").
6. Safe typo correction:
   - Correct unambiguous typos using a curated dictionary of common English
     misspellings (e.g. "las" → "last", "semster" → "semester").
   - Optionally (``spell_corrector=``), correct other unknown words to a word of the
     indexed corpus (see :mod:`reformulation.spelling`).
7. Safe domain abbreviation expansion:
   - Expand safe, unambiguous abbreviations (e.g. "sem" → "semester", "prof" → "professor",
     "dept" → "department").
   - Preserve ambiguous abbreviations (e.g. "CS", "UG", "PG", "FT", "AI", "ML").
8. Sentence-casing & entity preservation:
   - Preserve known technical acronyms and domain entities (GPA, ERP, CSE, etc.).
   - Preserve proper nouns and CamelCase identifiers (GitHub, LinkedIn, YouTube, OpenAI, etc.).
   - Preserve course codes (e.g. CSE 301, CSE301, CSE-301).
   - Preserve dates, numbers, and document codes (e.g. 2025, 2025/26, ACAD-REG-001).
   - Preserve unknown alphanumeric identifiers and acronyms in normally-cased sentences.
   - Standardize standalone pronoun "I" and contractions ("I'm", "I've", "can't", "what's").
   - Apply human-readable sentence casing without altering protected entities.

Design constraints
------------------
* No external NLP libraries – runs on the Python standard library only.
* **Conservative by default**: ambiguous tokens are left untouched.
* Does **not** perform semantic query reformulation – that is the responsibility
  of :class:`reformulation.reformulator.QueryReformulator`.
* The function is deterministic, idempotent, and side-effect free.
* The original user query is **never** overwritten; callers must keep both
  ``original_query`` and ``normalized_query`` separately.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .spelling import CorpusSpellCorrector

# ---------------------------------------------------------------------------
# Internal data tables
# ---------------------------------------------------------------------------

# Curated abbreviation expansions.
# Tokens in this table are EXPANDED rather than preserved as-is.
_ABBREV_MAP: dict[str, str] = {
    "sem":    "semester",
    "sems":   "semesters",
    "dept":   "department",
    "depts":  "departments",
    "prof":   "professor",
    "profs":  "professors",
    "univ":   "university",
    "univs":  "universities",
    "acad":   "academic",
    "admin":  "administration",
    "lib":    "library",
    "libs":   "libraries",
    "regs":   "regulations",
    "yr":     "year",
    "yrs":    "years",
    "approx": "approximately",
    "govt":   "government",
}

# Curated typo corrections. Keys are lowercase misspellings.
_TYPO_MAP: dict[str, str] = {
    "laast":        "last",
    "las":          "last",
    "lst":          "last",
    "semster":      "semester",
    "semestre":     "semester",
    "semseter":     "semester",
    "smester":      "semester",
    "semter":       "semester",
    "semestter":    "semester",
    "acadmic":      "academic",
    "acadmeic":     "academic",
    "acdemic":      "academic",
    "academc":      "academic",
    "acadmics":     "academics",
    "universty":    "university",
    "univercity":   "university",
    "universitty":  "university",
    "universtiy":   "university",
    "universiy":    "university",
    "univresity":   "university",
    "univesity":    "university",
    "grdae":        "grade",
    "grdaes":       "grades",
    "gradse":       "grades",
    "gradees":      "grades",
    "polciy":       "policy",
    "polcy":        "policy",
    "poilcy":       "policy",
    "integirty":    "integrity",
    "integrty":     "integrity",
    "intergrity":   "integrity",
    "atendance":    "attendance",
    "attendace":    "attendance",
    "attendnce":    "attendance",
    "sylalbus":     "syllabus",
    "syllbus":      "syllabus",
    "slyllabus":    "syllabus",
    "syllabuss":    "syllabus",
    "corse":        "course",
    "coures":       "course",
    "cource":       "course",
    "facutly":      "faculty",
    "faculity":     "faculty",
    "faculy":       "faculty",
    "regsitration": "registration",
    "registraton":  "registration",
    "registraion":  "registration",
    "schdule":      "schedule",
    "schedle":      "schedule",
    "calndar":      "calendar",
    "calender":     "calendar",
    "examinaton":   "examination",
    "examintion":   "examination",
    "libary":       "library",
    "libraray":     "library",
    "hostle":       "hostel",
    "hospitel":     "hospital",
    "hospial":      "hospital",
    "hosiptal":     "hospital",
    "recieve":      "receive",
    "recive":       "receive",
    "collegue":     "colleague",
    "colleage":     "colleague",
    "accomodation": "accommodation",
    "accomdation":  "accommodation",
    "acheive":      "achieve",
    "acheivment":   "achievement",
    "comittee":     "committee",
    "commitee":     "committee",
    "committe":     "committee",
    "departmnt":    "department",
    "deparment":    "department",
    "departement":  "department",
    "subejct":      "subject",
    "subjet":       "subject",
    "profesor":     "professor",
    "professsor":   "professor",
    "proffessor":   "professor",
    "proffesor":    "professor",
    "proffsor":     "professor",
    "stuent":       "student",
    "studnet":      "student",
    "studennt":     "student",
    "admision":     "admission",
    "eligiblity":   "eligibility",
    "elgibility":   "eligibility",
    "requirment":   "requirement",
    "requriment":   "requirement",
}

# Tokens with known canonical casing: UPPERCASE(token) → canonical form.
_PRESERVED_TOKENS: frozenset[str] = frozenset({
    # Technical / platform acronyms
    "GPA", "CGPA", "ERP", "API", "PDF", "URL", "USB",
    "LMS", "CMS", "VPN", "DNS", "IP", "ID", "IDs", "ATM",
    "CPU", "GPU", "RAM", "ROM", "OS", "HTML", "CSS", "SQL",
    "AWS", "JSON", "REST", "SSH", "HTTP", "HTTPS", "LLM", "RRF", "RAG",
    # Known proper nouns / platforms
    "GitHub", "GitLab", "LinkedIn", "YouTube", "Bitbucket",
    "Supabase", "Postgres", "PostgreSQL",
    "Python", "Java", "Linux", "Unix", "Windows", "C++", "C#", ".NET", "Node.js",
    "Einstein", "Newton", "Turing", "OpenAI",
    # Academic degree abbreviations
    "MBA", "BBA", "MCA", "BCA", "BTech", "MTech", "PhD", "MSc", "BSc", "BE", "ME",
    # Common university department / discipline codes
    "CSE", "ECE", "EEE", "MECH", "CIVIL", "IT", "BIO", "CHEM", "PHY", "MATH",
    "NLP", "ML", "AI", "DL",
    # Governance / ambiguous acronyms (preserved verbatim)
    "HOD", "UGC", "NAAC", "AICTE", "NIRF", "IQAC", "COE", "IEEE", "ACM",
    "SOP", "UG", "PG", "FT", "BOD", "AC", "NSS", "CC", "IPR", "CS",
    "IIIT", "LAS",
})

_PRESERVED_BY_UPPER: dict[str, str] = {t.upper(): t for t in _PRESERVED_TOKENS}

# Common English and domain vocabulary. These tokens represent standard words
# that should be rendered in lowercase in sentence-case unless starting a sentence.
_COMMON_WORDS: frozenset[str] = frozenset({
    # Common functional words
    "a", "about", "above", "access", "according", "after", "again", "against",
    "all", "almost", "also", "although", "always", "am", "among", "an", "and",
    "another", "any", "anyone", "anybody", "anything", "apply", "are", "aren't",
    "as", "ask", "at", "be", "because", "been", "before", "being", "below",
    "beneath", "between", "both", "but", "by", "can", "can't", "cannot",
    "card", "cards", "change", "check", "class", "classes", "code", "could",
    "couldn't", "criteria", "date", "dates", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "download", "during", "each", "either",
    "email", "enroll", "enrollment", "even", "every", "everyone", "everybody",
    "everything", "fail", "failed", "few", "find", "first", "for", "form",
    "forms", "from", "get", "give", "good", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if",
    "in", "into", "is", "isn't", "it", "it's", "its", "itself", "just", "know",
    "last", "leave", "look", "make", "many", "marks", "may", "me", "might", "more",
    "most", "much", "must", "my", "myself", "neither", "new", "next", "no",
    "nobody", "none", "noone", "nor", "not", "now", "of", "off", "often", "old",
    "on", "once", "one", "only", "or", "other", "others", "our", "ours",
    "ourselves", "out", "over", "own", "per", "please", "portal", "previous",
    "prior", "regarding", "related", "repeat", "retake", "same", "say", "see",
    "shall", "she", "should", "shouldn't", "show", "since", "so", "some",
    "somebody", "someone", "something", "sometimes", "still", "such", "take",
    "tell", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "time", "to", "too", "under", "until", "up",
    "upload", "us", "use", "used", "using", "usually", "very", "view", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
    "what", "what's", "when", "where", "where's", "which", "while", "who",
    "who's", "whom", "whose", "why", "will", "with", "within", "without",
    "won't", "would", "wouldn't", "year", "years", "yes", "yet", "you",
    "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves",
    # Domain concepts and academic vocabulary
    "academic", "academics", "accommodation", "achievement", "activity",
    "address", "admission", "admissions", "advisor", "affairs", "aid",
    "appeal", "application", "approval", "assignment", "assignments",
    "attendance", "book", "books", "branch", "calendar", "campus",
    "certificate", "college", "committee", "complaint", "complete", "completed",
    "completing", "completion", "conduct",
    "coordinator", "course", "courses", "credit", "credits", "curricula",
    "curriculum", "deadline", "deadlines", "degree", "department",
    "departments", "director", "discipline", "dissertation", "document",
    "documents", "due", "dues", "eligibility", "exam", "examination",
    "examinations", "exams", "faculty", "fee", "fees", "field", "file",
    "final", "fine", "fines", "gender", "grade", "grades", "grading",
    "graduate", "graduation", "grievance", "guidelines", "hall", "homework",
    "hospital", "hostel", "hostels", "information", "instructor", "instructors",
    "insurance", "integrity", "internship", "lab", "laboratory", "labs",
    "lecture", "lectures", "library", "libraries", "loan", "login", "mail",
    "manual", "mark", "marks", "medical", "meeting", "midsem", "endsem",
    "name", "network", "number", "office", "paper", "papers", "pass",
    "payment", "permission", "placement", "policy", "policies", "procedure",
    "procedures", "process", "processes", "professor", "professors",
    "program", "programs", "project", "projects", "protocol", "protocols",
    "provisional", "quiz", "quizzes", "refund", "registration", "regulation",
    "regulations", "relativity", "report", "requirement", "requirements",
    "research", "result", "results", "review", "role", "roles", "room",
    "rooms", "rule", "rules", "schedule", "scholarship", "scholarships",
    "school", "score", "scores", "section", "semester", "semesters",
    "session", "staff", "status", "stipend", "student", "students",
    "subject", "subjects", "submission", "submit", "syllabus", "system",
    "systems", "teacher", "teachers", "test", "tests", "theory", "thesis",
    "time", "timetable", "timing", "timings", "transcript", "transformer",
    "tuition", "undergraduate", "university", "vacation", "verification",
    "verify", "vs", "welfare", "wifi", "withdrawal",
})

# Course code pattern: 2-5 alphabetic department code optionally hyphenated/underscored with 2-4 digits.
_COURSE_CODE_RE = re.compile(r"^[A-Za-z]{2,5}[-_]?\d{2,4}[A-Za-z]?$")
_EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}$")
_ORDINAL_RE = re.compile(r"^\d+(?:st|nd|rd|th)$", re.IGNORECASE)

_LEET_MAP: dict[str, list[str]] = {
    "0": ["o"],
    "1": ["i", "l"],
    "3": ["e"],
    "4": ["a"],
    "5": ["s"],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_str(query: Any) -> str:
    """Internal helper that guarantees the input is a ``str``."""
    if not isinstance(query, str):
        raise TypeError("normalize_query expects a string input")
    return query


def _collapse_repeated_letters(word: str) -> str:
    """Normalize elongated words caused by repeated keystrokes (e.g. 'HOWWWW' -> 'HOW')."""
    if len(word) < 4 or not re.search(r"([A-Za-z])\1{2,}", word):
        return word

    low = word.lower()
    # Try collapsing 3+ repeats to 1 letter
    c1 = re.sub(r"([A-Za-z])\1{2,}", r"\1", low)
    if c1 in _COMMON_WORDS or c1 in _TYPO_MAP or c1 in _ABBREV_MAP:
        if word.isupper():
            return c1.upper()
        if word[0].isupper():
            return c1.capitalize()
        return c1

    # Try collapsing 3+ repeats to 2 letters (e.g. 'seee' -> 'see')
    c2 = re.sub(r"([A-Za-z])\1{2,}", r"\1\1", low)
    if c2 in _COMMON_WORDS or c2 in _TYPO_MAP or c2 in _ABBREV_MAP:
        if word.isupper():
            return c2.upper()
        if word[0].isupper():
            return c2.capitalize()
        return c2

    if word.isupper():
        return c1.upper()
    return c1


def _clean_punctuation(s: str) -> str:
    """Collapse repeated and mixed punctuation, and remove space before punctuation."""
    # Remove space before punctuation only when followed by whitespace or string boundary
    s = re.sub(r"\s+([.,;:!?]+)(?=\s|$)", r"\1", s)

    # Strip repeated/redundant enclosing brackets: ((grades)) -> grades
    s = re.sub(r"\({2,}(.*?)\){2,}", r"\1", s)
    s = re.sub(r"\[{2,}(.*?)\]{2,}", r"\1", s)

    # Collapse mixed question/exclamation: ?!!! or !??? or ?! -> ?
    s = re.sub(r"\?[!?]+|[!]+\?+[!?]*", "?", s)

    # Deduplicate repeated identical marks
    s = re.sub(r"\?{2,}", "?", s)
    s = re.sub(r"!{2,}", "!", s)
    s = re.sub(r",{2,}", ",", s)
    s = re.sub(r";{2,}", ";", s)
    s = re.sub(r":{2,}", ":", s)

    # If the sentence is a question starting with a question word, and ends with trailing '...'
    if re.search(r"^(how|what|where|when|why|who|which|can|could|is|are|do|does)\b", s, re.IGNORECASE):
        s = re.sub(r"\.{2,}\s*$", "?", s)
    else:
        s = re.sub(r"\.{4,}", "...", s)

    return s


def _clean_token_noise(token: str) -> str:
    """Strip accidental surrounding noise from tokens while protecting valid entities."""
    if token in _PRESERVED_TOKENS or token.upper() in _PRESERVED_BY_UPPER:
        return token
    if _EMAIL_RE.match(token):
        return token
    if _COURSE_CODE_RE.match(token):
        return token
    if any(c.isdigit() for c in token) and ("-" in token or "@" in token) and token.replace("-", "").replace("@", "").isalnum():
        return token
    if token.startswith("#") and token[1:].upper() in _PRESERVED_BY_UPPER:
        return "#" + _PRESERVED_BY_UPPER[token[1:].upper()]

    # Separate trailing question/punctuation first: e.g. [courses]? -> [courses] + ?
    trail = re.search(r"([.,;:!?]+)$", token)
    suffix = trail.group(1) if trail else ""
    t = token[:-len(suffix)] if suffix else token

    # Strip repeated enclosing brackets: ((grades)) -> grades
    t = re.sub(r"^[\(\[\{]{2,}(.*?)[\)\]\}]{2,}$", r"\1", t)

    # Strip single enclosing brackets if surrounding an ordinary word: [grades] -> grades
    m_bracket = re.match(r"^([\[\{\(])(.+?)([\]\}\)])$", t)
    if m_bracket:
        inner = m_bracket.group(2)
        if inner.lower() in _COMMON_WORDS or inner.lower() in _TYPO_MAP or inner.lower() in _ABBREV_MAP:
            t = inner

    # Trailing noise: @ or # (unless C#)
    if t.endswith("@"):
        t = re.sub(r"@+$", "", t)
    if t.endswith("#") and t.upper() != "C#":
        t = re.sub(r"#+$", "", t)

    # Leading noise:
    if t.startswith("#"):
        rest = t[1:]
        if rest.upper() not in _PRESERVED_BY_UPPER:
            t = rest

    if t.startswith("@"):
        rest = t[1:]
        if rest.lower() in _COMMON_WORDS or rest.lower() in _TYPO_MAP or rest.lower() in _ABBREV_MAP:
            t = rest

    # Trailing excessive exclamation noise: grades!!! -> grades
    if re.search(r"[a-zA-Z]!{2,}$", t):
        t = re.sub(r"!+$", "", t)

    return t + suffix


def _normalize_leetspeak(core: str) -> str:
    """Normalize unambiguous leetspeak character substitutions in ordinary vocabulary."""
    if core.isdigit() or _ORDINAL_RE.match(core):
        return core
    if not any(ch in "01345" for ch in core):
        return core
    if _COURSE_CODE_RE.match(core):
        return core
    if any(ch.isdigit() for ch in core) and ("-" in core or "@" in core or "/" in core):
        return core
    digit_indices = [i for i, ch in enumerate(core) if ch in _LEET_MAP]
    if len(digit_indices) > 3 or len(core) < 3:
        return core
    if not any(c.isalpha() for c in core):
        return core

    import itertools
    replacement_options = [_LEET_MAP[core[i]] for i in digit_indices]
    candidates = []
    for combo in itertools.product(*replacement_options):
        cand_list = list(core)
        for idx, repl in zip(digit_indices, combo):
            cand_list[idx] = repl
        cand_word = "".join(cand_list).lower()
        if cand_word in _COMMON_WORDS or cand_word in _TYPO_MAP or cand_word in _ABBREV_MAP:
            candidates.append(cand_word)

    if len(candidates) == 1:
        cand_word = candidates[0]
        if core[0].isupper() and (len(core) == 1 or not core[1:].isupper()):
            return cand_word.capitalize()
        elif core.isupper():
            return cand_word.upper()
        else:
            return cand_word
    return core


def _split_token(token: str, query_has_balanced_paren: bool) -> tuple[str, str, str, str]:
    """Split token into (prefix, core, possessive, suffix), cleaning stray punctuation.

    Examples
    --------
    ")To"          → ("", "To", "", "")
    ",check"       → ("", "check", "", "")
    "rules???"     → ("", "rules", "", "?")
    "UNIVERSITY'S" → ("", "UNIVERSITY", "'s", "")
    "(CSE)"        → ("(", "CSE", "", ")")
    "Einstein's."  → ("", "Einstein", "'s", ".")
    """
    token = _clean_token_noise(token)

    # Strip malformed leading punctuation: closing brackets ')]}', or separators ',;:!?/\|~^*+=<>'
    token = re.sub(r"^[)\]},;:!?/\\|~^*+=<>]+", "", token)
    if not query_has_balanced_paren:
        token = re.sub(r"^[(\[{]+", "", token)

    lead = re.match(r"^([(\"'\`]+)", token)
    prefix = lead.group(1) if lead else ""
    token_no_lead = token[len(prefix):]

    trail = re.search(r"([.,;:!?)\"'\`\]}]+)$", token_no_lead)
    suffix = trail.group(1) if trail else ""
    core_with_poss = token_no_lead[:-len(suffix)] if suffix else token_no_lead

    possessive = ""
    if core_with_poss.lower().endswith(("'s", "’s")) and len(core_with_poss) > 2:
        possessive = "'s"
        core = core_with_poss[:-2]
    else:
        core = core_with_poss

    if not query_has_balanced_paren:
        suffix = re.sub(r"[)\]}]+", "", suffix)
    suffix = _clean_punctuation(suffix)

    return prefix, core, possessive, suffix


def _classify_core(
    core: str,
    is_first: bool,
    is_noisy: bool,
    spell_corrector: Optional["CorpusSpellCorrector"] = None,
) -> str:
    """Return the normalised canonical form of a single token core.

    Evaluated in priority order:
    1. Keystroke elongation normalization.
    2. Typo correction (via curated mapping).
    3. Abbreviation expansion (via curated mapping).
    4. Standalone pronoun "I" and contractions.
    5. Common English words: if token is a standard word (like 'me', 'it', 'be', 'how', 'check')
       only treat as acronym if explicitly uppercase in a non-noisy query.
    6. Known preserved acronyms and entities (GPA, CSE, GitHub, Einstein, etc.).
    7. Course codes (e.g. CSE301, CSE-301, CS101, math201).
    8. Slash-separated entities (e.g. AI/ML, 2025/26).
    9. Legitimate CamelCase identifiers (e.g. GitHub, GitLab, YouTube, DevOps).
    10. Digits / Numbers / Years.
    11. Document ID patterns (e.g. ACAD-REG-001, CSE-CRS-304).
    12. Preserved unknown acronyms in normal-cased sentences (e.g. 'ABCD' in 'Where is ABCD portal?').
    13. Noisy casing: if query was typed in all-caps or erratic casing, convert unfamiliar words to lowercase.
    14. In normal-cased query, a Title-Case word mid-sentence is likely a proper noun.
    15. Default → lowercase.
    """
    if not core:
        return core

    upper = core.upper()

    # Protect preserved entities and acronyms from repeated-letter elongation collapse
    if upper not in _PRESERVED_BY_UPPER:
        core = _collapse_repeated_letters(core)
    core = _normalize_leetspeak(core)
    low = core.lower()
    upper = core.upper()

    # 0. Preserved hashtag entities (e.g. #CSE)
    if core.startswith("#") and core[1:].upper() in _PRESERVED_BY_UPPER:
        return "#" + _PRESERVED_BY_UPPER[core[1:].upper()]

    # 1. Typo correction (protect uppercase preserved acronyms like LAS)
    if low in _TYPO_MAP:
        if not (core.isupper() and upper in _PRESERVED_BY_UPPER):
            low = _TYPO_MAP[low]
            upper = low.upper()
            core = low

    # 2. Abbreviation expansion
    if low in _ABBREV_MAP:
        low = _ABBREV_MAP[low]
        upper = low.upper()
        core = low

    # 2b. Corpus spelling correction for unknown ordinary words. Acronyms and
    # mid-sentence Title-Case words (likely names) are left alone.
    looks_ordinary = core.islower() or is_noisy or (is_first and core[1:].islower())
    if (
        spell_corrector is not None
        and looks_ordinary
        and core.isalpha()
        and low not in _COMMON_WORDS
        and upper not in _PRESERVED_BY_UPPER
    ):
        corrected = spell_corrector.correct(low)
        if corrected:
            low = corrected
            upper = low.upper()
            core = low

    # 3. Pronoun "I" and contractions
    if low == "i":
        return "I"
    if low in ("i'm", "i've", "i'll", "i'd"):
        return "I" + low[1:]

    # 4. Common English words: if token is a standard word (like 'me', 'it', 'be', 'how', 'check')
    # only treat as acronym if explicitly uppercase in a non-noisy query
    if low in _COMMON_WORDS:
        if core.isupper() and not is_noisy and upper in _PRESERVED_BY_UPPER:
            return _PRESERVED_BY_UPPER[upper]
        return low

    # 5. Known preserved acronyms / entities
    if upper in _PRESERVED_BY_UPPER:
        return _PRESERVED_BY_UPPER[upper]

    # 6. Course codes (e.g. CSE301, CSE-301, CS101, math201, Cse301)
    if _COURSE_CODE_RE.match(core):
        return upper

    # 7. Slash-separated entities (e.g. AI/ML, 2025/26)
    if "/" in core:
        parts = core.split("/")
        classified_parts = [
            _classify_core(p, is_first=False, is_noisy=is_noisy, spell_corrector=spell_corrector) for p in parts
        ]
        return "/".join(classified_parts)

    # 8. Known / legitimate CamelCase identifiers (e.g. GitHub, GitLab, YouTube, DevOps)
    if re.search(r"[a-z][A-Z]", core) and len(core) >= 3:
        return core

    # 9. Digits / Numbers / Years
    if core.isdigit():
        return core

    # 10. Document ID patterns (e.g. ACAD-REG-001, CSE-CRS-304, ABC-123@2025)
    if any(c.isdigit() for c in core) and ("-" in core or "@" in core) and core.replace("-", "").replace("@", "").isalnum():
        return upper

    # 10b. Preserved uppercase alphanumeric identifiers / codes (e.g. A1, R2D2, B2B, ABC123, ABCXYZ123)
    if not is_noisy and core.isupper() and any(c.isdigit() for c in core) and any(c.isalpha() for c in core) and core.isalnum():
        return core

    # 11. Preserved unknown acronyms in normal-cased sentences (e.g. 'ABCD' in 'Where is ABCD portal?')
    if not is_noisy and core.isupper() and 2 <= len(core) <= 6 and core.isalpha():
        return core

    # 12. Noisy casing: if query was typed in all-caps or erratic casing,
    # convert unfamiliar words to lowercase.
    if is_noisy:
        return low

    # 13. In normal-cased query, a Title-Case word mid-sentence is likely a proper noun.
    if not is_first and len(core) >= 2 and core[0].isupper() and core[1:].islower():
        return core

    # 14. Default
    return low



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_query(query: Any, spell_corrector: Optional["CorpusSpellCorrector"] = None) -> str:
    """Return a cleaned, canonical version of *query*.

    The function applies a deterministic, conservative set of normalisation
    transformations. It does **not** perform semantic query reformulation –
    that is handled by :class:`reformulation.reformulator.QueryReformulator`.

    Steps performed
    ~~~~~~~~~~~~~~~
    1. Type guard (raises ``TypeError`` for non-string input).
    2. Strip leading/trailing whitespace and collapse internal whitespace runs.
    3. Punctuation cleanup: deduplicate marks, clean spacing, normalize trailing '...'.
    4. Token decomposition: strip stray brackets/separators, preserve balanced brackets,
       isolate possessives ('s) and contractions.
    5. Keystroke elongation cleanup: collapse 3+ repeated letters.
    6. Correct unambiguous typos and expand safe domain abbreviations.
    7. Sentence-case: restore known acronyms, proper nouns, course codes, and pronoun "I".
    8. Capitalise the first alphabetical character (unless starting with a mixed-case identifier).
    9. Final whitespace-collapse + punctuation cleanup pass.

    Parameters
    ----------
    query:
        The raw user input string.
    spell_corrector:
        Optional corpus spelling corrector; when given, unknown ordinary words are
        corrected to words of the indexed corpus.

    Returns
    -------
    str
        The normalised query string.

    Raises
    ------
    TypeError
        If *query* is not a ``str``.

    Notes
    -----
    * The function is **deterministic**, **idempotent**, and **side-effect free**.
    * Ambiguous tokens (e.g. "CS", "AI", "ABCD", "xyz") are preserved verbatim.
    * The original query must be stored separately by the caller.
    """
    s = _ensure_str(query)

    # Step 1: Strip and collapse internal whitespace
    s = re.sub(r"[\s]+", " ", s.strip())
    if not s:
        return ""

    # Step 2: Global punctuation cleanup
    s = _clean_punctuation(s)

    tokens = s.split()
    if not tokens:
        return ""

    # Check balanced parentheses in query
    open_paren = s.count("(")
    close_paren = s.count(")")
    has_balanced_paren = (open_paren > 0 and open_paren == close_paren)

    # Detect noisy casing: all-caps or >= 50% uppercase words
    alpha_tokens = [re.sub(r"[^A-Za-z]", "", t) for t in tokens if re.search(r"[A-Za-z]", t)]
    if not alpha_tokens:
        is_noisy = False
    else:
        all_caps_count = sum(1 for t in alpha_tokens if len(t) >= 2 and t.isupper())
        is_all_caps = all(t.isupper() for t in alpha_tokens if len(t) >= 2)
        is_noisy = is_all_caps or (len(alpha_tokens) > 2 and all_caps_count / len(alpha_tokens) > 0.5)

    result_tokens: list[str] = []
    for idx, token in enumerate(tokens):
        prefix, core, possessive, suffix = _split_token(token, has_balanced_paren)
        norm_core = _classify_core(core, is_first=(idx == 0), is_noisy=is_noisy, spell_corrector=spell_corrector)
        if norm_core:
            result_tokens.append(prefix + norm_core + possessive + suffix)

    joined = " ".join(result_tokens)

    # Capitalise first alphabetical character of the first word, unless it is a mixed-case identifier
    # or the query begins with a numeric, ordinal, or currency expression (e.g. 2nd, 75, 10, ₹50,000)
    first_token = result_tokens[0] if result_tokens else ""
    if re.match(r"^[(\"'\`\[\{]*[a-zA-Z]", first_token) and not re.match(r"^[(\"'\`\[\{]*[a-z]+[A-Z]", first_token):
        def _cap_first(m: re.Match) -> str:
            return m.group(0).upper()
        joined = re.sub(r"^[(\"'\`\[\{]*[a-zA-Z]", _cap_first, joined, count=1)

    # Final cleanup pass
    joined = re.sub(r"[\s]+", " ", joined.strip())
    joined = _clean_punctuation(joined)

    return joined

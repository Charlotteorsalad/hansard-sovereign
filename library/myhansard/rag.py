import json
import os
import random
import re
import time
from collections.abc import Generator

import requests
from lingua import Language, LanguageDetectorBuilder

import myhansard

# Where Ollama lives. Defaults to localhost; set OLLAMA_BASE_URL to point a
# containerised backend at Ollama running on the Docker host
# (e.g. http://host.docker.internal:11434).
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Generation model. Default is the 8B (strongest content grounding); set
# RAG_MODEL=hansard-qwen to serve the QLoRA fine-tune instead — a fast,
# fully-GPU-resident 1.5B that emits the citation format natively.
DEFAULT_MODEL = os.environ.get("RAG_MODEL", "llama3.1:8b-instruct-q4_K_M")

# Built once at import; loading the language models is expensive. English vs Malay only.
_DETECTOR = (
    LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.MALAY)
    .build()
)


def _detect_lang(query: str) -> str:
    lang = _DETECTOR.detect_language_of(query)
    return "Bahasa Malaysia" if lang == Language.MALAY else "English"


# Function words dropped from the keyword pass: a Malay "yang"/"dalam" or an
# English "the"/"what" appears in nearly every speech, so LIKE-matching them
# just returns noise. Filtering them sharpens keyword precision in BOTH
# languages (this also removes the "say" -> "saya" substring false-match).
_STOP = {
    # English
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "from", "that", "this", "these", "those", "is", "are", "was", "were", "be",
    "did", "do", "does", "what", "who", "how", "why", "when", "where", "any",
    "about", "say", "said", "says", "raised", "raise", "discussed", "discuss",
    "issue", "issues", "member", "members", "parliament", "government", "there",
    # Malay
    "apa", "apakah", "yang", "tentang", "mengenai", "dalam", "untuk", "adakah",
    "oleh", "itu", "ini", "dan", "atau", "di", "ke", "dari", "pada", "dengan",
    "telah", "akan", "ada", "kah", "sebarang", "isu", "isu-isu", "ahli",
    "parlimen", "kerajaan", "berkenaan", "berkaitan", "dibangkitkan",
}

# The corpus is Malay, so an English question's keywords ("subsidies") never
# LIKE-match the source text. Map common English topic terms to their Malay
# equivalents so the keyword half of hybrid retrieval fires for English queries
# too. (The dense bge-m3 half is already cross-lingual; this restores the other
# half.) Multi-word phrases are matched as substrings before tokenising.
_EN_PHRASES = {
    "cost of living": ["sara hidup", "kos sara hidup"],
    "flood mitigation": ["tebatan banjir"],
    "public transport": ["pengangkutan awam"],
    "affordable housing": ["perumahan mampu milik"],
    "national security": ["keselamatan negara"],
    "clean water": ["air bersih"],
}
_EN_TO_MS = {
    "subsidy": ["subsidi"], "subsidies": ["subsidi"],
    "fuel": ["minyak", "petrol", "diesel"], "petrol": ["petrol"],
    "education": ["pendidikan", "pelajaran"], "school": ["sekolah"],
    "schools": ["sekolah"], "student": ["pelajar"], "students": ["pelajar"],
    "health": ["kesihatan"], "healthcare": ["kesihatan"],
    "hospital": ["hospital"], "hospitals": ["hospital"],
    "flood": ["banjir"], "floods": ["banjir"],
    "transport": ["pengangkutan"], "transportation": ["pengangkutan"],
    "housing": ["perumahan"], "house": ["rumah"], "homes": ["rumah"],
    "crime": ["jenayah"], "corruption": ["rasuah"],
    "tax": ["cukai"], "taxes": ["cukai"], "taxation": ["cukai"],
    "employment": ["pekerjaan"], "job": ["pekerjaan"], "jobs": ["pekerjaan"],
    "unemployment": ["pengangguran"],
    "digital": ["digital"], "digitalisation": ["pendigitalan"],
    "digitalization": ["pendigitalan"],
    "agriculture": ["pertanian"], "farmer": ["petani"], "farmers": ["petani"],
    "environment": ["alam", "sekitar", "persekitaran"],
    "security": ["keselamatan"], "economy": ["ekonomi"], "economic": ["ekonomi"],
    "water": ["air"], "wage": ["gaji", "upah"], "wages": ["gaji", "upah"],
    "salary": ["gaji"], "development": ["pembangunan"], "minister": ["menteri"],
    "budget": ["belanjawan", "bajet"], "poverty": ["kemiskinan"],
    "welfare": ["kebajikan"], "infrastructure": ["infrastruktur"],
    "road": ["jalan"], "roads": ["jalan"], "children": ["kanak-kanak"],
    "women": ["wanita"], "youth": ["belia"], "religion": ["agama"],
    "language": ["bahasa"], "flooding": ["banjir"], "subsidised": ["subsidi"],
}


def _keyword_terms(query: str) -> list[str]:
    """Content words for the keyword half of retrieval.

    Drops stopwords/very short tokens, and — because the corpus is Malay — maps
    English topic terms (words and a few phrases) to their Malay equivalents so
    an English question's keyword pass still matches the source text.
    """
    ql = query.lower()
    terms: list[str] = []

    english = _detect_lang(query) == "English"
    if english:
        for phrase, mapped in _EN_PHRASES.items():
            if phrase in ql:
                terms.extend(mapped)

    words = [w.strip(".,;:!?'\"()[]") for w in ql.split()]
    for w in words:
        if len(w) > 2 and w not in _STOP:
            terms.append(w)
            if english:
                terms.extend(_EN_TO_MS.get(w, []))

    return list(dict.fromkeys(terms))  # preserve order, drop duplicates


# Intro templates. Generated in Python so the wording and language are reliable
# rather than left to the model (small models drift on both).
#
# They must be COUNT-NEUTRAL: when streaming, the intro is emitted before the
# body exists, so we can't know whether the answer will hold one item or eight.
# Nothing here may claim "several"/"beberapa" or force a plural ("isu-isu").
# Malay doesn't mark plural obligatorily, so dropping those makes it neutral.
#
# There is deliberately NO conclusion template. A canned closing line ("these
# issues reflect the urgent need to address infrastructure...") is an editorial
# claim with no source behind it, and it was appended to every answer regardless
# of topic — directly at odds with a system whose answers are grounded in [n]
# citations. The cited list is the answer; it ends where the evidence ends.
_INTROS_MS = [
    "Berikut adalah perkara yang dibangkitkan dalam sidang Dewan Rakyat:",
    "Berdasarkan rekod Hansard, berikut yang dibangkitkan dalam Parlimen:",
    "Berikut adalah ringkasan daripada sidang Dewan Rakyat:",
    "Daripada rekod sidang Dewan Rakyat:",
    "Berikut perkara yang dibincangkan dalam Parlimen:",
]

_INTROS_EN = [
    "Here is what was raised in the Dewan Rakyat, based on the Hansard record:",
    "The following was raised during the Dewan Rakyat sitting:",
    "From the Hansard record of the Dewan Rakyat:",
    "Here is a summary of what was discussed in Parliament:",
    "Based on the parliamentary record:",
]


def _random_intro(lang: str) -> str:
    return random.choice(_INTROS_EN if lang == "English" else _INTROS_MS)


# Speaker parsing
_HONORIFIC_RE = re.compile(
    r"\b(bin|binti|bt\.?|Dato'?|Dato['’ʼ]|Datuk|Datin|Tan Sri|Tun|Dr|Tuan|"
    r"Puan|Haji|Hajah|YB|YAB)\b",
    re.IGNORECASE,
)
_ROLE_RE = re.compile(r"\bMenteri\b|\bPengerusi\b|Yang di-Pertua", re.IGNORECASE)


def _format_speaker(speaker_raw: str) -> str:
    """Turn "Tuan Khoo Poay Tiong [Kota Melaka]" into the "(Kota Melaka)" form.

    Handles role-prefixed and number-prefixed variants. Mirrors the frontend's
    parseSource() so context names match the source cards.
    """
    clean = re.sub(r"\s+", " ", speaker_raw.replace("\n", " ")).strip()
    clean = re.sub(r"^\d+\.\s*", "", clean)  # drop leading "2. "

    m = re.match(r"^(.*?)\s*\[([^\]]+)\]", clean)
    if not m:
        return clean

    outer, inner = m.group(1).strip(), m.group(2).strip()
    # strip motion text that sometimes leaks into the bracket
    inner = re.sub(r"\s+(minta|menyatakan|soalan).*$", "", inner)

    if _HONORIFIC_RE.search(inner) or _ROLE_RE.search(outer):
        name, constituency = inner, outer
    else:
        name, constituency = outer, inner

    return f"{name} ({constituency})" if constituency else name


# Post-processing
_LEADING_TITLE_RE = re.compile(
    r"^(?:(?:Tuan|Puan|Dato['’ʼ]?|Datuk|Datin|Tan Sri|Tun|Dr|Haji|Hajah|"
    r"YB|YAB)\.?\s+)+",
    re.IGNORECASE,
)


def _fix_constituency(text: str) -> str:
    # "MP name dari Constituency verb..." → "MP name (Constituency) verb..."
    return re.sub(
        r" dari ((?:[A-Z][a-zA-Z']*(?:\s+(?=[A-Z]))?)+)",
        lambda m: f" ({m.group(1).strip()})",
        text,
    )


def _normalize_speaker_names(text: str, speeches: list) -> str:
    """Replace the model's loose '(Ahli Parlimen) <name>' with the canonical
    'Honorific Name (Constituency)' parsed from speaker_raw.
    """
    entries = []
    for s in speeches:
        formatted = _format_speaker(s[1])  # "Tuan Khoo Poay Tiong (Kota Melaka)"
        name_part = re.sub(r"\s*\(.*?\)\s*$", "", formatted).strip()
        core = _LEADING_TITLE_RE.sub("", name_part).strip()  # "Khoo Poay Tiong"
        if core and "(" in formatted:
            entries.append((core, formatted))

    # longest core first so a shorter name can't partially match
    entries.sort(key=lambda e: len(e[0]), reverse=True)

    for core, formatted in entries:
        pattern = re.compile(
            r"(?:Ahli Parlimen\s+)?"
            r"(?:(?:Tuan|Puan|Dato['’ʼ]?|Datuk|Datin|Tan Sri|Tun|Dr|Haji|"
            r"Hajah|YB|YAB)\.?\s+)*"
            + re.escape(core)
            + r"(?!\s*\()",  # skip if already followed by "(...)"
        )
        text = pattern.sub(formatted, text)

    return text


# Matches "An MP from X", "A Member of Parliament (MP) from X", "Ahli Parlimen dari X".
# When the model drops the name and keeps only the constituency, we look the name up.
_ANON_MP_RE = re.compile(
    r"\b(?:A|An|Another|The|Seorang)\s+"
    r"(?:Members?\s+of\s+Parliament|MPs?|Ahli\s+Parlimen)"
    r"(?:\s*\(MP\))?\s+(?:from|for|dari|bagi)\s+"
    r"(?P<const>[A-Z][\w']*(?:\s+[A-Z][\w']*)*)",
)


_OPENER_RE = re.compile(r"^(?:yes|ya)\b", re.IGNORECASE)


def _is_boilerplate_opener(line: str) -> bool:
    """A short generic intro line that duplicates the Python intro."""
    return line.endswith(":") or (bool(_OPENER_RE.match(line)) and len(line) < 100)


def _fix_anonymous_mp(text: str, speeches: list) -> str:
    by_const = {}
    for s in speeches:
        formatted = _format_speaker(s[1])
        m = re.search(r"\(([^)]+)\)\s*$", formatted)
        if m:
            by_const[m.group(1).strip().lower()] = formatted

    def repl(match: re.Match) -> str:
        return by_const.get(match.group("const").strip().lower(), match.group(0))

    return _ANON_MP_RE.sub(repl, text)


def _auto_cite(text: str, speeches: list) -> str:
    stop = {
        "the", "a", "an", "in", "of", "to", "and", "is", "was", "are", "were",
        "for", "that", "from", "said", "about", "by", "at", "their", "have",
        "has", "had", "been", "on", "as", "with", "this", "it", "he", "she",
        "they", "we", "i", "be", "not", "but", "or", "if", "its", "also",
    }

    src_word_sets = []
    for idx, s in enumerate(speeches, 1):
        words = (
            set(re.sub(r"[^\w\s]", "", f"{s[1]} {s[2]}").lower().split()) - stop
        )
        src_word_sets.append((idx, words))

    def _cite(segment: str) -> str:
        segment = segment.strip()
        if not segment:
            return segment
        # Never trust the model's own [n] — a weaker model mislabels which source
        # a line refers to. Strip whatever it wrote and re-derive the citation
        # from content (speaker + summary vs the retrieved speech), so [n] always
        # points at the source the line actually describes.
        segment = re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", segment)
        segment = re.sub(r"\s*\[\d+\s*$", "", segment)  # dangling "[8" (no close)
        segment = re.sub(r"\s+([.!?,;:])", r"\1", segment).strip()
        if not segment:
            return segment
        seg_words = set(re.sub(r"[^\w\s]", "", segment).lower().split()) - stop
        best_idx, best_score = None, 0
        for idx, src_words in src_word_sets:
            score = len(seg_words & src_words)
            if score > best_score and score >= 3:
                best_score, best_idx = score, idx
        if not best_idx:
            return segment
        if segment[-1] in ".!?":
            return f"{segment[:-1]} [{best_idx}]{segment[-1]}"
        return f"{segment} [{best_idx}]"

    text = _normalize_speaker_names(text, speeches)
    text = _fix_anonymous_mp(text, speeches)

    lines = [_fix_constituency(line.strip()) for line in text.split("\n")]
    lines = [line for line in lines if line]

    # A list item is "1." numbered or a "-"/"*"/"•" bullet. Strip the marker and
    # renumber. Bullets are a common model variation, so accepting them avoids
    # throwing away an otherwise-good answer.
    item_re = re.compile(r"^(?:\d+[.)]|[-*•])\s+")
    items = [item_re.sub("", line) for line in lines if item_re.match(line)]

    if items:
        result_lines = [
            f"{i}. {_cite(it)}" for i, it in enumerate(items, 1) if it.strip()
        ]
    else:
        # No list: the model answered in prose. Keep and cite the substantive
        # sentences, but drop Q&A scaffolding ("Q1:"/"A1:"), the one prose shape
        # we never want to surface.
        qa_re = re.compile(r"^[QA]\d*\s*[:.]", re.IGNORECASE)
        kept = [ln for ln in lines if not qa_re.match(ln)]
        # Drop a leading generic opener that just duplicates our Python intro.
        # Only short/connector lines; a long opener carries real content.
        while kept and _is_boilerplate_opener(kept[0]):
            kept.pop(0)
        result_lines = [_cite(ln) for ln in kept]

    return "\n".join(result_lines)


def _retrieve(query: str, collection, conn) -> list:
    """Hybrid search: vector + keyword. Returns deduplicated speech rows."""
    results = myhansard.query_speeches(collection, query)
    ids = [m["id"] for m in results["metadatas"][0]]

    cursor = conn.cursor()
    keywords = _keyword_terms(query)
    if keywords:
        # Bind each %keyword% as a parameter — never interpolate into the SQL,
        # or a keyword containing a quote (e.g. a name like "Dato'") both breaks
        # the statement and opens a SQL-injection hole. Rank the matches by how
        # many distinct query terms each speech contains, so the most on-topic
        # keyword hits lead (a real ranked list to fuse with the vectors).
        like_params = [f"%{k}%" for k in keywords]
        conditions = " OR ".join("content LIKE ?" for _ in keywords)
        hits_expr = " + ".join("(content LIKE ?)" for _ in keywords)
        cursor.execute(
            "SELECT id FROM speeches"
            f" WHERE ({conditions}) AND LENGTH(content) > 100"
            f" ORDER BY ({hits_expr}) DESC LIMIT 10",
            like_params + like_params,  # first N for WHERE, next N for the score
        )
        keyword_ids = [row[0] for row in cursor.fetchall()]
    else:
        keyword_ids = []

    # Reciprocal-rank fusion of the two ranked lists (dense vectors + keyword).
    # A speech both halves agree on rises to the top, and keyword-confirmed hits
    # actually enter the context — previously 10 vector hits filled every slot
    # and the keyword half was inert, so "hybrid" was vector-only in practice.
    # This is what lets an English question (whose terms are mapped to Malay in
    # _keyword_terms) pull the right Malay speeches up. Cap at 8: a small context
    # keeps prompt-eval / TTFT fast on GPU-limited hardware and the list short.
    scores: dict = {}
    for rank_list in (ids, keyword_ids):
        for rank, _id in enumerate(rank_list):
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (rank + 60)  # RRF, k=60
    ordered_ids = sorted(scores, key=lambda i: scores[i], reverse=True)[:8]
    placeholders = ",".join("?" * len(ordered_ids))
    cursor.execute(
        "SELECT id, speaker_raw, content, date, source_file, page FROM speeches"
        f" WHERE id IN ({placeholders})",
        ordered_ids,
    )
    by_id = {r[0]: r for r in cursor.fetchall() if len(r[2].strip()) > 100}
    return [by_id[i] for i in ordered_ids if i in by_id]


def _build_prompt(query: str, speeches: list) -> str:
    lang = _detect_lang(query)
    numbered_context = "\n\n".join(
        [
            f"[{idx + 1}] Speaker: {_format_speaker(s[1])}"
            f"\nDate: {s[3]}\nContent: {s[2][:600]}"  # truncate for fast eval
            for idx, s in enumerate(speeches)
        ]
    )
    if lang == "English":
        example = (
            "1. **Orang Asli Water Shortage**: Dato' Ali (Kuala Lumpur)"
            " said residents still lack clean water and urged the government"
            " to act immediately [2].\n"
            "2. **Recurring Flash Floods**: Tuan Lim (Pulau Pinang)"
            " proposed that a special fund be allocated to tackle"
            " flash floods [5]."
        )
    else:
        example = (
            "1. **Kekurangan Air Orang Asli**: Dato' Ali (Kuala Lumpur)"
            " menyatakan penduduk masih kekurangan air bersih"
            " dan meminta kerajaan bertindak segera [2].\n"
            "2. **Banjir Kilat Berulang**: Tuan Lim (Pulau Pinang)"
            " mencadangkan agar dana khas diperuntukkan"
            " bagi menangani banjir kilat [5]."
        )
    return (
        f"Summarise the following Malaysian Parliament speeches"
        f" IN {lang.upper()}.\n\n"
        f"Output ONLY a numbered list. Each line MUST be:\n"
        f"N. **3-5 word title**: Speaker (Constituency) one sentence summary [n].\n\n"
        f"Include ONLY speeches that actually answer the question. Skip the"
        f" rest — there is no need to use every speech, and three real answers"
        f" are better than eight padded ones. NEVER write an item saying a"
        f" speaker gave no view, did not comment, or is not relevant.\n\n"
        f"Base each title and summary STRICTLY on what that speech itself says."
        f" Do not borrow specific words or activities from the Question that the"
        f" speech does not itself mention — e.g. if the question asks about a"
        f" specific activity but a speech only discusses the broader topic in"
        f" general, describe it as general, not as if it addressed that specific"
        f" activity.\n\n"
        f"Example (write yours in {lang}):\n{example}\n\n"
        f"Context:\n{numbered_context}\n\n"
        f"Question: {query}\n\n"
        f"Write the answer in {lang.upper()} only:\n1."
    )


def _build_system(lang: str) -> str:
    # The Hansard corpus is predominantly Malay (with some English speeches),
    # whatever language the question is in — so state that honestly rather than
    # flipping the "source language" with the answer language.
    return (
        "You are a parliamentary research assistant that summarises Hansard "
        "debates. The source text is mainly in Malay, but you MUST write your "
        f"ENTIRE answer in {lang} only — do not use any other language "
        "(keep proper nouns such as names and places as-is). "
        "Summarise each issue in your own words, but always keep the "
        "speaker's exact name and constituency. "
        "Always cite sources with [n] bracket notation."
    )

_TEMPERATURE = 0.3


def _options() -> dict:
    # num_predict bounds the answer: the summaries are short lists, and it stops
    # a model that doesn't emit EOS cleanly (e.g. the small fine-tune) from
    # generating to the context limit.
    return {"temperature": _TEMPERATURE, "seed": random.randint(0, 99999),
            "num_predict": 512}


def _sources_payload(speeches: list) -> list:
    return [
        {
            "index": idx + 1,
            "speaker": s[1],
            "date": s[3],
            "content": s[2],
            "source_file": s[4],
            "page": s[5],
        }
        for idx, s in enumerate(speeches)
    ]


# Generation. Small models occasionally drift into Q&A or prose, so we retry.
_ITEM_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s")


def _list_items(text: str) -> list:
    return [ln for ln in text.split("\n") if _ITEM_RE.match(ln)]


def _well_formatted(text: str) -> bool:
    """A list whose items follow "**Title**: ..." (at least two bold items)."""
    return sum("**" in ln for ln in _list_items(text)) >= 2


# Safety net for padding. Given 8 retrieved speeches the model tends to emit 8
# items, inventing a non-answer ("X did not give a specific opinion on ...")
# for any speech that doesn't address the question. Such an item carries no
# information and cites a source that doesn't support it.
#
# Deliberately narrow: it matches only the "no view was expressed" summarising
# construction. It must NOT match a bare "tidak menjawab" / "did not answer",
# because an MP pressing a minister for not answering is real, citable content
# — that phrasing is left alone on purpose.
_NON_ANSWER_RE = re.compile(
    r"tidak\s+(?:mem)?beri(?:kan)?\s+(?:sebarang\s+)?(?:pandangan|komen|maklum)"
    r"|tidak\s+menyatakan\s+(?:sebarang\s+)?(?:pandangan|komen)"
    r"|tidak\s+menyentuh|tidak\s+membincangkan"
    r"|tidak\s+(?:berkaitan|relevan)\s+dengan\s+(?:soalan|persoalan)"
    r"|tidak\s+ada\s+jawapan\s+yang\s+jelas"
    r"|did\s+not\s+(?:give|provide|offer|express|state|share)\s+"
    r"(?:a\s+|an\s+|any\s+)?(?:specific\s+)?(?:opinion|view|comment|position)"
    r"|did\s+not\s+(?:specifically\s+)?(?:comment|mention|address|discuss)"
    r"|do(?:es)?\s+not\s+(?:specifically\s+)?(?:address|mention|discuss|relate)"
    r"|no\s+(?:specific\s+)?(?:opinion|view|comment)s?\s+(?:was|were|given)"
    r"|no\s+clear\s+answer"
    r"|(?:is|are)\s+not\s+relevant\s+to\s+the\s+question",
    re.IGNORECASE,
)
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s")


def _drop_non_answers(body: str) -> str:
    """Drop padded "this speaker said nothing about it" items and renumber.

    If every item looks like a non-answer the body is returned untouched —
    better to show a weak answer than an empty one.
    """
    lines = body.split("\n")
    kept = [
        ln
        for ln in lines
        if not (_ITEM_RE.match(ln) and _NON_ANSWER_RE.search(ln))
    ]
    if not any(_ITEM_RE.match(ln) for ln in kept):
        return body

    out, n = [], 0
    for ln in kept:
        if _NUMBERED_RE.match(ln):  # renumber only digits; leave bullets alone
            n += 1
            ln = _NUMBERED_RE.sub(f"{n}. ", ln)
        out.append(ln)
    return "\n".join(out).strip()


# Generic domain vocabulary that recurs in almost any Hansard sports/parliament
# answer regardless of which specific source got cited. Excluded from the
# grounding scrub below so it only checks genuinely narrow/specific query terms
# (like "larian" = running) rather than words that are expected to recur no
# matter which speech was actually cited.
_GENERIC_GROUNDING_TERMS = {
    "sukan", "program", "tahun", "acara", "penganjuran", "pandangan",
    "perkara", "soalan", "persoalan", "dana", "peruntukan", "kementerian",
    "sports", "programme", "issue", "issues", "question", "questions",
    "view", "views", "members", "member", "parliament",
}
_CITE_NUMS_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _scrub_ungrounded_terms(body: str, query: str, speeches: list) -> str:
    """Strip a query's own specific wording from a line whose cited source
    doesn't actually contain it.

    Small models anchor on the literal question: asked about "sukan larian"
    (running), a source that only discusses sports funding in general can
    still get summarised with a title like "Penganjuran Larian Sukan" —
    inventing a level of specificity the cited speech never states. Verified
    this keeps happening even with an explicit "base it strictly on the
    source, don't borrow the question's wording" prompt instruction (4/4
    repeated test runs still injected it), so it's enforced deterministically
    here instead of trusting the prompt alone.
    """
    terms = [
        t for t in dict.fromkeys(_keyword_terms(query))
        if t not in _GENERIC_GROUNDING_TERMS and len(t) > 3
    ]
    if not terms:
        return body

    out = []
    for ln in body.split("\n"):
        m = _CITE_NUMS_RE.search(ln)
        if not m:
            out.append(ln)
            continue
        idxs = [int(x) for x in m.group(1).replace(" ", "").split(",")]
        cited_text = " ".join(
            speeches[i - 1][2] for i in idxs if 1 <= i <= len(speeches)
        ).lower()
        for term in terms:
            pat = rf"\b{re.escape(term)}\b"
            if re.search(pat, ln, re.IGNORECASE) and not re.search(pat, cited_text):
                ln = re.sub(rf"\s*{pat}", "", ln, flags=re.IGNORECASE)
        ln = re.sub(r"\s+([.,;:!?])", r"\1", ln)  # tidy space before punctuation
        ln = re.sub(r"[ \t]{2,}", " ", ln).rstrip()
        out.append(ln)
    return "\n".join(out)


def _generate_body(
    prompt: str, system: str, speeches: list, model: str, query: str, tries: int = 2
) -> str:
    """Call the model, retrying with a fresh seed when it drifts off-format.

    Per attempt, in preference order:
      1. a well-formatted **bold-title** list: use immediately
      2. any plain or bulleted list:           keep as fallback
      3. Q&A or prose:                          discard and retry

    Returns the post-processed (cited) body. Debate-style queries often only
    reach tier 2 because the model can't attribute them per-speaker, so we accept
    a plain list rather than burning every retry chasing bold formatting.
    """
    raw = ""
    fallback = ""
    for _ in range(tries):
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "system": system,
                "prompt": prompt,
                "stream": False,
                "options": _options(),  # new random seed each attempt
            },
        )
        raw = resp.json()["response"]
        if _well_formatted(raw):
            break
        if not fallback and _list_items(raw):
            fallback = raw  # good enough; keep it in case later attempts fail
    else:
        raw = fallback or raw
    body = _scrub_ungrounded_terms(_auto_cite(raw, speeches), query, speeches)
    return _drop_non_answers(body)


def answer(
    query: str, collection, conn, model: str = DEFAULT_MODEL
) -> dict:
    speeches = _retrieve(query, collection, conn)
    lang = _detect_lang(query)
    prompt = _build_prompt(query, speeches)

    body = _generate_body(prompt, _build_system(lang), speeches, model, query)
    intro = _random_intro(lang)
    answer_text = f"{intro}\n\n{body}" if body else intro

    return {"answer": answer_text, "sources": _sources_payload(speeches)}


def _stream_typing(text: str, delay: float = 0.012) -> Generator[dict, None, None]:
    """Emit text character by character with a small delay.

    Locally-generated strings (intro/conclusion) type out visibly instead of
    appearing all at once, matching the live-streamed body's pace.
    """
    for ch in text:
        yield {"type": "token", "text": ch}
        time.sleep(delay)


def stream_answer(
    query: str, collection, conn, model: str = DEFAULT_MODEL
) -> Generator[dict, None, None]:
    """Yield SSE-style dicts:

      {"type": "token", "text": "..."}                     incremental text
      {"type": "done", "answer": "...", "sources": [...]}   final result

    The body streams live from Ollama token by token so the user sees progress
    immediately, which matters on hardware where one generation takes tens of
    seconds. The model's leading preamble is buffered away so it doesn't echo our
    intro; names and citations are finalised once in the `done` event, which
    carries the fully post-processed answer.
    """
    speeches = _retrieve(query, collection, conn)
    lang = _detect_lang(query)
    prompt = _build_prompt(query, speeches)
    intro = _random_intro(lang)

    yield from _stream_typing(intro)

    full_text = ""
    pending = ""
    live = False          # have we started forwarding body tokens?
    body_started = False  # have we emitted the intro/body separator?

    def _begin_body():
        nonlocal body_started
        if not body_started:
            body_started = True
            return {"type": "token", "text": "\n\n"}
        return None

    with requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "system": _build_system(lang),
            "prompt": prompt,
            "stream": True,
            "options": _options(),
        },
        stream=True,
    ) as r:
        for line in r.iter_lines():
            if not line:
                continue
            token = json.loads(line).get("response", "")
            if not token:
                continue
            full_text += token
            if live:
                yield {"type": "token", "text": token}
                continue
            # Buffer until the first list marker (dropping the preamble), or until
            # enough prose has accumulated that there clearly is no list.
            pending += token
            m = re.search(r"\d+[.)]\s", pending)
            if m:
                live = True
                sep = _begin_body()
                if sep:
                    yield sep
                yield {"type": "token", "text": pending[m.start():]}
                pending = ""
            elif len(pending) > 200:
                live = True
                sep = _begin_body()
                if sep:
                    yield sep
                yield {"type": "token", "text": pending}
                pending = ""

    body = _scrub_ungrounded_terms(_auto_cite(full_text, speeches), query, speeches)
    body = _drop_non_answers(body)

    if body:
        # Short prose that never tripped the live threshold: emit it now.
        if not body_started:
            yield {"type": "token", "text": "\n\n"}
            yield from _stream_typing(body)

    final_answer = f"{intro}\n\n{body}" if body else intro
    yield {
        "type": "done",
        "answer": final_answer,
        "sources": _sources_payload(speeches),
    }

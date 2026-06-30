import json
import random
import re
import time
from collections.abc import Generator

import requests
from lingua import Language, LanguageDetectorBuilder

import myhansard

# Built once at import; loading the language models is expensive. English vs Malay only.
_DETECTOR = (
    LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.MALAY)
    .build()
)


def _detect_lang(query: str) -> str:
    lang = _DETECTOR.detect_language_of(query)
    return "Bahasa Malaysia" if lang == Language.MALAY else "English"


# Intro/conclusion templates. Generated in Python so the wording is reliable and
# varies per call, rather than left to the model.
_INTROS_MS = [
    "Ya, beberapa isu telah dibangkitkan dalam sidang Dewan Rakyat."
    " Berikut adalah ringkasannya:",
    "Terdapat beberapa isu yang telah dibincangkan dalam Parlimen."
    " Berikut adalah senarai isu tersebut:",
    "Beberapa ahli parlimen telah membangkitkan isu-isu berikut"
    " dalam sidang Dewan Rakyat:",
    "Dalam sidang Dewan Rakyat, beberapa isu penting telah dibangkitkan."
    " Berikut adalah senarai isu yang dikemukakan:",
    "Isu-isu berikut telah dikemukakan oleh beberapa ahli parlimen"
    " dalam sidang Dewan Rakyat:",
]

_INTROS_EN = [
    "Yes, several issues were raised during the Dewan Rakyat session."
    " Here is a summary:",
    "The following issues were brought up in Parliament:",
    "Several members of parliament raised the following issues"
    " during the Dewan Rakyat sitting:",
    "Here are the issues raised in the parliamentary session:",
    "A number of issues were debated in the Dewan Rakyat."
    " Below is a summary:",
]

_CONCLUSIONS_MS = [
    "Secara keseluruhan, isu-isu ini mencerminkan keperluan mendesak"
    " untuk menangani masalah bekalan dan infrastruktur di Malaysia.",
    "Kesimpulannya, pelbagai isu berkaitan perkhidmatan awam telah"
    " mendapat perhatian serius daripada ahli parlimen.",
    "Semua isu ini menunjukkan perlunya kerajaan mengambil tindakan segera"
    " bagi memastikan kesejahteraan rakyat.",
    "Isu-isu ini mencerminkan keperluan dasar yang komprehensif untuk"
    " menangani cabaran infrastruktur negara.",
    "Secara keseluruhannya, isu-isu ini menggambarkan cabaran yang"
    " dihadapi rakyat dan perlunya penyelesaian segera.",
]

_CONCLUSIONS_EN = [
    "In summary, these issues reflect the urgent need to address"
    " infrastructure and public service challenges in Malaysia.",
    "Overall, parliament members have raised significant concerns"
    " that require immediate government attention.",
    "These issues collectively highlight the need for comprehensive"
    " policy solutions to improve public services.",
    "In conclusion, these debates underscore the importance of timely"
    " government action on essential services.",
    "Together, these issues demonstrate the ongoing challenges faced"
    " by citizens and the need for effective solutions.",
]


def _random_intro(lang: str) -> str:
    return random.choice(_INTROS_EN if lang == "English" else _INTROS_MS)


def _random_conclusion(lang: str) -> str:
    pool = _CONCLUSIONS_EN if lang == "English" else _CONCLUSIONS_MS
    return random.choice(pool)


# Speaker parsing
_HONORIFIC_RE = re.compile(
    r"\b(bin|binti|bt\.?|Dato'?|Dato['’ʼ]|Datuk|Datin|Tan Sri|Tun|Dr|Tuan|"
    r"Puan|Haji|Hajah|YB|YAB)\b",
    re.IGNORECASE,
)
_ROLE_RE = re.compile(r"\bMenteri\b|\bPengerusi\b|Yang di-Pertua", re.IGNORECASE)


def _format_speaker(speaker_raw: str) -> str:
    """Turn "Tuan Khoo Poay Tiong [Kota Melaka]" into "Tuan Khoo Poay Tiong (Kota Melaka)".

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
        if not segment or re.search(r"\[\d+\]", segment):
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
    keywords = query.replace("?", "").split()
    keyword_conditions = " OR ".join([f"content LIKE '%{k}%'" for k in keywords])
    cursor.execute(
        "SELECT id FROM speeches"
        f" WHERE ({keyword_conditions}) AND LENGTH(content) > 100 LIMIT 10"
    )
    keyword_ids = [row[0] for row in cursor.fetchall()]

    # Vector hits are relevance-ranked; keep that order, append keyword extras,
    # dedup, cap at 8. A smaller context keeps prompt-eval (and time-to-first-token)
    # fast on GPU-limited hardware and keeps the source list readable.
    ordered_ids = list(dict.fromkeys(ids + keyword_ids))[:8]
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
        f"Example (write yours in {lang}):\n{example}\n\n"
        f"Context:\n{numbered_context}\n\n"
        f"Question: {query}\n\n"
        f"Write the answer in {lang.upper()} only:\n1."
    )


def _build_system(lang: str) -> str:
    source = "Malay" if lang == "English" else "English"
    return (
        "You are a parliamentary research assistant that summarises Hansard "
        f"debates. The source text is in {source}, but you MUST write your "
        f"ENTIRE answer in {lang} only — do not use any other language "
        "(keep proper nouns such as names and places as-is). "
        "Summarise each issue in your own words, but always keep the "
        "speaker's exact name and constituency. "
        "Always cite sources with [n] bracket notation."
    )

_TEMPERATURE = 0.3


def _options() -> dict:
    return {"temperature": _TEMPERATURE, "seed": random.randint(0, 99999)}


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


def _generate_body(
    prompt: str, system: str, speeches: list, model: str, tries: int = 2
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
            "http://localhost:11434/api/generate",
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
    return _auto_cite(raw, speeches)


def answer(
    query: str, collection, conn, model: str = "llama3.1:8b-instruct-q4_K_M"
) -> dict:
    speeches = _retrieve(query, collection, conn)
    lang = _detect_lang(query)
    prompt = _build_prompt(query, speeches)

    body = _generate_body(prompt, _build_system(lang), speeches, model)
    intro = _random_intro(lang)
    conclusion = _random_conclusion(lang)
    answer_text = f"{intro}\n\n{body}\n\n{conclusion}" if body else intro

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
    query: str, collection, conn, model: str = "llama3.1:8b-instruct-q4_K_M"
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
    conclusion = _random_conclusion(lang)

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
        "http://localhost:11434/api/generate",
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

    body = _auto_cite(full_text, speeches)

    if body:
        # Short prose that never tripped the live threshold: emit it now.
        if not body_started:
            yield {"type": "token", "text": "\n\n"}
            yield from _stream_typing(body)
        yield {"type": "token", "text": "\n\n"}
        yield from _stream_typing(conclusion)

    final_answer = f"{intro}\n\n{body}\n\n{conclusion}" if body else intro
    yield {
        "type": "done",
        "answer": final_answer,
        "sources": _sources_payload(speeches),
    }

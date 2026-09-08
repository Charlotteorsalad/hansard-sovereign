# RAG + small-model lessons learned (Hansard Sovereign)

This document records the recurring problems hit — and the fixes landed on — while building
Hansard RAG on **Llama 3.1 8B (local Ollama)**. One-line summary:

> **A small model is a "content generator," not a "format engine." Anything Python can
> handle deterministically shouldn't be delegated to it.**

Related files:
- [library/myhansard/rag.py](../library/myhansard/rag.py) — retrieval / prompt / post-processing
- [web/app/page.tsx](../web/app/page.tsx) — frontend rendering / streaming

---

## 1. Output language didn't follow the question's language

**Symptom:** a user asks in English, but the answer comes back in Malay.

**Root cause:**
1. `_SYSTEM` originally hardcoded `"Always respond in English"`, then was loosened to a
   generic `"same language as question"` — under a **Malay-language context**, the small
   model ignores that instruction and defaults to whatever language the context is in.
2. The example in the prompt was written in Malay → the model imitates the example's
   language.

**Fix:** the system prompt is **built dynamically from the detected language**, worded
forcefully and specifically, with the example also swapped into the target language.

```python
def _build_system(lang: str) -> str:
    # The corpus is predominantly Malay regardless of which language the
    # question is in — say that plainly rather than deriving "source
    # language" from the answer language.
    return (
        "You are a parliamentary research assistant that summarises Hansard "
        "debates. The source text is mainly in Malay, but you MUST write your "
        f"ENTIRE answer in {lang} only — do not use any other language "
        "(keep proper nouns such as names and places as-is). "
        "Summarise each issue in your own words, but always keep the "
        "speaker's exact name and constituency. "
        "Always cite sources with [n] bracket notation."
    )
```

**A follow-up bug hit along the way:** the first version had
`source = "Malay" if lang == "English" else "English"` — trying to pair "source language"
inversely against "answer language," but the condition was flipped: when the question was in
Malay, it told the model "the source is English." The model could still read the Malay
context fine, so the answer itself wasn't wrong, but the system prompt was lying to it.
**Lesson:** any "inversely paired" logic — especially a conditional expression juggling two
languages/two directions — deserves a second look. This class of bug doesn't crash anything;
it just quietly feeds the model wrong information, and it's very hard to spot from the output
alone.

The prompt **anchors the language at both ends**:

```python
return (
    f"Summarise the following Malaysian Parliament speeches"
    f" IN {lang.upper()}.\n\n"
    # ...format instructions + a language-matched example...
    f"Question: {query}\n\n"
    f"Write the answer in {lang.upper()} only:\n1."
)
```

**Lesson:** for things a model **is capable of** (like language), instructions need to be
**forceful, specific, and repeated at both the start and end** — even the example must be
swapped into the target language. Don't rely on a static, generic system prompt.

---

## 2. Hardcoded word-list language detection didn't scale

**Symptom:** language detection used a hand-written English/Malay word list; every new
sentence pattern meant adding more words.

**Root cause:** enumeration never converges — you can never finish the list.

**Fix:** switched to the [`lingua`](https://github.com/pemistahl/lingua-py) library — three
lines, and it handles any sentence.

```python
from lingua import Language, LanguageDetectorBuilder

# Built once (loading the language models is expensive); English vs Malay only.
_DETECTOR = (
    LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.MALAY)
    .build()
)

def _detect_lang(query: str) -> str:
    lang = _DETECTOR.detect_language_of(query)
    return "Bahasa Malaysia" if lang == Language.MALAY else "English"
```

Verified it even handles loanword traps correctly:

| Question | Detected |
|------|------|
| `Isu tentang data center di Melaka` | Bahasa Malaysia ✅ (has English loanwords, but more Malay markers) |
| `Summarise the education debate` | English ✅ |

**Lesson:** **use a library instead of hand-written rules whenever one exists.** A hand-rolled
word list or regex enumeration is a sign of technical debt.

---

## 3. Formatting kept breaking (bold titles / names / constituencies)

**Symptom:** every prompt tweak meant to fix one thing would break another (fix A, break B)
on the small model. Examples: an MP's name getting generalized to
`Ahli Parlimen Khoo Poay Tiong`, a constituency rendered as `dari Selangau` instead of
`(Selangau)`, honorifics (Tuan/Dato') dropped.

**Root cause:** relying on a small model to strictly follow a complex format is unreliable.

**Fix:** anything requiring "100% consistency" moved to **Python post-processing**, not left
to the model.

### 3a. speaker_raw → a clean "Name (Constituency)"

```python
def _format_speaker(speaker_raw: str) -> str:
    """
    "Tuan Khoo Poay Tiong [Kota Melaka]" -> "Tuan Khoo Poay Tiong (Kota Melaka)"
    Handles role prefixes and numeric prefixes; logic mirrors the frontend's parseSource().
    """
    clean = re.sub(r"\s+", " ", speaker_raw.replace("\n", " ")).strip()
    clean = re.sub(r"^\d+\.\s*", "", clean)               # drop leading "2. "
    m = re.match(r"^(.*?)\s*\[([^\]]+)\]", clean)
    if not m:
        return clean
    outer, inner = m.group(1).strip(), m.group(2).strip()
    inner = re.sub(r"\s+(minta|menyatakan|soalan).*$", "", inner)
    if _HONORIFIC_RE.search(inner) or _ROLE_RE.search(outer):
        name, constituency = inner, outer     # the brackets actually held the name
    else:
        name, constituency = outer, inner
    return f"{name} ({constituency})" if constituency else name
```

### 3b. Correcting the model's "Ahli Parlimen X" back to the real name

```python
def _normalize_speaker_names(text: str, speeches: list) -> str:
    """Replace the model's loose '(Ahli Parlimen) <name>' with 'Honorific Name (Constituency)'."""
    entries = []
    for s in speeches:
        formatted = _format_speaker(s[1])                 # "Tuan Khoo Poay Tiong (Kota Melaka)"
        name_part = re.sub(r"\s*\(.*?\)\s*$", "", formatted).strip()
        core = _LEADING_TITLE_RE.sub("", name_part).strip()  # "Khoo Poay Tiong"
        if core and "(" in formatted:
            entries.append((core, formatted))

    entries.sort(key=lambda e: len(e[0]), reverse=True)   # longest name first, avoid partial matches

    for core, formatted in entries:
        pattern = re.compile(
            r"(?:Ahli Parlimen\s+)?"
            r"(?:(?:Tuan|Puan|Dato['’ʼ]?|Datuk|Datin|Tan Sri|Tun|Dr|Haji|"
            r"Hajah|YB|YAB)\.?\s+)*"
            + re.escape(core)
            + r"(?!\s*\()",                               # skip if already followed by "(...)"
        )
        text = pattern.sub(formatted, text)
    return text
```

### 3c. "dari X" → "(X)" fallback

```python
def _fix_constituency(text: str) -> str:
    return re.sub(
        r" dari ((?:[A-Z][a-zA-Z']*(?:\s+(?=[A-Z]))?)+)",
        lambda m: f" ({m.group(1).strip()})",
        text,
    )
```

**Lesson:** **the model handles content, Python handles format.** Anything requiring strict
consistency — title structure, names, citations — should be enforced in code; don't expect a
small model to comply every time.

---

## 4. Intro/conclusion drift (stray citations, count mismatches, fabricated closing lines)

**Symptom:** letting the model write its own opening line and closing paragraph led to it
tacking on stray `[n]` citations, and sometimes even turning the closing line into another
bullet list.

**Fix v1:** fixed boilerplate lines via **Python-side random templates**, chosen by language,
never touched by the model.

```python
_INTROS_EN = [
    "Yes, several issues were raised during the Dewan Rakyat session. Here is a summary:",
    "The following issues were brought up in Parliament:",
    # ...
]
def _random_intro(lang: str) -> str:
    return random.choice(_INTROS_EN if lang == "English" else _INTROS_MS)
```

**Two further problems surfaced later, both boiling down to "the template itself was also
fabricating information":**

1. **The intro hardcoded a count word:** `"Yes, **several** issues were raised"` /
   `"**beberapa** isu"` — if retrieval was narrow and the answer ended up with only one item,
   the intro still said "several/beberapa" (multiple), a mismatch. And because the intro is
   emitted **before** the body is generated during streaming, at that point it isn't yet known
   how many items there will be, so it can't branch on count. **Fix:** reworded the intro to
   be count-neutral (dropped "several"/"beberapa" and the forced plural `isu-isu`) — valid
   whether the answer ends up with one item or eight.

2. **The conclusion was pure fabrication:** e.g. *"these issues reflect the urgent need to
   improve infrastructure and public services"* — regardless of whether the topic was
   taxation or a running event, the closing text never changed, and **it had no source behind
   it at all**, directly violating the "every sentence needs an `[n]` citation" design
   principle. **Fix:** dropped the conclusion entirely rather than trying to build a
   count/topic-aware version — a list that **ends where the evidence ends** is correct
   behavior on its own; it doesn't need a bolted-on summary sentence.

```python
answer_text = f"{intro}\n\n{body}" if body else intro   # no conclusion anymore
```

**Lesson:** "boilerplate doesn't need the model to generate it" is correct on its own, but
**boilerplate can fabricate information too** — its content must either be genuinely neutral
with respect to count/outcome, or simply not exist. Don't take the shortcut of writing a line
that "sounds about right" but doesn't actually match the answer.

The intros currently shipped (count-neutral, no forced plural) look like this:

```python
_INTROS_EN = [
    "Here is what was raised in the Dewan Rakyat, based on the Hansard record:",
    "The following was raised during the Dewan Rakyat sitting:",
    "From the Hansard record of the Dewan Rakyat:",
    "Here is a summary of what was discussed in Parliament:",
    "Based on the parliamentary record:",
]
```

---

## 5. `_auto_cite`: only process numbered lines, drop the noise

During post-processing, the text is split on `\n` and **only numbered lines are kept** —
which conveniently drops any hallucinated bullets or stray preamble the model produced.

```python
text = _normalize_speaker_names(text, speeches)

result_lines = []
for line in text.split("\n"):
    line = _fix_constituency(line.strip())
    if not line:
        continue
    if re.match(r"^\d+\.", line):
        result_lines.append(_cite(line))
    # non-numbered lines (hallucinated intro/conclusion/bullets) are simply dropped
return "\n".join(result_lines)
```

---

## 6. Hybrid retrieval: the "keyword half" was completely inert (fixed with RRF)

**Symptom:** retrieval quality for English questions was noticeably worse than for Malay
questions, even though the corpus (parliamentary speeches) is predominantly Malay to begin
with. Digging in revealed two layered problems:

1. Keyword retrieval was `LIKE %word%`-matching English words directly against the Malay
   source text, so it never hit (`subsidies` never appears in a Malay speech).
2. **A deeper bug:** even after adding cross-lingual keyword mapping, the original fusion
   logic was `list(dict.fromkeys(ids + keyword_ids))[:8]` — vector retrieval alone returned 10
   results, which already filled the top-8, so `keyword_ids` could never rank into the top 8.
   **It had never actually taken effect.** Something named "hybrid retrieval" had, in
   practice, always been pure vector retrieval.

**Fix:**
- `_keyword_terms()`: filters Chinese/English/Malay stopwords, maps English words to their
  Malay equivalents (`subsidies→subsidi`, `education→pendidikan`), with common phrases mapped
  separately (`cost of living→sara hidup`).
- Fuse the two ranked lists with **Reciprocal Rank Fusion** (RRF, k=60), instead of a naive
  concatenate-dedupe-truncate:

```python
scores: dict = {}
for rank_list in (ids, keyword_ids):
    for rank, _id in enumerate(rank_list):
        scores[_id] = scores.get(_id, 0.0) + 1.0 / (rank + 60)
ordered_ids = sorted(scores, key=lambda i: scores[i], reverse=True)[:8]
```

Effect: after mapping, the English query "education funding" went from matching 174 Malay
records to matching 4193; after fusion, the education minister's speech correctly landed as
the #1 retrieval result.

**Lesson:** "concatenate + dedupe + truncate" (`a + b, dedup, [:N]`) looks like a hybrid
fusion, but as soon as one list's length is already ≥ N, the other list **can never make it
in**. This kind of bug doesn't throw an error and doesn't make the whole feature look
"obviously broken" — it just leaves one component quietly doing nothing. Code that structurally
looks like "two signals fused" isn't proof enough — you have to actually count how much of
each source ends up in the final fused result.

---

## 7. Hallucination isn't just fabricating facts — it also fabricates "specificity" (grounding scrub)

**Symptom:** asked "what did members say about running events (sukan larian) in 2025," the
most relevant retrieved source was actually a **generic community sports-funding
announcement** that never mentions the word "larian" (running) anywhere. But the model's
generated title read "Penganjuran Larian Sukan," and the body said "funding allocated for
larian" — it had directly pinned the question's own keyword onto the source, fabricating a
specificity the source never had.

**First attempt (insufficient):** adding "base this strictly on the source content, don't
borrow the question's wording" to the prompt — across 4 repeated real generation runs, **all
4** still wrote "larian" into the title. For a small model like 8B-q4, prompt instructions
alone can't suppress its tendency to anchor on the question's own keywords.

**Actual fix:** deterministic verification — for each answer line, check whether the
**specific source it cites** actually contains the term in question; if not, strip that term
from the line and clean up the resulting extra whitespace/punctuation.

```python
for term in terms:  # narrow keywords extracted from the question (excluding domain-generic words like sukan/tahun)
    pat = rf"\b{re.escape(term)}\b"
    if re.search(pat, ln, re.IGNORECASE) and not re.search(pat, cited_text):
        ln = re.sub(rf"\s*{pat}", "", ln, flags=re.IGNORECASE)
```

Key design point: only "narrow, specific" terms (like larian) are verified this way — generic
domain words like `sukan`/`program`/`tahun` are explicitly excluded from the check, otherwise
a large amount of normal phrasing would get misjudged as "unsupported" and stripped
incorrectly.

**Lesson:** a small model's hallucination isn't limited to "making up facts" — a subtler form
is **pinning the question's specific wording onto a generic source**, which makes the answer
read as more precise and more on-topic than the source actually supports. Because this kind
of hallucination is fluent and well-formatted, it's actually harder to catch than an obvious
non-sequitur. Prompt instructions are largely ineffective against it — what's needed is a
deterministic check of "does the source cited by this line actually support this line's
wording," not just trusting the prompt.

---

## 8. Two small but real bugs: SQL injection + not every retrieved result belongs in the answer

**SQL injection:** early on, `_retrieve` concatenated keywords directly into the SQL string.
MP names commonly contain apostrophes (e.g. `Dato'`), and dropping one into
`LIKE '%...%'` directly breaks the statement — this isn't just a crash risk, it's a real SQL
injection surface. **Fix:** keywords are always bound via parameterized queries
(`content LIKE ?`), never string-concatenated.

**Padding with filler entries:** given 8 retrieved sources, the model tends to **write one
item per source**, even when some sources don't address the question at all — inventing a
line like "this MP did not express a specific view on this" just to fill the slot. That line
carries no information and cites a source that doesn't support it. **Fix:** the prompt
explicitly says "only list sources that actually answer the question — three real answers
beat eight padded ones," and Python adds a second layer of regex filtering
(`_drop_non_answers`) as a safety net — but it **deliberately does not match** phrasing like
`"tidak menjawab"` / `"did not answer"`, because an MP pressing a minister on why they didn't
answer is real, citable substance and shouldn't be caught by this filter.

**Lesson:** not every retrieved result belongs in the answer; "better to answer with fewer
items that are correct than to pad with fabricated ones" should be a design-level rule fixed
in code, not something left to the model's discretion.

---

## 9. Frontend: streaming content bled across chat windows on switch

**Symptom:** window A finishes answering, a question is asked in window B, and switching back
to A shows **B's streaming content appended below A**.

**Root cause:** `streamingContent` / `loading` were **global state**, shared across all
windows.

**Fix:** added `streamingChatId`, and the streaming bubble only renders when it equals the
currently active window.

```tsx
const [streamingChatId, setStreamingChatId] = useState<string | null>(null);

// on send:
setStreamingChatId(currentChatId);

// in finally:
setStreamingChatId(null);

// on render:
{loading && streamingChatId === activeChatId && (
  <StreamingBubble text={streamingContent} />
)}
```

**Lesson:** **state that belongs to a single session must be keyed by an ID** — it can't be
rendered straight off global state.

**The same lesson was hit again later, at a larger scale:** navigating away from the page or
deleting a chat that was still streaming didn't actually cancel the in-flight request — the
orphaned fetch kept occupying Ollama (the 4 GB GPU can only run one generation at a time),
causing subsequent requests to queue up and stall; and deleting a chat used a stale `chats`
snapshot captured in a closure to write back to `localStorage`, so a deleted chat could get
"resurrected" once its stream finished. **Fix:** added an `AbortController` (aborting the
previous one on any new action or unmount), and on completion, results are merged using
`chatsRef` (a ref that always points at the latest state) instead of the snapshot captured at
the start of the function, plus an existence check — if the chat has already been deleted, the
result is simply discarded. **Lesson:** "keying state by ID" solves the rendering-level
crosstalk, but the lifecycle of the async request itself (cancellation, and which "latest
state" to merge results into on completion) is a separate concern that needs its own design —
it doesn't automatically get fixed just because the UI stopped bleeding across windows.

---

## 10. Streaming noise: leaking preamble + too many source cards

**Preamble leak:** during streaming, the model may emit an opening line first; this has to be
buffered until `"1."` is detected before forwarding begins.

```python
pending, list_started = "", False
for token in stream:
    full_text += token
    if list_started:
        yield {"type": "token", "text": token}
    else:
        pending += token
        m = re.search(r"\d+\.", pending)
        if m:                       # list start detected
            list_started = True
            yield {"type": "token", "text": pending[m.start():]}
            pending = ""
```

**20 source cards is too many:** the retrieval stage caps at a fixed 8 (reasoning: a longer
prompt slows down prefill/TTFT, and a longer source list is also harder to read). How those 8
actually get chosen from the two ranked lists is covered in section 6's RRF fusion — this is
also where the earlier "concatenate-dedupe-truncate" bug that neutralized the keyword half
originally lived.

---

## Overall principles (don't repeat these)

1. **A small model is a content generator, not a format engine.** Anything structural gets
   backstopped in code.
2. **Change one variable at a time, and test immediately after.** Chained prompt edits cause
   regressions in features that already worked — most of the "this actually made things
   messier" moments in this project trace back to this.
3. **Prefer deterministic over probabilistic wherever possible.** Language templates, name
   formatting, detection libraries — hand off what can be made certain, don't leave it to
   chance.
4. **Test against the real model before editing the prompt** (hit Ollama directly with
   `curl` for comparison) — about 10× faster than editing blind:

```bash
curl -s http://localhost:11434/api/generate -d '{
  "model":"llama3.1:8b-instruct-q4_K_M",
  "system":"...","prompt":"...","stream":false,
  "options":{"temperature":0.3,"seed":7}
}'
```

5. **Things the model can reliably do (language):** instructions need to be forceful,
   specific, repeated at both the start and end, with the example matching too.
   **Things the model can't reliably do (formatting/names/boilerplate):** force it with Python
   post-processing.
6. **"Looks like hybrid/fusion" doesn't mean "actually works."** A concatenate-dedupe-truncate
   pattern means that once one input list is already long enough, the other input may never
   make it into the result at all. After changing fusion logic, count how much of each source
   ends up in the actual result — don't just eyeball the code structure.
7. **Hallucination isn't limited to fabricating facts — it also fabricates "specificity."**
   Pinning a question's specific wording onto a generic source makes an answer read as more
   precise than the source itself supports — and it's harder to catch precisely because it's
   more fluent. Prompt instructions alone can't suppress this; it needs a deterministic check
   of whether the wording in a given line is actually supported by the source that line cites.
8. **Async lifecycle is a separate problem from rendering logic.** "Keying state by ID" fixes
   cross-window bleed, but canceling in-flight requests and deciding which "latest state" a
   completed request should merge into are a separate layer that needs its own design.

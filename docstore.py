"""
docstore.py - PDF parsing and storage: the backend of the 4 document tools.

Only the tool functions in harness.py read STORE. The agent loop never sees document text
except through tool results, so parsing at upload is the tools' backend, not pre-reading.

    1. LIMITS + NORMALISE  - upload limits; text cleaning shared with the evidence check
    2. PAGE TEXT           - one styled pass per page: blocks in reading order, table rows,
                             hidden text, running headers/footers, contents pages
    3. HEADINGS            - PDF outline -> font size / bold -> numbered headings
    4. SEARCH              - positional inverted index with 3 matching tiers
    5. STORE               - Doc + DocStore (in memory, keyed by a hash of the file)
"""

from __future__ import annotations

import bisect
import hashlib
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

# =============================================================================
# 1. LIMITS + NORMALISE
# =============================================================================

MAX_FILE_MB = 50
MAX_PAGES = 2000         # generous on purpose: refusing the judges' PDF costs more than a slow upload
MIN_CHARS_PER_PAGE = 50  # below this on average, the PDF is probably scanned (we have no OCR)

_CHAR_FIXES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                             "–": "-", "—": "-", "­": "", "∗": "*",  # ∗ -> * ("A∗ search")
                             # Hyphens that look like '-' (live: gpt-oss quoted "decision‑useful" with U+2011,
                             # the page had '-', so a real quote failed). NFKC turns U+2011 into U+2010.
                             "‐": "-", "‑": "-", "‒": "-", "―": "-", "−": "-"})


def clean(text: str) -> str:
    """Display cleaning: ligatures (NFKC: 'ﬁ' -> 'fi'), curly quotes and dashes, and words
    split across lines by a hyphen ('contro-\\nversial' -> 'controversial')."""
    text = unicodedata.normalize("NFKC", text).translate(_CHAR_FIXES)
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def normalize(text: str) -> str:
    """Matching form: cleaned, lower case, single spaces. Used by search AND the evidence check.
    '|' becomes a space: it is our own table-cell separator, so a quote matches with or without it."""
    return re.sub(r"\s+", " ", clean(text).replace("|", " ")).strip().lower()


# =============================================================================
# 2. PAGE TEXT: one pass per page in "dict" mode (text + position, size, bold, colour)
# =============================================================================

HIDDEN_MARK = "[HIDDEN TEXT: invisible in the PDF] "  # white or tiny text: marked, never removed
MIN_VISIBLE_PT = 4   # text smaller than this is unreadable on paper
ROW_TOLERANCE_PT = 2  # lines whose bottoms are this close are on the same row
CELL_GAP_EM = 2       # same-row pieces this far apart (in font sizes) are table cells -> ' | ';
                      # measured on the sample PDF: same sentence ~0.8-1 em apart, table cells 8-12 em
_WHITE = 0xFFFFFF

_EDGE_NUMBER = re.compile(r"^\d{1,4}(?=\s)|(?<=\s)\d{1,4}$|^\d{1,4}$")  # bare page number at either end
_DOT_LEADER = re.compile(r"(?:\.\s*){4,}(?:\|\s*)?\d{1,4}$")             # "Notice period ....... 12"
_ENDS_NUMBER = re.compile(r"\s\d{1,4}$")


@dataclass
class Line:
    """One visible text line, kept for heading detection."""
    page: int
    block: int     # index of its block on the page (to skip running headers/footers)
    size: float    # largest font size on the line
    bold: bool
    alone: bool    # the only line in its block: a standalone heading usually is
    text: str


def _join_spans(spans: list[dict]) -> str:
    """Rebuild a line from its spans. Some PDFs store each word as its own span without a space,
    so a space is added wherever there is a visible gap between two spans."""
    text = spans[0]["text"]
    for prev, cur in zip(spans, spans[1:]):
        gap = cur["bbox"][0] - prev["bbox"][2]
        text += (" " if gap > 0.15 * cur["size"] and not text.endswith(" ") else "") + cur["text"]
    return " ".join(clean(text).split())


def read_page(page, pno: int) -> tuple[list[str], list[Line]]:
    """Returns (blocks, lines) for one page.
    blocks: block texts in reading order (block mode keeps margin notes apart from body text).
            Pieces on the same row are joined: with ' | ' when far apart (table cells), with a space
            when close (one sentence split by the PDF). Invisible lines (white, or under
            MIN_VISIBLE_PT) get HIDDEN_MARK.
    lines:  the visible lines with size and boldness, for heading detection."""
    blocks: list[str] = []
    lines: list[Line] = []
    for block in page.get_text("dict", sort=True)["blocks"]:
        rows: list[list] = []  # [bottom y, right x, text]
        first = len(lines)
        for ln in block.get("lines", []):  # image blocks have no lines
            spans = [s for s in ln["spans"] if s["text"].strip()]
            if not spans:
                continue
            text, (x0, _, x1, bottom) = _join_spans(spans), ln["bbox"]
            size = max(s["size"] for s in spans)
            if all(s["color"] == _WHITE or s["size"] < MIN_VISIBLE_PT for s in spans):
                rows.append([bottom, x1, HIDDEN_MARK + text])
                continue
            prev = rows[-1] if rows else None
            if prev and abs(prev[0] - bottom) <= ROW_TOLERANCE_PT and HIDDEN_MARK not in prev[2]:
                gap = (x0 - prev[1]) / size
                prev[2] += (" " if 0 <= gap < CELL_GAP_EM else " | ") + text
                prev[1] = x1
            else:
                rows.append([bottom, x1, text])
            bold = all(s["flags"] & 16 or "bold" in s["font"].lower() for s in spans)  # flag 16 = bold
            lines.append(Line(pno, len(blocks), round(size, 1), bold, False, text))
        if len(lines) - first == 1:
            lines[-1].alone = True
        if rows:
            blocks.append(clean("\n".join(text for _, _, text in rows)).strip())
    return blocks, lines


def body_size(lines: list[Line]) -> float:
    """The body font size: the most common size, counted by characters (paragraphs outweigh headings)."""
    sizes: Counter = Counter()
    for ln in lines:
        sizes[ln.size] += len(ln.text)
    return sizes.most_common(1)[0][0] if sizes else 0.0


def running_blocks(pages: list[list[str]], big: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """(page index, block index) of running headers/footers: a short first or last block that
    starts/ends with a bare page number, or repeats on 3+ pages once digits are masked.
    big: blocks with heading-size text. Running headers are small print, so these never count
    (otherwise templated headings like 'Section 1: Rules', 'Section 2: Rules' would be dropped)."""
    def mask(t: str) -> str:
        return re.sub(r"\d+", "#", " ".join(t.lower().split()))

    edges = [(pi, bi, " ".join(blocks[bi].split())) for pi, blocks in enumerate(pages) if blocks
             for bi in sorted({0, len(blocks) - 1}) if len(blocks[bi].split()) <= 12 and (pi, bi) not in big]
    repeats = Counter(mask(t) for _, _, t in edges)
    return {(pi, bi) for pi, bi, t in edges if _EDGE_NUMBER.search(t) or repeats[mask(t)] >= 3}


def is_contents_page(text: str) -> bool:
    """Dot leaders on 3+ lines, or the word 'contents' plus mostly lines ending in a page number.
    Kept strict so a page with a table of numbers is not mistaken for a contents page."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    leaders = sum(bool(_DOT_LEADER.search(line)) for line in lines)
    numbered = sum(bool(_ENDS_NUMBER.search(line)) for line in lines)
    return leaders >= 3 or ("contents" in text.lower() and numbered >= 5 and numbered >= 0.4 * len(lines))


# =============================================================================
# 3. HEADINGS: (level, page, title), first fallback that finds anything wins
# =============================================================================

def _looks_like_heading(text: str) -> bool:
    return (len(text.split()) <= 12 and not text.endswith((".", ",", ";", ":"))
            and bool(re.search(r"[A-Za-z]{3}", text)))


def headings_from_outline(pdf) -> list[tuple[int, int, str]]:
    """The PDF's own bookmarks. One or two bookmarks (often just the title) are not a contents list."""
    toc = [(lvl, page, clean(title).strip()) for lvl, title, page in pdf.get_toc() if page >= 1 and title.strip()]
    return toc if len(toc) >= min(3, pdf.page_count) else []


def headings_from_fonts(lines: list[Line], skip: set[tuple[int, int]]) -> list[tuple[int, int, str]]:
    """A heading is a short line either at least 15% larger than the body font, or bold and alone in
    its block at body size (Word-style sub-headings). Body = the most common size by character count.
    Level = rank of the heading sizes, largest = 1. A size used only once (a cover-page title) is level 1
    and is left out of the ranking, so it does not push the real sections down. Wrapped titles are merged.
    skip: running header/footer blocks, so a bold header on every page is not taken as a heading."""
    lines = [ln for ln in lines if (ln.page - 1, ln.block) not in skip]
    if not lines:
        return []
    body = body_size(lines)
    found: list[list] = []  # [page, size, text, line index]
    for i, ln in enumerate(lines):
        if not (ln.size >= body * 1.15 or (ln.bold and ln.alone and ln.size >= body * 0.95)):
            continue
        prev = found[-1] if found else None
        if prev and prev[0] == ln.page and prev[1] == ln.size and prev[3] == i - 1:  # wrapped title
            prev[2], prev[3] = prev[2] + " " + ln.text, i
        else:
            found.append([ln.page, ln.size, ln.text, i])
    found = [f for f in found if _looks_like_heading(f[2])]
    uses = Counter(f[1] for f in found)
    ranked = sorted({size for size, n in uses.items() if n > 1 or len(found) < 4}, reverse=True)
    level = {size: min(rank, 4) for rank, size in enumerate(ranked, start=1)}
    return [(level.get(size, 1), pno, text) for pno, size, text, _ in found]


_NUMBERED = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+[A-Z]")          # "2.1 Notice period"
_NAMED = re.compile(r"^(?:Chapter|Section|Part|Article|Appendix)\s+[\dIVXLC]+\b", re.IGNORECASE)


def headings_from_numbers(pages: list[list[str]], skip: set[tuple[int, int]],
                          contents: list[int]) -> list[tuple[int, int, str]]:
    """Numbered lines like '2.1 Title' (level = numbering depth) or 'Section 4' (level 1).
    Running headers/footers and contents pages are skipped."""
    out = []
    for pi, blocks in enumerate(pages):
        if pi + 1 in contents:
            continue
        for bi, block in enumerate(blocks):
            if (pi, bi) in skip:
                continue
            for line in (" ".join(l.split()) for l in block.splitlines()):
                m = _NUMBERED.match(line)
                if m and _looks_like_heading(line):
                    out.append((min(m.group(1).count(".") + 1, 4), pi + 1, line))
                elif _NAMED.match(line) and _looks_like_heading(line):
                    out.append((1, pi + 1, line))
    return out


# =============================================================================
# 4. SEARCH: positional inverted index, 2 tiers (word family phrase -> loose)
# =============================================================================

_TOKEN = re.compile(r"\w+(?:\.\w+)*\*?")  # keeps "4.2" and "a*" as one token
_SUFFIXES = ("ations", "ation", "ments", "ment", "ings", "ing", "ies", "es", "ed", "s", "y")
FAMILY_EXTRA_LETTERS = 3  # 'term' matches 'terms' but not 'terminal'; 'learn' matches 'learning'


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(normalize(text))


def word_root(word: str) -> str:
    """Strip one common suffix, keeping at least 4 letters: 'terminations' -> 'termin'."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (inserts, deletes, substitutions), computed row by row."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def build_index(page_texts: dict[int, str]) -> dict[str, dict[int, list[int]]]:
    """word -> {page -> [positions of the word on that page]}"""
    index: dict[str, dict[int, list[int]]] = {}
    for pno, text in page_texts.items():
        for pos, token in enumerate(tokenize(text)):
            index.setdefault(token, {}).setdefault(pno, []).append(pos)
    return index


# =============================================================================
# 5. STORE
# =============================================================================

@dataclass
class Doc:
    doc_id: str
    title: str
    filename: str
    pages: list[str]                        # cleaned text, index 0 = page 1 (what get_page returns)
    labels: list[str]                       # printed page labels from the PDF (often empty)
    headings: list[tuple[int, int, str]]    # (level, page, title)
    heading_source: str                     # outline | font | numbered | none
    index: dict[str, dict[int, list[int]]]  # search only: running headers and contents pages left out
    vocab: list[str]                        # sorted index words, for prefix (bisect) and typo lookups
    has_text: bool                          # False = probably scanned (no OCR)
    contents_pages: list[int] = field(default_factory=list)
    parse_ms: int = 0                       # how long the upload took to parse (shown in the UI)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def search(self, keyword: str) -> tuple[list[int], str]:
        """Pages containing the keyword (ascending) and the tier that matched. The first tier that finds
        anything wins: word family as a phrase (words adjacent), then loose: typo-tolerant words on the
        same page in any order.
        No separate exact tier: a word's family always includes the word itself, and on the sample PDF
        exact-only missed key pages ('neural network' missed the perceptron chapter, which says 'networks')."""
        words = tokenize(keyword)
        if not words:
            return [], "none"
        for tier, expand in (("family", self._family), ("loose", self._typo)):
            options = [expand(w) for w in words]  # candidate index words for each query word
            if all(options):
                pages = self._pages_with(options, adjacent=tier == "family")
                if pages:
                    return pages, tier
        return [], "none"

    def _exact(self, word: str) -> list[str]:
        return [word] if word in self.index else []

    def _family(self, word: str) -> list[str]:
        """Index words with the same root, at most FAMILY_EXTRA_LETTERS longer than the query word:
        'terminations' -> root 'termin' -> termination, terminate, terminal. Found with bisect on vocab."""
        root = word_root(word)
        if len(root) < 4:  # too short to prefix-match safely ("ai" would match "aim", "air")
            return self._exact(word)
        i, out = bisect.bisect_left(self.vocab, root), []
        while i < len(self.vocab) and self.vocab[i].startswith(root):
            if len(self.vocab[i]) <= len(word) + FAMILY_EXTRA_LETTERS:
                out.append(self.vocab[i])
            i += 1
        return out

    def _typo(self, word: str) -> list[str]:
        if len(word) < 5:  # short words: typo matching would hit too many unrelated words
            return self._exact(word)
        k = 2 if len(word) >= 9 else 1
        return [v for v in self.vocab  # same first letter + similar length keeps this fast
                if v[0] == word[0] and abs(len(v) - len(word)) <= k and edit_distance(v, word) <= k]

    def _pages_with(self, options: list[list[str]], adjacent: bool) -> list[int]:
        """Pages containing a candidate for every query word; if adjacent, as consecutive words."""
        per_word: list[dict[int, set[int]]] = []  # per query word: page -> positions (candidates merged)
        for candidates in options:
            slot: dict[int, set[int]] = {}
            for c in candidates:
                for page, positions in self.index[c].items():
                    slot.setdefault(page, set()).update(positions)
            per_word.append(slot)
        pages = set.intersection(*sorted((set(s) for s in per_word), key=len))  # rarest word first
        if adjacent and len(per_word) > 1:
            pages = {p for p in pages
                     if any(all(start + k in per_word[k][p] for k in range(1, len(per_word)))
                            for start in per_word[0][p])}
        return sorted(pages)


def parse_pdf(pdf, doc_id: str, filename: str) -> Doc:
    read = [read_page(page, pno) for pno, page in enumerate(pdf, start=1)]  # the one pass over the PDF
    blocks = [page_blocks for page_blocks, _ in read]
    lines = [line for _, page_lines in read for line in page_lines]
    body = body_size(lines)
    big = {(ln.page - 1, ln.block) for ln in lines if ln.size >= body * 1.15}
    running = running_blocks(blocks, big)
    pages = ["\n\n".join(b) for b in blocks]
    contents = [pno for pno, text in enumerate(pages, start=1) if is_contents_page(text)]
    searchable = {pno: "\n".join(line for bi, block in enumerate(page) if (pno - 1, bi) not in running
                                 for line in block.splitlines() if not line.startswith(HIDDEN_MARK))
                  for pno, page in enumerate(blocks, start=1) if pno not in contents}
    index = build_index(searchable)
    headings, source = headings_from_outline(pdf), "outline"
    if not headings:
        headings, source = headings_from_fonts(lines, running), "font"
    if not headings:
        headings, source = headings_from_numbers(blocks, running, contents), "numbered"
    if not headings:
        source = "none"
    return Doc(doc_id=doc_id, title=(pdf.metadata or {}).get("title") or Path(filename).stem,
               filename=filename, pages=pages, labels=[page.get_label() for page in pdf],
               headings=headings, heading_source=source, index=index, vocab=sorted(index),
               has_text=sum(map(len, pages)) >= MIN_CHARS_PER_PAGE * max(len(pages), 1),
               contents_pages=contents)


class DocStore:
    """In-memory store keyed by doc_id (first 8 hex characters of the file's SHA-1)."""

    def __init__(self):
        self.docs: dict[str, Doc] = {}

    def add(self, filename: str, data: bytes) -> Doc:
        if len(data) > MAX_FILE_MB * 1024 * 1024:
            raise ValueError(f"file is larger than {MAX_FILE_MB} MB")
        doc_id = hashlib.sha1(data).hexdigest()[:8]
        if doc_id in self.docs:  # same file uploaded again: already parsed
            return self.docs[doc_id]
        try:
            pdf = pymupdf.open(stream=data, filetype="pdf")
        except Exception as e:
            raise ValueError(f"not a readable PDF ({e})") from e
        with pdf:
            if pdf.needs_pass:
                raise ValueError("PDF is password-protected")
            if pdf.page_count > MAX_PAGES:
                raise ValueError(f"PDF has {pdf.page_count} pages; the limit is {MAX_PAGES}")
            t0 = time.time()
            doc = parse_pdf(pdf, doc_id, filename)
            doc.parse_ms = int((time.time() - t0) * 1000)
        self.docs[doc_id] = doc
        return doc

    def add_file(self, path: str | Path) -> Doc:
        path = Path(path)
        return self.add(path.name, path.read_bytes())

    def get(self, doc_id: str) -> Doc:
        if doc_id not in self.docs:
            raise ValueError(f"unknown doc_id '{doc_id}'. Known documents: {', '.join(self.docs) or 'none'}")
        return self.docs[doc_id]


STORE = DocStore()

"""3-pass Markdown chunking, token-accurate against the real e5 tokenizer.

The tokenizer is the e5 one frozen in ``e5_tokenizer.json.gz`` (see
``E5_TOKENIZER_FILE``), so chunking works offline and before the embedding
model is downloaded. It holds about 250 MB loaded, so it is loaded on demand
and released ``TOKENIZER_IDLE_SECONDS`` after its last use (``_tokenizer_lease``,
the reaper thread); ``truncate_to_tokens`` does not load it for a text that
provably fits (``_token_upper_bound``).

Ported from the R&D ingestion POC and re-targeted at multilingual-e5-small:

1. Split on Markdown headers (h1–h4) — chunks follow document structure.
2. Sub-split large sections with a token-accurate recursive splitter
   (separators favor paragraph > line > sentence > word boundaries). The
   FAISS-era chunker estimated ~3 chars/token against a model truncating at
   128 tokens — two thirds of every chunk was invisible to dense retrieval.
3. Prefix each chunk with its heading breadcrumb ("# A > ## B") and
   re-attach Markdown table headers lost mid-split.

Budget [N6]: the full embedded text is
``passage: [document_name:NAME]\\n<breadcrumb>\\n\\n<chunk>``; with the
default target (~180 tokens) plus breadcrumb and name the total stays far
inside e5's 512-token window. The R&D sweeps found 150–200-token chunks the
most consistent — big chunks measurably HURT dense retrieval.
"""

from __future__ import annotations

import contextlib
import ctypes
import gc
import gzip
import json
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core import config
from src.core.logging import logger
from src.ingestion import phase_timings
from src.ingestion.types import ExtractedDocument

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from tokenizers import Tokenizer

# NB: langchain_text_splitters is imported in FUNCTION scope only (issue #160).
# Its package __init__ eagerly imports every splitter it ships and probes the
# optional NLP stacks they wrap, and this module sits on the boot path (seed ->
# utils -> kb_utils), so a module-level import here taxes every backend start
# for a splitter that is only needed at KB ingestion time.

# The intfloat/multilingual-e5-small tokenizer (snapshot 614241f6), frozen once
# as transformers 5.14.1 builds it: ``AutoTokenizer.from_pretrained(...)
# .backend_tokenizer.to_str()``, gzipped (7.9 MB of JSON, 3.5 MB on disk). That
# build, not the snapshot's raw tokenizer.json, is the one chunking always used:
# the raw file normalizes differently and splits 7 of 43 measured documents
# into other chunks. Loaded with the pure ``tokenizers`` package, so counting
# needs neither transformers nor a download; tests/test_e5_tokenizer.py holds
# it to transformers' ids and chunks.
E5_TOKENIZER_FILE = config.ROOT_DIR / "src" / "ingestion" / "e5_tokenizer.json.gz"

DEFAULT_TARGET_TOKENS = 180
DEFAULT_OVERLAP_TOKENS = 27  # ~15 %

# What fits in e5's 512-token window once ``<s>`` and ``</s>`` are counted.
# Bounding the raw text at 510 content tokens keeps every input inside the
# window, but is NOT byte-identical to an id-level cut at 512 specials
# included: the re-tokenized prefix can end on a different final token
# (measured: 27 of 300 long texts differ there; vector parity stays >=
# 0.9999). The bound
# exists Erudi-side because llama-server (the Windows/Linux embedding child,
# #594) refuses, not truncates, past 512 -- and fails the whole batch.
E5_MAX_CONTENT_TOKENS = 510

HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]

#: Released after this long unused. A burst of KB questions and an ingestion
#: (its chunking, then a passage truncation per embedding batch, seconds
#: apart) keep it loaded; a lone question minutes later reloads it (about
#: 0.5 s, against 250 MB held between questions).
TOKENIZER_IDLE_SECONDS = 60.0

# Lifecycle state, guarded by ``_state``: the tokenizer, how many calls hold
# it, when the last one ended, and the reaper thread that releases it.
_state = threading.Condition()
_load_lock = threading.Lock()
_tokenizer: Tokenizer | None = None
_users = 0
_idle_since = 0.0
_reaper: threading.Thread | None = None
_reaper_stop: threading.Event | None = None

# The tokenizer's normalizer and pre-tokenizer alone (``_token_upper_bound``),
# resident: they weigh about a megabyte (the vocabulary is the 250 MB).
_bound_pipeline: Tokenizer | None = None
_bound_specials: tuple[str, ...] = ()
_bound_lock = threading.Lock()


def _load_frozen_tokenizer() -> Tokenizer:
    from tokenizers import Tokenizer

    return Tokenizer.from_buffer(gzip.decompress(E5_TOKENIZER_FILE.read_bytes()))


def _loaded_tokenizer() -> Tokenizer:
    """The tokenizer, loaded if it is not (the caller holds a lease)."""
    global _tokenizer
    with _state:
        tokenizer = _tokenizer
    if tokenizer is not None:
        return tokenizer
    with _load_lock:
        with _state:
            tokenizer = _tokenizer
        if tokenizer is None:
            with phase_timings.timed("tokenizer"):
                tokenizer = _load_frozen_tokenizer()
            with _state:
                _tokenizer = tokenizer
                _start_reaper_locked()
    return tokenizer


@contextlib.contextmanager
def _tokenizer_lease() -> Iterator[Tokenizer]:
    """Hold the tokenizer for the length of the block: it is never released
    while a lease is held, and the idle clock starts when the last one ends.
    Leases nest (``chunk_markdown`` holds one around its ``count_tokens``).
    The last one out also starts a reaper when the tokenizer has none (a
    ``shutdown_tokenizer`` under a lease, a reaper that died), so a loaded
    tokenizer is always on its way out."""
    global _users, _idle_since
    with _state:
        _users += 1
    try:
        yield _loaded_tokenizer()
    finally:
        with _state:
            _users -= 1
            if _users == 0:
                _idle_since = time.monotonic()
                if _tokenizer is not None:
                    _start_reaper_locked()
                _state.notify_all()


def tokenizer_loaded() -> bool:
    """Whether the e5 tokenizer is in memory now."""
    with _state:
        return _tokenizer is not None


def _release_if_idle_locked(now: float | None) -> bool:
    global _tokenizer
    if _tokenizer is None or _users > 0:
        return False
    moment = time.monotonic() if now is None else now
    if moment - _idle_since < TOKENIZER_IDLE_SECONDS:
        return False
    _tokenizer = None
    return True


def release_if_idle(now: float | None = None) -> bool:
    """Release the tokenizer once unused for ``TOKENIZER_IDLE_SECONDS``.

    ``now`` (a ``time.monotonic`` value) exists for tests; the reaper passes
    nothing. Never releases under a lease. Returns whether it released.
    """
    with _state:
        released = _release_if_idle_locked(now)
    if released:
        _return_freed_memory()
    return released


def _start_reaper_locked() -> None:
    global _reaper, _reaper_stop
    if _reaper is not None:
        return
    _reaper_stop = threading.Event()
    _reaper = threading.Thread(
        target=_reaper_loop, args=(_reaper_stop,), name="e5-tokenizer-reaper", daemon=True
    )
    _reaper.start()


def _reaper_loop(stop: threading.Event) -> None:
    """Wait out the idle delay, release, exit: a reaper lives only while the
    tokenizer is loaded, and the next load starts a new one."""
    global _reaper
    try:
        with _state:
            released = False
            while not stop.is_set() and _tokenizer is not None:
                if _release_if_idle_locked(None):
                    released = True
                    break
                if _users > 0:
                    _state.wait()
                else:
                    _state.wait(max(0.0, _idle_since + TOKENIZER_IDLE_SECONDS - time.monotonic()))
            if _reaper is threading.current_thread():
                _reaper = None
        if released:
            _return_freed_memory()
            logger.info(f"e5 tokenizer released after {TOKENIZER_IDLE_SECONDS:.0f} s unused")
    except Exception as exc:
        # Nobody joins this thread: this record is the only trace. The
        # tokenizer stays loaded until the next call ends and starts a new
        # reaper.
        logger.error(f"e5 tokenizer idle release failed: {exc}", exc_info=True)
        with _state:
            if _reaper is threading.current_thread():
                _reaper = None


def shutdown_tokenizer() -> None:
    """Stop the reaper and release the tokenizer (lifespan shutdown).

    A call still holding it keeps it: when that call ends, a new reaper
    starts and releases it after the idle delay. A later call loads it again
    and starts a new reaper. Idempotent.
    """
    global _reaper, _reaper_stop, _tokenizer
    with _state:
        reaper, stop = _reaper, _reaper_stop
        _reaper = None
        _reaper_stop = None
        if stop is not None:
            stop.set()
        released = _tokenizer is not None and _users == 0
        if released:
            _tokenizer = None
        _state.notify_all()
    if reaper is not None and reaper is not threading.current_thread():
        reaper.join(timeout=5)
    if released:
        _return_freed_memory()


def _allocator_relief() -> Callable[[], object] | None:
    """The libc call that hands the heap pages ``free()`` kept back to the OS.

    Freed memory stays in the allocator's zones, and in the footprint, until
    then: measured on macOS, 290 MB loaded, 74 MB once released, 50 MB (the
    footprint before the load) after ``malloc_zone_pressure_relief``. glibc's
    equivalent is ``malloc_trim(0)``. Windows gets no call: no documented
    call trims what the native allocations leave, and its heap decommits
    large freed blocks itself. ``None`` when the library or the symbol is
    missing (musl has no ``malloc_trim``).
    """
    try:
        if sys.platform == "darwin":
            relief = ctypes.CDLL("/usr/lib/libSystem.B.dylib").malloc_zone_pressure_relief
            relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            relief.restype = ctypes.c_size_t
            return lambda: relief(None, 0)
        if sys.platform.startswith("linux"):
            trim = ctypes.CDLL("libc.so.6").malloc_trim
            trim.argtypes = [ctypes.c_size_t]
            trim.restype = ctypes.c_int
            return lambda: trim(0)
    except (OSError, AttributeError):
        # No such library or symbol: the memory is freed all the same, and
        # the allocator returns it to the OS on its own schedule.
        return None
    return None


def _return_freed_memory() -> None:
    """Collect, then ask the allocator to give the freed pages back."""
    gc.collect()
    relief = _allocator_relief()
    if relief is not None:
        relief()


def _load_bound_pipeline() -> tuple[Tokenizer, tuple[str, ...]]:
    """The frozen tokenizer's normalizer and pre-tokenizer over a one-word
    model, and its added tokens' literals, read from the head of the file:
    they precede the vocabulary there (about 320 kB of the 7.9 MB of JSON)."""
    from tokenizers import Tokenizer

    head = ""
    with gzip.open(E5_TOKENIZER_FILE, "rt", encoding="utf-8") as stream:
        while '"model"' not in head:
            block = stream.read(1 << 16)
            if not block:
                raise ValueError(f"{E5_TOKENIZER_FILE.name} has no model section")
            head += block
    decoder = json.JSONDecoder()
    sections = {}
    for key in ("added_tokens", "normalizer", "pre_tokenizer"):
        start = head.index(":", head.index(f'"{key}"')) + 1
        while head[start].isspace():
            start += 1
        sections[key], _ = decoder.raw_decode(head, start)
    pipeline = Tokenizer.from_str(
        json.dumps(
            {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [],
                "normalizer": sections["normalizer"],
                "pre_tokenizer": sections["pre_tokenizer"],
                "post_processor": None,
                "decoder": None,
                "model": {"type": "WordLevel", "vocab": {"<unk>": 0}, "unk_token": "<unk>"},
            }
        )
    )
    return pipeline, tuple(token["content"] for token in sections["added_tokens"])


def _token_upper_bound(text: str) -> int | None:
    """An upper bound on ``text``'s e5 token count, read without the vocabulary.

    The frozen tokenizer splits out its added (special) tokens, normalizes
    (``Precompiled``), pre-tokenizes (``WhitespaceSplit``, then ``Metaspace``
    prefixing each word with ``▁``), and runs the Unigram model on each
    piece. Every Unigram token covers at least one character of its piece:
    no vocabulary entry is empty and ``byte_fallback`` is off, so an unknown
    character becomes one ``<unk>`` covering it
    (``tests/test_e5_tokenizer_lifecycle.py`` pins both). The token count is
    therefore at most the characters of the pieces, which the same normalizer
    and pre-tokenizer compute here; a character the normalizer expands
    (U+FDFA becomes 18) is counted expanded. A text holding an added token's
    literal (``<s>``, ``<mask>``, ...) gets ``None``: that split precedes the
    normalizer and is not modelled.
    """
    global _bound_pipeline, _bound_specials
    pipeline = _bound_pipeline
    if pipeline is None:
        with _bound_lock:
            if _bound_pipeline is None:
                _bound_pipeline, _bound_specials = _load_bound_pipeline()
            pipeline = _bound_pipeline
    if any(special in text for special in _bound_specials):
        return None
    normalized = pipeline.normalizer.normalize_str(text)
    if any(special in normalized for special in _bound_specials):
        return None
    return sum(len(piece) for piece, _ in pipeline.pre_tokenizer.pre_tokenize_str(normalized))


def count_tokens(text: str) -> int:
    """Real e5 subword count (no special tokens) — never a chars/N estimate."""
    with _tokenizer_lease() as tokenizer:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)


def truncate_to_tokens(text: str, max_tokens: int = E5_MAX_CONTENT_TOKENS) -> str:
    """The longest prefix of ``text`` holding at most ``max_tokens`` e5 tokens.

    Bounds what the dedicated embedding child receives (#594): the frozen
    tokenizer's offsets give the character where token ``max_tokens`` starts,
    and the text is cut there — a plain prefix of the original, never a
    decode round-trip (decoding re-spaces subwords and would alter the text).
    A text already within the bound is returned unchanged, identity included,
    and one whose ``_token_upper_bound`` is within it is returned without
    loading the tokenizer: a short query never loads it.
    The guarantee is the BOUND, not boundary identity: the cut prefix can
    re-tokenize to a different final token than an id-level truncation (see
    ``E5_MAX_CONTENT_TOKENS``); mean pooling over ~510 tokens keeps the
    vectors within cosine 0.9999 either way.
    """
    bound = _token_upper_bound(text)
    if bound is not None and bound <= max_tokens:
        return text
    with _tokenizer_lease() as tokenizer:
        encoding = tokenizer.encode(text, add_special_tokens=False)
    if len(encoding.ids) <= max_tokens:
        return text
    return text[: encoding.offsets[max_tokens][0]]


@dataclass(slots=True)
class Chunk:
    """One retrieval unit, ready for ``build_embedding_text`` + embedding."""

    chunk_index: int
    text: str
    token_count: int
    page_number: int | None = None


def _heading_breadcrumb(metadata: dict[str, str]) -> str:
    parts: list[str] = []
    for key in ("h1", "h2", "h3", "h4"):
        if key in metadata:
            level = int(key[1])
            parts.append(f"{'#' * level} {metadata[key]}")
    return " > ".join(parts)


def _reattach_table_headers(chunks: list[str]) -> list[str]:
    """Re-inject the Markdown table header into sub-chunks that lost it."""
    if not chunks:
        return chunks

    table_header: str | None = None
    result: list[str] = []

    for chunk in chunks:
        lines = chunk.split("\n")

        # Scan for a table header (header row + separator row) in this chunk.
        found_header = False
        for i in range(len(lines) - 1):
            line = lines[i].strip()
            next_line = lines[i + 1].strip()
            if (
                line.startswith("|")
                and line.endswith("|")
                and next_line.startswith("|")
                and "---" in next_line
            ):
                table_header = lines[i] + "\n" + lines[i + 1]
                found_header = True
                break

        if found_header:
            result.append(chunk)
            continue

        first_content = next((ln.strip() for ln in lines if ln.strip()), "")

        if first_content.startswith("|") and table_header is not None:
            # Headerless continuation of the running table.
            prefix_lines: list[str] = []
            rest_lines: list[str] = []
            in_prefix = True
            for line in lines:
                if in_prefix and not line.strip().startswith("|"):
                    prefix_lines.append(line)
                else:
                    in_prefix = False
                    rest_lines.append(line)
            if prefix_lines:
                result.append(
                    "\n".join(prefix_lines) + "\n" + table_header + "\n" + "\n".join(rest_lines)
                )
            else:
                result.append(table_header + "\n" + chunk)
        else:
            if not first_content.startswith("|"):
                table_header = None
            result.append(chunk)

    return result


def _make_sub_splitter(target_tokens: int, overlap_tokens: int) -> RecursiveCharacterTextSplitter:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Measured with count_tokens: from_huggingface_tokenizer only takes a
    # transformers tokenizer, and counted len(tokenize(text)), the same number.
    return RecursiveCharacterTextSplitter(
        chunk_size=target_tokens,
        chunk_overlap=overlap_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=count_tokens,
    )


def chunk_markdown(
    text: str,
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    page_number: int | None = None,
    start_index: int = 0,
) -> list[Chunk]:
    """Run the 3-pass chunker over one Markdown text. See module docstring."""
    if not text or not text.strip():
        return []

    # PostgreSQL text columns reject NUL bytes; some parsers emit them.
    text = text.replace("\x00", "")

    from langchain_text_splitters import MarkdownHeaderTextSplitter

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    sub_splitter = _make_sub_splitter(target_tokens, overlap_tokens)

    chunks: list[Chunk] = []
    # One lease for the whole text: its many count_tokens calls share one load.
    with _tokenizer_lease():
        for section in md_splitter.split_text(text):
            breadcrumb = _heading_breadcrumb(section.metadata)

            sub_texts = sub_splitter.split_text(section.page_content)
            sub_texts = _reattach_table_headers(sub_texts)

            for sub_text in sub_texts:
                chunk_text = f"{breadcrumb}\n\n{sub_text}" if breadcrumb else sub_text

                chunks.append(
                    Chunk(
                        chunk_index=start_index + len(chunks),
                        text=chunk_text,
                        token_count=count_tokens(chunk_text),
                        page_number=page_number,
                    )
                )

    return chunks


def chunk_document(
    document: ExtractedDocument,
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk an ``ExtractedDocument`` — page by page when paginated (PDF),
    so each chunk carries its ``page_number``; whole-Markdown otherwise.

    ``pending_vision`` documents yield no chunks (indexed by the OCR/VLM
    tiers of a later release).
    """
    if document.status != "active":
        return []

    if document.pages:
        chunks: list[Chunk] = []
        for page in document.pages:
            chunks.extend(
                chunk_markdown(
                    page.text,
                    target_tokens=target_tokens,
                    overlap_tokens=overlap_tokens,
                    page_number=page.page_number,
                    start_index=len(chunks),
                )
            )
    else:
        chunks = chunk_markdown(
            document.markdown,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        )

    if chunks:
        avg_tokens = sum(chunk.token_count for chunk in chunks) / len(chunks)
        logger.info(
            f"Document chunked: {len(chunks)} chunks, "
            f"avg {avg_tokens:.0f} tokens/chunk "
            f"(extractor={document.metadata.get('extractor', '?')})"
        )
    else:
        logger.debug(
            f"Document chunked: 0 chunks " f"(extractor={document.metadata.get('extractor', '?')})"
        )
    return chunks

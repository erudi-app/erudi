"""Reference e5 vectors for the suite's Knowledge Base tests (#594).

The app embeds through a dedicated child process (``embedder_child``), which
backend-ci cannot run on every leg. The suite therefore embeds through
:class:`ReferenceE5Embeddings`: the production ``ChildE5Embeddings`` (its
``passage: ``/``query: `` prefixes and its 512-token bound included) with
only the transport replaced by a lookup in
``tests/fixtures/e5_reference_vectors.json`` — real multilingual-e5-small
vectors, keyed by the exact text the child would have received.

Two modes:

- **lenient** (the default): a text outside the file gets a deterministic,
  hash-seeded unit vector. Enough for a test that indexes and reads back
  without asserting what the vectors mean.
- **strict** (``strict_e5_reference`` fixture): a text outside the file is a
  hard failure naming the text. A test that asserts e5 semantics (ranking,
  golden queries, calibrated similarities, parity with a real child) runs
  strict, so it can never pass on hash vectors.

Recording: ``pytest <files> --record-e5-reference`` embeds every strict miss
through the REAL platform child (the e5 snapshot's ``mlx_vlm`` child on
macOS, the bundled GGUF's ``llama-server`` on Windows/Linux) and merges them
into the file at the end of the session. Record on macOS: the snapshot child
serves the fp32 model, the GGUF is Q8_0.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

from src.ingestion.embeddings import EMBEDDING_DIMENSIONS, ChildE5Embeddings

REFERENCE_FILE = Path(__file__).resolve().parent / "fixtures" / "e5_reference_vectors.json"

#: The texts the real-child parity tests embed (``test_embedder_child_real``
#: on macOS, ``test_llama_embedder_real`` on Windows/Linux): one set, so both
#: children are held to the same reference vectors. The last one runs past
#: the 512-token window.
PARITY_DOCUMENTS = [
    "Paris est la capitale de la France.",
    "[document_name:rapport.pdf]\n# Introduction > ## Contexte\n\nLe projet vise "
    "à réduire la consommation énergétique des bâtiments publics de 30 %.",
    "[document_name:table.md]\n| Model | Dimensions |\n|---|---|\n| e5-small | 384 |",
    "[document_name:nihongo.md]\n日本語のテキストを埋め込みます。",
    "The quarterly report shows a sharp increase in customer retention.",
    "Phrase utile numéro un, qui décrit un fait précis sur le produit. " * 120,
]
PARITY_QUERIES = [
    "Quelle est la capitale de la France ?",
    "how does the knowledge base work?",
]


class MissingReferenceVector(AssertionError):
    """A strict test embedded a text the reference file does not hold."""


class _Store:
    """The loaded file, the mode, and what a recording session collected."""

    def __init__(self) -> None:
        self.vectors: dict[str, list[float]] = {}
        self.loaded = False
        self.strict = False
        self.record = False
        self.recorded: dict[str, list[float]] = {}

    def load(self) -> None:
        if self.loaded:
            return
        if REFERENCE_FILE.exists():
            payload = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
            self.vectors = {entry["text"]: entry["embedding"] for entry in payload["texts"]}
        self.loaded = True


STORE = _Store()


def fallback_vector(text: str) -> list[float]:
    """A deterministic unit vector for a text outside the reference file."""
    raw: list[float] = []
    counter = 0
    seed = text.encode("utf-8")
    while len(raw) < EMBEDDING_DIMENSIONS:
        digest = hashlib.sha256(seed + struct.pack("<I", counter)).digest()
        raw.extend(b / 255.0 - 0.5 for b in digest)
        counter += 1
    raw = raw[:EMBEDDING_DIMENSIONS]
    norm = math.sqrt(sum(v * v for v in raw)) or 1.0
    return [v / norm for v in raw]


def reference_vectors(texts: list[str]) -> list[list[float]]:
    """The reference vector of each text, in the current mode (see module)."""
    STORE.load()
    missing = [t for t in texts if t not in STORE.vectors and t not in STORE.recorded]
    if missing and STORE.strict:
        if not STORE.record:
            raise MissingReferenceVector(
                f"{len(missing)} text(s) have no reference e5 vector, e.g. {missing[0][:120]!r}. "
                "Record them on macOS: pytest <these test files> --record-e5-reference"
            )
        from src.ingestion import embedder_child

        for text, vector in zip(missing, embedder_child.embed_texts(missing)):
            STORE.recorded[text] = vector
    known = {**STORE.vectors, **STORE.recorded}
    return [known[t] if t in known else fallback_vector(t) for t in texts]


def write_recorded() -> int:
    """Merge the session's recorded vectors into the file; returns how many."""
    if not STORE.recorded:
        return 0
    STORE.load()
    merged = {**STORE.vectors, **STORE.recorded}
    # One entry per line: a re-recording diffs as added/changed texts.
    entries = ",\n".join(
        json.dumps(
            {"text": text, "embedding": [round(v, 7) for v in merged[text]]},
            ensure_ascii=False,
        )
        for text in sorted(merged)
    )
    header = json.dumps(
        {
            "model": "intfloat/multilingual-e5-small",
            "recorded_with": "the app's embedding child (mlx_vlm, fp32 snapshot, macOS)",
        },
        ensure_ascii=False,
    )[:-1]
    REFERENCE_FILE.write_text(f'{header}, "texts": [\n{entries}\n]}}\n', encoding="utf-8")
    return len(STORE.recorded)


class ReferenceE5Embeddings(ChildE5Embeddings):
    """``ChildE5Embeddings`` with the reference vectors as its transport."""

    @staticmethod
    def _embed(texts: list[str]) -> list[list[float]]:
        return reference_vectors(texts)

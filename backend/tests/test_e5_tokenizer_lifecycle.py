"""The e5 tokenizer is loaded on demand and released after an idle delay.

Loaded, the frozen tokenizer holds about 250 MB of the backend's footprint.
It is held only while a call uses it and for ``TOKENIZER_IDLE_SECONDS``
after the last one, then released and its memory handed back to the
operating system. ``truncate_to_tokens`` never loads it for a text whose
token count is provably within the bound (the normalized pre-tokenized
characters bound the Unigram token count), so a short KB question does not
load it at all.
"""

import threading
import time
import types
from pathlib import Path

import pytest

from src.ingestion import chunking

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]

# The parity corpus of tests/test_e5_tokenizer.py, plus texts that stress the
# normalizer: characters NFKC expands (U+FDFA expands to 18 characters, U+337F
# to 4), combining sequences, whitespace runs, and the special-token literals.
CORPUS = [
    "",
    " ",
    "OK",
    "42",
    "€",
    "?",
    "café",
    "Le café des développeurs était fermé, l'équipe a donc travaillé à la maison.",
    "Quel est le préavis de résiliation prévu par le contrat Nimbus ?",
    "The quarterly security audit found no critical vulnerabilities.",
    "日本語のテキストと中文文本混合在一起。数据库迁移在启动时自动执行。",
    "Привет, как дела? Ελληνικά κείμενα. مرحبا بالعالم",
    "emoji 🚀🔥 and  double  spaces\tand tabs\nand\nnewlines\n\n\nand more",
    "def count_tokens(text: str) -> int:\n    return len(tok.encode(text).ids)  # noqa\n",
    "const x = await fetch(`${API_BASE_URL}/erudi/llms`, { method: 'GET' });",
    "| Article | Prix |\n| --- | --- |\n| Produit 1 | 100 € |\n| Produit 2 | 200 € |",
    "# Guide > ## Installation\n\nÉtapes détaillées pour installer le produit.",
    "A literal <mask> token, a <s> start and a </s> end, plus <pad> and <unk>.",
    "FAC-2024-0917 a été réglée le 15/03/2024 à 14h32 (UTC+01:00).",
    " non-breaking space and zero​width and ligature ﬁ and fullwidth ＡＢＣ",
    "Ünïcödé ñ ç ß ø å æ œ — “quotes” ‘single’ … ellipsis",
    "ﷺ" * 40,
    "㍿ ﷺ ﬃ ㎏ ½ ⑳ ẛ̣ é̂̃ 한국어 텍스트",
    "a" * 600,
    " " * 2000 + "x",
    "​" * 700,
    "😀" * 300,
]


def _long_text() -> str:
    return (REPO_ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")


def _corpus() -> list[str]:
    long_text = _long_text()
    texts = list(CORPUS)
    texts += [f"query: {text}" for text in CORPUS]
    texts += [f"passage: {text}" for text in CORPUS]
    # Prefixes of a real document around the default bound, in characters.
    texts += [long_text[:n] for n in (100, 400, 500, 510, 520, 1000, 3000, 20000)]
    return texts


def _reference_truncate(tokenizer, text: str, max_tokens: int) -> str:
    """``truncate_to_tokens`` as it was with the resident tokenizer."""
    encoding = tokenizer.encode(text, add_special_tokens=False)
    if len(encoding.ids) <= max_tokens:
        return text
    return text[: encoding.offsets[max_tokens][0]]


@pytest.fixture(scope="module")
def frozen_tokenizer():
    return chunking._load_frozen_tokenizer()


@pytest.fixture(autouse=True)
def _fresh_tokenizer_state():
    chunking.shutdown_tokenizer()
    yield
    chunking.shutdown_tokenizer()


def _forbid_loading(monkeypatch):
    def boom():
        raise AssertionError("the e5 tokenizer must not be loaded")

    monkeypatch.setattr(chunking, "_load_frozen_tokenizer", boom)


class _BlockingTokenizer:
    """Stands for the tokenizer: ``encode`` parks until ``proceed`` is set."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.proceed = threading.Event()

    def encode(self, text, add_special_tokens=False):
        self.entered.set()
        assert self.proceed.wait(timeout=10)
        return types.SimpleNamespace(ids=[0] * len(text.split()), offsets=[])


def _released_within(seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while chunking.tokenizer_loaded() and time.monotonic() < deadline:
        time.sleep(0.01)
    return not chunking.tokenizer_loaded()


class _CountingLoader:
    def __init__(self, factory) -> None:
        self.factory = factory
        self.loads = 0

    def __call__(self):
        self.loads += 1
        return self.factory()


class TestShortCircuit:
    def test_a_short_query_does_not_load_the_tokenizer(self, monkeypatch):
        _forbid_loading(monkeypatch)
        text = "query: Quel est le préavis de résiliation prévu par le contrat Nimbus ?"
        assert chunking.truncate_to_tokens(text) is text
        assert not chunking.tokenizer_loaded()

    def test_a_text_near_the_bound_in_characters_does_not_load_it(self, monkeypatch):
        _forbid_loading(monkeypatch)
        text = "query: " + "mot " * 120
        assert chunking.truncate_to_tokens(text) is text
        assert not chunking.tokenizer_loaded()

    def test_the_bound_never_undercounts(self, frozen_tokenizer):
        for text in _corpus():
            count = len(frozen_tokenizer.encode(text, add_special_tokens=False).ids)
            bound = chunking._token_upper_bound(text)
            if bound is None:
                assert any(special in text for special in ("<s>", "</s>", "<mask>"))
                continue
            assert bound >= count, (text[:60], bound, count)

    def test_the_bound_holds_across_the_unicode_planes(self, frozen_tokenizer):
        # Every assigned-or-not code point of planes 0-2, 64 at a time.
        points = [cp for cp in range(0x30000) if not 0xD800 <= cp <= 0xDFFF]
        for start in range(0, len(points), 64):
            text = "".join(chr(cp) for cp in points[start : start + 64])
            bound = chunking._token_upper_bound(text)
            if bound is None:
                continue
            count = len(frozen_tokenizer.encode(text, add_special_tokens=False).ids)
            assert bound >= count, (hex(points[start]), bound, count)

    def test_the_proof_premises_hold_on_the_frozen_file(self):
        # What _token_upper_bound's proof rests on: a Unigram model without
        # byte fallback and without an empty piece, after a Precompiled
        # normalizer and the WhitespaceSplit + Metaspace pre-tokenizer.
        import gzip
        import json

        frozen = json.loads(gzip.decompress(chunking.E5_TOKENIZER_FILE.read_bytes()))
        model = frozen["model"]
        assert model["type"] == "Unigram"
        assert model["byte_fallback"] is False
        assert min(len(piece) for piece, _ in model["vocab"]) >= 1
        assert frozen["normalizer"]["type"] == "Precompiled"
        assert [step["type"] for step in frozen["pre_tokenizer"]["pretokenizers"]] == [
            "WhitespaceSplit",
            "Metaspace",
        ]
        pipeline, specials = chunking._load_bound_pipeline()
        assert specials == tuple(token["content"] for token in frozen["added_tokens"])
        assert pipeline.normalizer.normalize_str("ﬁ ＡＢＣ") == "fi ABC"

    def test_a_special_token_literal_is_never_short_circuited(self):
        # Matched before normalization, a special token is outside what the
        # bound models: the text goes to the real tokenizer.
        assert chunking._token_upper_bound("a <mask> b") is None
        assert chunking._token_upper_bound("</s>") is None

    def test_a_long_text_loads_it_and_truncates_as_before(self, frozen_tokenizer):
        long_text = "query: " + _long_text()
        truncated = chunking.truncate_to_tokens(long_text)
        assert chunking.tokenizer_loaded()
        assert truncated == _reference_truncate(frozen_tokenizer, long_text, 510)
        assert len(truncated) < len(long_text)

    @pytest.mark.parametrize("max_tokens", [1, 5, 17, 50, 180, 510])
    def test_output_is_identical_to_the_resident_implementation(self, frozen_tokenizer, max_tokens):
        for text in _corpus():
            expected = _reference_truncate(frozen_tokenizer, text, max_tokens)
            assert chunking.truncate_to_tokens(text, max_tokens=max_tokens) == expected


class TestIdleRelease:
    def test_released_after_the_idle_delay_and_reloaded_on_next_use(self, monkeypatch):
        loader = _CountingLoader(chunking._load_frozen_tokenizer)
        monkeypatch.setattr(chunking, "_load_frozen_tokenizer", loader)
        assert chunking.count_tokens("Bonjour le monde") == 3
        assert chunking.tokenizer_loaded()
        start = time.monotonic()
        assert not chunking.release_if_idle(now=start + chunking.TOKENIZER_IDLE_SECONDS - 5)
        assert chunking.tokenizer_loaded()
        assert chunking.release_if_idle(now=start + chunking.TOKENIZER_IDLE_SECONDS + 5)
        assert not chunking.tokenizer_loaded()
        assert chunking.count_tokens("Bonjour le monde") == 3
        assert loader.loads == 2

    def test_a_burst_of_calls_loads_it_once(self, monkeypatch):
        loader = _CountingLoader(chunking._load_frozen_tokenizer)
        monkeypatch.setattr(chunking, "_load_frozen_tokenizer", loader)
        for i in range(200):
            chunking.count_tokens(f"Phrase numéro {i}.")
        chunking.chunk_markdown("# Titre\n\n" + "Un paragraphe utile. " * 400)
        assert loader.loads == 1

    def test_never_released_while_a_call_holds_it(self, monkeypatch):
        fake = _BlockingTokenizer()
        monkeypatch.setattr(chunking, "_load_frozen_tokenizer", lambda: fake)
        result = {}
        worker = threading.Thread(target=lambda: result.update(n=chunking.count_tokens("a b c")))
        worker.start()
        assert fake.entered.wait(timeout=10)
        far = time.monotonic() + 10 * chunking.TOKENIZER_IDLE_SECONDS
        assert not chunking.release_if_idle(now=far)
        assert chunking.tokenizer_loaded()
        fake.proceed.set()
        worker.join(timeout=10)
        assert result == {"n": 3}
        assert chunking.release_if_idle(now=time.monotonic() + 10 * chunking.TOKENIZER_IDLE_SECONDS)
        assert not chunking.tokenizer_loaded()

    def test_the_idle_clock_restarts_when_the_last_call_ends(self, monkeypatch):
        # A simulated clock: the call holds the tokenizer 50 s, so an idle
        # clock stamped at the lease's entry would release it 50 s early.
        clock = [1000.0]
        monkeypatch.setattr(chunking, "time", types.SimpleNamespace(monotonic=lambda: clock[0]))
        fake = _BlockingTokenizer()
        monkeypatch.setattr(chunking, "_load_frozen_tokenizer", lambda: fake)
        worker = threading.Thread(target=lambda: chunking.count_tokens("a"))
        worker.start()
        assert fake.entered.wait(timeout=10)
        clock[0] = 1050.0
        fake.proceed.set()
        worker.join(timeout=10)
        idle = chunking.TOKENIZER_IDLE_SECONDS
        assert chunking._idle_since == 1050.0
        assert not chunking.release_if_idle(now=1000.0 + idle + 1)
        assert chunking.release_if_idle(now=1050.0 + idle + 1)

    def test_the_reaper_thread_releases_it(self, monkeypatch):
        monkeypatch.setattr(chunking, "TOKENIZER_IDLE_SECONDS", 0.05)
        released = threading.Event()
        real_release = chunking._return_freed_memory
        monkeypatch.setattr(
            chunking, "_return_freed_memory", lambda: (real_release(), released.set())
        )
        chunking.count_tokens("Bonjour le monde")
        reaper = chunking._reaper
        assert reaper is not None and reaper.is_alive()
        assert released.wait(timeout=10)
        reaper.join(timeout=10)
        assert not reaper.is_alive()
        assert not chunking.tokenizer_loaded()

    def test_shutdown_stops_the_reaper_and_releases_it(self):
        chunking.count_tokens("Bonjour le monde")
        reaper = chunking._reaper
        assert reaper is not None and reaper.is_alive()
        chunking.shutdown_tokenizer()
        assert not reaper.is_alive()
        assert chunking._reaper is None
        assert not chunking.tokenizer_loaded()

    def test_shutdown_never_releases_under_a_call(self, monkeypatch):
        fake = _BlockingTokenizer()
        monkeypatch.setattr(chunking, "_load_frozen_tokenizer", lambda: fake)
        worker = threading.Thread(target=lambda: chunking.count_tokens("a b"))
        worker.start()
        assert fake.entered.wait(timeout=10)
        chunking.shutdown_tokenizer()
        assert chunking.tokenizer_loaded()
        fake.proceed.set()
        worker.join(timeout=10)

    def test_a_tokenizer_left_loaded_by_a_shutdown_is_still_released(self, monkeypatch):
        # shutdown_tokenizer under a lease stops the reaper but cannot release:
        # the lease's end starts a new reaper, which releases after the delay.
        monkeypatch.setattr(chunking, "TOKENIZER_IDLE_SECONDS", 0.05)
        fake = _BlockingTokenizer()
        monkeypatch.setattr(chunking, "_load_frozen_tokenizer", lambda: fake)
        worker = threading.Thread(target=lambda: chunking.count_tokens("a b"))
        worker.start()
        assert fake.entered.wait(timeout=10)
        chunking.shutdown_tokenizer()
        assert chunking._reaper is None
        assert chunking.tokenizer_loaded()
        fake.proceed.set()
        worker.join(timeout=10)
        assert _released_within(10)

    def test_a_reaper_that_died_is_replaced_at_the_next_call(self, monkeypatch):
        monkeypatch.setattr(chunking, "TOKENIZER_IDLE_SECONDS", 3600.0)
        chunking.count_tokens("Bonjour le monde")
        # A reaper that died: the tokenizer loaded, no reaper left.
        with chunking._state:
            reaper, stop = chunking._reaper, chunking._reaper_stop
            chunking._reaper = None
            chunking._reaper_stop = None
        stop.set()
        with chunking._state:
            chunking._state.notify_all()
        reaper.join(timeout=10)
        assert chunking.tokenizer_loaded()
        monkeypatch.setattr(chunking, "TOKENIZER_IDLE_SECONDS", 0.05)
        chunking.count_tokens("encore")
        assert chunking._reaper is not None
        assert _released_within(10)


class TestAllocatorRelief:
    def test_missing_library_is_a_no_op(self, monkeypatch):
        def no_library(*args, **kwargs):
            raise OSError("no such library")

        monkeypatch.setattr(chunking.ctypes, "CDLL", no_library)
        assert chunking._allocator_relief() is None
        chunking._return_freed_memory()

    def test_missing_symbol_is_a_no_op(self, monkeypatch):
        monkeypatch.setattr(chunking.ctypes, "CDLL", lambda *a, **k: types.SimpleNamespace())
        assert chunking._allocator_relief() is None
        chunking._return_freed_memory()

    def test_no_call_on_windows(self, monkeypatch):
        monkeypatch.setattr(chunking.sys, "platform", "win32")
        assert chunking._allocator_relief() is None

    @pytest.mark.parametrize(
        "platform_name, symbol",
        [("darwin", "malloc_zone_pressure_relief"), ("linux", "malloc_trim")],
    )
    def test_the_platform_symbol_is_called(self, monkeypatch, platform_name, symbol):
        calls = []

        class _Function:
            def __call__(self, *args):
                calls.append(args)
                return 0

        monkeypatch.setattr(chunking.sys, "platform", platform_name)
        monkeypatch.setattr(
            chunking.ctypes, "CDLL", lambda *a, **k: types.SimpleNamespace(**{symbol: _Function()})
        )
        relief = chunking._allocator_relief()
        assert relief is not None
        relief()
        assert len(calls) == 1

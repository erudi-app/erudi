"""Measured sizes on Hugging Face search hits (#761).

A search hit used to carry only ``param_size``, so its card showed the
``param_size x 0.6`` estimate, up to 27 % below what the download installs.
After the hits are filtered, each one is enriched with ``artifact_size_bytes``:
one ``model_info(files_metadata=True)`` request per hit, the downloader's own
file selection applied to the listing, in a small pool, under a short global
budget, fail-soft, stopping at the first 429. HF is faked: no network.
"""

import threading
import time
from types import SimpleNamespace

import pytest

from src.domains.llms import hf_search


def _hit(mid, downloads=1000):
    return SimpleNamespace(
        id=mid,
        downloads=downloads,
        likes=3,
        gated=False,
        pipeline_tag="text-generation",
        tags=[],
        safetensors=None,
    )


def _info(files):
    """A ``model_info(files_metadata=True)`` answer: one sibling per file."""
    return SimpleNamespace(siblings=[SimpleNamespace(rfilename=n, size=s) for n, s in files])


class _MlxEngine:
    FORMAT_TAG = "mlx"
    USES_GGUF = False

    @staticmethod
    def is_runnable(link):
        return True


class _GgufEngine(_MlxEngine):
    FORMAT_TAG = "gguf"
    USES_GGUF = True


class _Status(Exception):
    """An HTTP error the way huggingface_hub raises one: ``.response.status_code``."""

    def __init__(self, status, headers=None):
        super().__init__(f"{status} Client Error")
        self.response = SimpleNamespace(status_code=status, headers=headers or {})


class _FakeApi:
    """``list_models`` returns ``hits``; ``model_info`` answers from ``infos``
    (a value, an exception to raise, or a callable taking the repo id)."""

    def __init__(self, hits, infos):
        self._hits = hits
        self._infos = infos
        self.calls = []
        self.kwargs = []
        self._lock = threading.Lock()

    def list_models(self, **k):
        return list(self._hits)

    def model_info(self, repo_id, **k):
        with self._lock:
            self.calls.append(repo_id)
            self.kwargs.append(k)
        answer = self._infos.get(repo_id)
        if callable(answer):
            answer = answer(repo_id)
        if isinstance(answer, BaseException):
            raise answer
        if answer is None:
            raise _Status(404)
        return answer


@pytest.fixture(autouse=True)
def _empty_size_cache(monkeypatch):
    hf_search._size_cache.clear()
    monkeypatch.setattr(hf_search, "_rate_limited_until", 0.0)
    yield
    hf_search._size_cache.clear()


class _Clock:
    """An injected ``hf_search._clock``: time moves only when the test says so."""

    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(hf_search, "_clock", c)
    return c


def _use(monkeypatch, api, engine=_MlxEngine):
    monkeypatch.setattr(hf_search.config, "LLM_Engine", engine, raising=False)
    monkeypatch.setattr(hf_search.config, "get_hf_api", lambda: api)


@pytest.mark.unit
class TestSearchHitSizes:
    def test_sums_the_files_a_download_fetches(self, monkeypatch):
        api = _FakeApi(
            [_hit("mlx-community/Qwen2.5-VL-3B-Instruct-4bit")],
            {
                "mlx-community/Qwen2.5-VL-3B-Instruct-4bit": _info(
                    [
                        ("model.safetensors", 3_089_000_000),
                        ("config.json", 700_000),
                        ("tokenizer.json", 13_215),
                    ]
                )
            },
        )
        _use(monkeypatch, api)

        (r,) = hf_search.search_huggingface("qwen vl", limit=5)
        assert r["artifact_size_bytes"] == 3_089_713_215

    def test_asks_for_file_sizes_without_the_retry_ladder(self, monkeypatch):
        """``repo_info`` delegates to the retrying ``model_info`` override, whose
        default ladder (5 retries, 30 s backoff) would outlive the search; the
        enrichment calls ``model_info`` directly with no retry and a short timeout."""
        api = _FakeApi([_hit("org/a-4bit")], {"org/a-4bit": _info([("w.safetensors", 10)])})
        _use(monkeypatch, api)

        hf_search.search_huggingface("a", limit=5)
        (k,) = api.kwargs
        assert k["files_metadata"] is True
        assert k["_max_retries"] == 0
        assert 0 < k["timeout"] <= hf_search._SIZE_BUDGET_SECONDS

    def test_multi_quant_gguf_counts_only_the_chosen_quant(self, monkeypatch):
        api = _FakeApi(
            [_hit("bartowski/Qwen2.5-7B-Instruct-GGUF")],
            {
                "bartowski/Qwen2.5-7B-Instruct-GGUF": _info(
                    [
                        ("Qwen2.5-7B-Instruct-Q2_K.gguf", 3_000_000_000),
                        ("Qwen2.5-7B-Instruct-Q4_K_M.gguf", 4_680_000_000),
                        ("Qwen2.5-7B-Instruct-Q8_0.gguf", 8_100_000_000),
                        ("README.md", 5_000),
                    ]
                )
            },
        )
        _use(monkeypatch, api, engine=_GgufEngine)

        (r,) = hf_search.search_huggingface("qwen", limit=5)
        assert r["artifact_size_bytes"] == 4_680_000_000 + 5_000

    def test_measuring_a_gguf_hit_logs_no_download_pick(self, monkeypatch, caplog):
        """The selection runs silently: a search is not a download, and a repo
        without a preferred quant must not put a WARNING in Diagnostics."""
        api = _FakeApi(
            [_hit("org/odd-GGUF")],
            {"org/odd-GGUF": _info([("odd-IQ2_XXS-2.56bpw.gguf", 900), ("README.md", 1)])},
        )
        _use(monkeypatch, api, engine=_GgufEngine)

        with caplog.at_level("DEBUG", logger="erudi"):
            (r,) = hf_search.search_huggingface("odd", limit=5)
        assert r["artifact_size_bytes"] == 901
        assert not [m for m in caplog.messages if "GGUF" in m or "quant" in m]
        assert not [r for r in caplog.records if r.levelname != "INFO" and r.levelname != "DEBUG"]

    def test_an_incomplete_split_is_not_a_failed_request(self, monkeypatch, caplog):
        api = _FakeApi(
            [_hit("org/split-GGUF")],
            {"org/split-GGUF": _info([("m-Q4_K_M-00001-of-00003.gguf", 5)])},
        )
        _use(monkeypatch, api, engine=_GgufEngine)

        with caplog.at_level("DEBUG", logger="erudi"):
            (r,) = hf_search.search_huggingface("split", limit=5)
        assert r["artifact_size_bytes"] is None
        assert not [r for r in caplog.records if r.levelname in ("WARNING", "ERROR")]

    def test_a_failed_request_leaves_none_and_keeps_every_hit(self, monkeypatch):
        api = _FakeApi(
            [_hit("org/a-4bit"), _hit("org/gated-4bit"), _hit("org/c-4bit")],
            {
                "org/a-4bit": RuntimeError("connection reset"),
                "org/gated-4bit": _Status(401),
                "org/c-4bit": _info([("w.safetensors", 42)]),
            },
        )
        _use(monkeypatch, api)

        res = hf_search.search_huggingface("x", limit=5)
        sizes = {r["link"]: r["artifact_size_bytes"] for r in res}
        assert sizes == {"org/a-4bit": None, "org/gated-4bit": None, "org/c-4bit": 42}

    def test_budget_exceeded_leaves_the_rest_none_without_raising(self, monkeypatch):
        monkeypatch.setattr(hf_search, "_SIZE_BUDGET_SECONDS", 0.3)
        release = threading.Event()

        def slow(_):
            release.wait(5)
            return _info([("w.safetensors", 1)])

        api = _FakeApi(
            [_hit("org/fast-4bit"), _hit("org/slow-4bit")],
            {"org/fast-4bit": _info([("w.safetensors", 7)]), "org/slow-4bit": slow},
        )
        _use(monkeypatch, api)

        started = time.monotonic()
        try:
            res = hf_search.search_huggingface("x", limit=5)
        finally:
            release.set()
        assert time.monotonic() - started < 2
        sizes = {r["link"]: r["artifact_size_bytes"] for r in res}
        assert sizes == {"org/fast-4bit": 7, "org/slow-4bit": None}

    def test_a_429_stops_every_further_request(self, monkeypatch):
        monkeypatch.setattr(hf_search, "_SIZE_WORKERS", 1)  # deterministic order
        hits = [_hit(f"org/m{i}-4bit") for i in range(6)]
        infos = {h.id: _info([("w.safetensors", 1)]) for h in hits}
        infos["org/m1-4bit"] = _Status(429)
        api = _FakeApi(hits, infos)
        _use(monkeypatch, api)

        res = hf_search.search_huggingface("x", limit=10)
        assert api.calls == ["org/m0-4bit", "org/m1-4bit"]
        assert [r["artifact_size_bytes"] for r in res] == [1, None, None, None, None, None]

    def test_a_429_is_a_warning_never_an_error(self, monkeypatch, caplog):
        monkeypatch.setattr(hf_search, "_SIZE_WORKERS", 1)
        api = _FakeApi([_hit("org/m0-4bit")], {"org/m0-4bit": _Status(429)})
        _use(monkeypatch, api)

        with caplog.at_level("DEBUG", logger="erudi"):
            hf_search.search_huggingface("x", limit=10)
        assert not [r for r in caplog.records if r.levelname in ("ERROR", "CRITICAL")]

    def test_never_more_requests_than_limit(self, monkeypatch):
        hits = [_hit(f"org/m{i}-4bit") for i in range(12)]
        api = _FakeApi(hits, {h.id: _info([("w.safetensors", 1)]) for h in hits})
        _use(monkeypatch, api)

        res = hf_search.search_huggingface("x", limit=5)
        assert len(res) == 5
        assert len(api.calls) == 5
        assert sorted(api.calls) == sorted(r["link"] for r in res)

    def test_a_cached_size_makes_no_request(self, monkeypatch):
        api = _FakeApi([_hit("org/a-4bit")], {"org/a-4bit": _info([("w.safetensors", 9)])})
        _use(monkeypatch, api)

        first = hf_search.search_huggingface("a", limit=5)
        second = hf_search.search_huggingface("a", limit=5)
        assert api.calls == ["org/a-4bit"]
        assert first[0]["artifact_size_bytes"] == second[0]["artifact_size_bytes"] == 9

    def test_the_cache_expires_and_is_bounded(self, monkeypatch, clock):
        monkeypatch.setattr(hf_search, "_SIZE_CACHE_MAX", 2)
        hf_search._cache_put("org/a", 1)
        hf_search._cache_put("org/b", 2)
        hf_search._cache_put("org/c", 3)
        assert hf_search._cache_get("org/a") is None  # evicted, oldest first
        assert hf_search._cache_get("org/c") == 3
        clock.now += hf_search._SIZE_CACHE_TTL_SECONDS + 1
        assert hf_search._cache_get("org/c") is None

    def test_after_a_429_no_search_sends_a_size_request_for_a_minute(
        self, monkeypatch, clock, caplog
    ):
        hits = [_hit("org/m0-4bit")]
        answers = [_Status(429)]
        api = _FakeApi(
            hits,
            {"org/m0-4bit": lambda _: answers.pop(0) if answers else _info([("w", 8)])},
        )
        _use(monkeypatch, api)

        hf_search.search_huggingface("x")
        assert len(api.calls) == 1

        clock.now += 59
        with caplog.at_level("INFO", logger="erudi"):
            (r,) = hf_search.search_huggingface("x")
        assert len(api.calls) == 1  # the listing went out, no size request did
        assert r["artifact_size_bytes"] is None
        assert any("cooldown=True" in m for m in caplog.messages)

        clock.now += 2  # 61 s after the 429
        (r,) = hf_search.search_huggingface("x")
        assert len(api.calls) == 2
        assert r["artifact_size_bytes"] == 8

    @pytest.mark.parametrize(
        "headers",
        [{"Retry-After": "120"}, {"RateLimit": '"api";r=0;t=120'}],
        ids=["retry-after", "ratelimit"],
    )
    def test_the_cooldown_follows_a_longer_announced_reset(self, monkeypatch, clock, headers):
        answers = [_Status(429, headers)]
        api = _FakeApi(
            [_hit("org/m0-4bit")],
            {"org/m0-4bit": lambda _: answers.pop(0) if answers else _info([("w", 8)])},
        )
        _use(monkeypatch, api)

        hf_search.search_huggingface("x")
        clock.now += 61
        hf_search.search_huggingface("x")
        assert len(api.calls) == 1  # still inside the announced 120 s
        clock.now += 60
        hf_search.search_huggingface("x")
        assert len(api.calls) == 2

    def test_a_shorter_announced_reset_keeps_the_floor(self, monkeypatch, clock):
        answers = [_Status(429, {"Retry-After": "5"})]
        api = _FakeApi(
            [_hit("org/m0-4bit")],
            {"org/m0-4bit": lambda _: answers.pop(0) if answers else _info([("w", 8)])},
        )
        _use(monkeypatch, api)

        hf_search.search_huggingface("x")
        clock.now += 30
        hf_search.search_huggingface("x")
        assert len(api.calls) == 1

    @pytest.mark.parametrize(
        "files",
        [
            [("mmproj-model-f16.gguf", 600), ("README.md", 1)],
            [("model-Q4_K_M.gguf.part1of2", 900), ("model-Q4_K_M.gguf.part2of2", 900)],
        ],
        ids=["only-mmproj", "only-raw-chunks"],
    )
    def test_a_gguf_repo_the_downloader_refuses_has_no_measured_size(self, monkeypatch, files):
        api = _FakeApi([_hit("org/odd-GGUF")], {"org/odd-GGUF": _info(files)})
        _use(monkeypatch, api, engine=_GgufEngine)

        (r,) = hf_search.search_huggingface("odd", limit=5)
        assert r["artifact_size_bytes"] is None

    def test_size_requests_run_on_daemon_threads(self, monkeypatch):
        """A request still on the wire at quit must not hold the interpreter's
        exit (the launcher's exit deadline): its thread is a daemon."""
        seen = []

        def record(_):
            seen.append(threading.current_thread().daemon)
            return _info([("w", 1)])

        hits = [_hit(f"org/m{i}-4bit") for i in range(3)]
        api = _FakeApi(hits, {h.id: record for h in hits})
        _use(monkeypatch, api)

        hf_search.search_huggingface("x")
        assert seen == [True, True, True]

    def test_a_failure_is_not_cached(self, monkeypatch):
        answers = [RuntimeError("blip"), _info([("w.safetensors", 5)])]
        api = _FakeApi([_hit("org/a-4bit")], {"org/a-4bit": lambda _: answers.pop(0)})
        _use(monkeypatch, api)

        assert hf_search.search_huggingface("a")[0]["artifact_size_bytes"] is None
        assert hf_search.search_huggingface("a")[0]["artifact_size_bytes"] == 5

    def test_one_info_line_per_search(self, monkeypatch, caplog):
        monkeypatch.setattr(hf_search, "_SIZE_WORKERS", 1)
        hits = [_hit("org/m0-4bit"), _hit("org/m1-4bit"), _hit("org/m2-4bit")]
        api = _FakeApi(
            hits, {"org/m0-4bit": _info([("w.safetensors", 1)]), "org/m1-4bit": _Status(429)}
        )
        _use(monkeypatch, api)

        with caplog.at_level("INFO", logger="erudi"):
            hf_search.search_huggingface("x", limit=10)
        lines = [r.getMessage() for r in caplog.records if "HF search sizes" in r.getMessage()]
        assert len(lines) == 1
        (line,) = lines
        assert "enriched=1/3" in line
        assert "stopped_on_429=True" in line
        assert "elapsed_ms=" in line
        assert line.isascii()

    def test_the_schema_carries_the_size(self):
        from src.domains.llms.schemas import HFSearchResult

        r = HFSearchResult(link="org/a", name="A", artifact_size_bytes=3_089_713_215)
        assert r.artifact_size_bytes == 3_089_713_215
        assert HFSearchResult(link="org/a", name="A").artifact_size_bytes is None

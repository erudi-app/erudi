"""The dedicated Knowledge Base embedding child's manager (#594).

Everything here runs on FAKES: a fake platform adapter (records spawns,
answers liveness) and a fake HTTP layer patched over the module's
``requests``. The real `mlx_vlm.server` child is exercised by
``tests/test_embedder_child_real.py`` (``mlx_only``).

Contract under test (`src.ingestion.embedder_child`):
- spawn → two-stage probe (`/health` with the Bearer key, then a 1-input
  ``POST /v1/embeddings``) → handle; a child that dies or never answers is
  terminated and reported with its log tail;
- the per-spawn API key reaches the child and the probe headers but never a
  log record;
- one child at a time: a live handle is reused, a dead one respawned;
- ``embed_texts`` batches requests, stamps the idle clock, and reports a
  child death with its tail;
- an ingestion lease keeps the idle reaper away; 5 minutes unused reaps the
  child; ``shutdown_embedder`` stops it deterministically.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests as requests_lib

from src.core.exceptions import EmbeddingModelMissingException, EngineException
from src.ingestion import chunking, embedder_child

pytestmark = pytest.mark.unit


# ====================== Fakes ======================


class FakeProc:
    _next_pid = 5000

    def __init__(self):
        FakeProc._next_pid += 1
        self.pid = FakeProc._next_pid
        self.alive = True
        self.exitcode = None
        self.terminated = False


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


def make_adapter(tmp_path: Path):
    """A fake platform adapter following the module-as-adapter protocol."""
    adapter = SimpleNamespace()
    adapter.server_name = "fake-embedder"
    adapter.spawns = []  # (port, api_key, model_path, log_path)
    adapter.terminated = []
    adapter.model_path = Path("/fake/snapshot")
    adapter.artifact_available = lambda: True
    adapter.artifact_downloadable = True
    adapter.exit_code = lambda proc: getattr(proc, "exitcode", None)
    adapter.resolve_model = lambda: adapter.model_path
    adapter.model_value = lambda path: str(path)
    adapter.prepare_log = lambda port: str(tmp_path / f"embedder-child-{port}.log")

    def spawn(*, port, api_key, model_path, log_path):
        proc = FakeProc()
        adapter.spawns.append(
            {"port": port, "api_key": api_key, "model_path": model_path, "log_path": log_path}
        )
        return proc

    def terminate(proc):
        proc.alive = False
        proc.terminated = True
        adapter.terminated.append(proc)

    adapter.spawn = spawn
    adapter.terminate = terminate
    adapter.is_alive = lambda proc: proc is not None and proc.alive
    adapter.read_tail = lambda log_path, since: "FAKE CHILD TAIL"
    adapter.discard_log = lambda log_path: None
    return adapter


class FakeHttp:
    """Stands in for the module's ``requests``: scriptable /health and
    /v1/embeddings answers, recorded calls."""

    RequestException = requests_lib.RequestException
    ConnectionError = requests_lib.ConnectionError

    def __init__(self):
        self.health_codes = [200]  # consumed left to right; last repeats
        self.gets = []
        self.posts = []
        self.post_handler = None  # optional callable(url, json) -> FakeResponse
        self.dimensions = 4

    def get(self, url, timeout=None, headers=None):
        self.gets.append({"url": url, "headers": headers})
        code = self.health_codes[0] if len(self.health_codes) == 1 else self.health_codes.pop(0)
        return FakeResponse(status_code=code)

    def post(self, url, json=None, timeout=None, headers=None):
        self.posts.append({"url": url, "json": json, "headers": headers})
        if self.post_handler is not None:
            return self.post_handler(url, json)
        inputs = json.get("input", [])
        payload = {
            "data": [
                {"object": "embedding", "index": i, "embedding": [0.5] * self.dimensions}
                for i in range(len(inputs))
            ]
        }
        return FakeResponse(status_code=200, payload=payload)


@pytest.fixture(autouse=True)
def _no_e5_tokenizer_reaper():
    """The thread-count assertions below must see no e5 tokenizer reaper:
    one an earlier test started can exit between two reads. Shutting the
    tokenizer down joins it (none of these tests loads the tokenizer)."""
    chunking.shutdown_tokenizer()


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    adapter = make_adapter(tmp_path)
    monkeypatch.setattr(embedder_child, "_adapter_override", adapter)
    embedder_child._reset_for_tests()
    yield adapter
    embedder_child._reset_for_tests()


@pytest.fixture
def http(monkeypatch):
    fake = FakeHttp()
    monkeypatch.setattr(embedder_child, "requests", fake)
    return fake


# ====================== Spawn and probe ======================


class TestSpawnAndProbe:
    def test_spawn_probes_health_then_one_input_embeddings(self, adapter, http):
        http.health_codes = [503, 503, 200]
        handle = embedder_child.ensure_embedder_running()

        assert len(adapter.spawns) == 1
        spawn = adapter.spawns[0]
        assert (
            embedder_child.EMBEDDER_PORT_START
            <= spawn["port"]
            < (embedder_child.EMBEDDER_PORT_START + embedder_child.EMBEDDER_PORT_COUNT)
        )
        # Stage 1 polled /health until 200, with the Bearer key.
        assert [g["url"].endswith("/health") for g in http.gets] == [True] * 3
        assert all(
            g["headers"] == {"Authorization": f"Bearer {spawn['api_key']}"} for g in http.gets
        )
        # Stage 2: exactly one 1-input embeddings probe, carrying the model
        # value (mlx_vlm pops the preload env var, so the field is mandatory).
        assert len(http.posts) == 1
        probe = http.posts[0]
        assert probe["url"].endswith("/v1/embeddings")
        assert probe["json"]["model"] == str(adapter.model_path)
        assert len(probe["json"]["input"]) == 1
        assert probe["headers"] == {"Authorization": f"Bearer {spawn['api_key']}"}
        # The handle carries what every later call needs.
        assert handle["port"] == spawn["port"]
        assert handle["api_key"] == spawn["api_key"]
        assert handle["base_url"] == f"http://127.0.0.1:{spawn['port']}"

    def test_api_key_is_fresh_per_spawn_and_never_logged(self, adapter, http, caplog):
        # A shutdown is terminal, so the respawn comes from a dead child.
        with caplog.at_level("DEBUG"):
            first = embedder_child.ensure_embedder_running()
            first["proc"].alive = False
            embedder_child.ensure_embedder_running()
        keys = [spawn["api_key"] for spawn in adapter.spawns]
        assert len(keys) == 2 and keys[0] != keys[1]
        assert all(len(key) == 64 for key in keys)
        for key in keys:
            assert all(key not in record.getMessage() for record in caplog.records)

    def test_reuses_live_child_without_a_second_spawn(self, adapter, http):
        first = embedder_child.ensure_embedder_running()
        second = embedder_child.ensure_embedder_running()
        assert len(adapter.spawns) == 1
        assert second["port"] == first["port"]

    def test_dead_child_is_respawned(self, adapter, http):
        first = embedder_child.ensure_embedder_running()
        first["proc"].alive = False
        second = embedder_child.ensure_embedder_running()
        assert len(adapter.spawns) == 2
        assert second["proc"] is not first["proc"]

    def test_child_dying_during_probe_reports_the_tail(self, adapter, http, monkeypatch):
        def get_and_kill(url, timeout=None, headers=None):
            adapter_proc.alive = False
            return FakeResponse(status_code=503)

        original_spawn = adapter.spawn

        def spawn(**kwargs):
            nonlocal adapter_proc
            adapter_proc = original_spawn(**kwargs)
            return adapter_proc

        adapter_proc = None
        adapter.spawn = spawn
        http.get = get_and_kill
        with pytest.raises(EngineException) as excinfo:
            embedder_child.ensure_embedder_running()
        assert "FAKE CHILD TAIL" in str(excinfo.value.message)
        assert embedder_child._handle is None

    def test_probe_stage2_failure_terminates_the_child(self, adapter, http):
        # mlx_vlm answers /health 200 even when the preload failed — only the
        # embeddings probe catches it.
        http.post_handler = lambda url, json: FakeResponse(status_code=500, text="load_failed")
        with pytest.raises(EngineException):
            embedder_child.ensure_embedder_running()
        assert len(adapter.terminated) == 1
        assert embedder_child._handle is None

    def test_probe_timeout_terminates_and_reports(self, adapter, http, monkeypatch):
        monkeypatch.setattr(embedder_child, "PROBE_TIMEOUT_S", 0.3)
        monkeypatch.setattr(embedder_child, "PROBE_POLL_INTERVAL_S", 0.05)
        http.health_codes = [503]
        with pytest.raises(EngineException) as excinfo:
            embedder_child.ensure_embedder_running()
        assert "FAKE CHILD TAIL" in str(excinfo.value.message)
        assert len(adapter.terminated) == 1

    def test_missing_model_raises_without_spawning(self, adapter, http):
        def resolve_model():
            raise EmbeddingModelMissingException()

        adapter.resolve_model = resolve_model
        with pytest.raises(EmbeddingModelMissingException):
            embedder_child.ensure_embedder_running()
        assert adapter.spawns == []


# ====================== embed_texts ======================


class TestEmbedTexts:
    def test_embeds_in_bounded_batches(self, adapter, http):
        texts = [f"text {i}" for i in range(embedder_child.EMBED_BATCH_SIZE * 2 + 6)]
        vectors = embedder_child.embed_texts(texts)
        assert len(vectors) == len(texts)
        embed_posts = [p for p in http.posts[1:]]  # posts[0] is the probe
        assert [len(p["json"]["input"]) for p in embed_posts] == [
            embedder_child.EMBED_BATCH_SIZE,
            embedder_child.EMBED_BATCH_SIZE,
            6,
        ]
        assert all(p["json"]["model"] == str(adapter.model_path) for p in embed_posts)

    def test_embed_touches_the_idle_clock(self, adapter, http):
        embedder_child.ensure_embedder_running()
        embedder_child._last_used -= 1000.0
        before = embedder_child._last_used
        embedder_child.embed_texts(["hello"])
        assert embedder_child._last_used > before

    def test_child_death_mid_request_reports_the_tail(self, adapter, http):
        handle = embedder_child.ensure_embedder_running()

        def post_dies(url, json):
            handle["proc"].alive = False
            raise FakeHttp.ConnectionError("peer closed connection")

        http.post_handler = post_dies
        with pytest.raises(EngineException) as excinfo:
            embedder_child.embed_texts(["hello"])
        assert "FAKE CHILD TAIL" in str(excinfo.value.message)
        # The dead handle is dropped so the next call respawns.
        assert embedder_child._handle is None

    def test_http_error_with_live_child_raises_and_keeps_it(self, adapter, http):
        handle = embedder_child.ensure_embedder_running()
        http.post_handler = lambda url, json: FakeResponse(status_code=400, text="too long")
        with pytest.raises(EngineException):
            embedder_child.embed_texts(["hello"])
        assert embedder_child._handle is handle
        assert handle["proc"].alive


# ====================== Lease, idle reap, shutdown ======================


class TestLifecycle:
    def test_idle_child_is_reaped_after_the_fixed_delay(self, adapter, http):
        embedder_child.ensure_embedder_running()
        now = embedder_child._last_used
        assert embedder_child.EMBEDDER_IDLE_SECONDS == 300.0
        assert not embedder_child.reap_if_idle(now=now + 299.0)
        assert embedder_child._handle is not None
        assert embedder_child.reap_if_idle(now=now + 301.0)
        assert embedder_child._handle is None
        assert len(adapter.terminated) == 1

    def test_lease_blocks_the_reaper(self, adapter, http):
        with embedder_child.ingestion_lease():
            embedder_child.ensure_embedder_running()
            now = embedder_child._last_used
            assert not embedder_child.reap_if_idle(now=now + 10_000.0)
            assert embedder_child._handle is not None
        # The lease's release restarts the idle clock.
        assert not embedder_child.reap_if_idle(now=embedder_child._last_used + 1.0)
        assert embedder_child.reap_if_idle(
            now=embedder_child._last_used + embedder_child.EMBEDDER_IDLE_SECONDS + 1.0
        )

    def test_nested_leases_all_must_release(self, adapter, http):
        with embedder_child.ingestion_lease():
            with embedder_child.ingestion_lease():
                embedder_child.ensure_embedder_running()
            assert not embedder_child.reap_if_idle(now=embedder_child._last_used + 10_000.0)
        assert embedder_child.reap_if_idle(
            now=embedder_child._last_used + embedder_child.EMBEDDER_IDLE_SECONDS + 1.0
        )

    def test_shutdown_terminates_the_child(self, adapter, http):
        embedder_child.ensure_embedder_running()
        embedder_child.shutdown_embedder()
        assert embedder_child._handle is None
        assert len(adapter.terminated) == 1
        # Idempotent.
        embedder_child.shutdown_embedder()
        assert len(adapter.terminated) == 1

    def test_reap_with_no_child_is_a_noop(self, adapter, http):
        assert not embedder_child.reap_if_idle(now=time.monotonic() + 10_000.0)


# ====================== Prewarm ======================


class TestPrewarm:
    def test_prewarm_spawns_in_the_background(self, adapter, http):
        embedder_child.prewarm_embedder()
        deadline = time.monotonic() + 5.0
        while embedder_child._handle is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert embedder_child._handle is not None
        assert len(adapter.spawns) == 1

    def test_prewarm_missing_model_is_one_info_not_an_error(self, adapter, http, caplog):
        def resolve_model():
            raise EmbeddingModelMissingException()

        adapter.resolve_model = resolve_model
        with caplog.at_level("INFO"):
            embedder_child._prewarm()  # the thread's body, run synchronously
        assert adapter.spawns == []
        infos = [r for r in caplog.records if "pre-start" in r.getMessage()]
        assert infos and all(r.levelname == "INFO" for r in infos)

    def test_prewarm_failure_is_one_warning_never_raised(self, adapter, http, caplog):
        def spawn(**kwargs):
            raise OSError("fork refused")

        adapter.spawn = spawn
        with caplog.at_level("WARNING"):
            embedder_child._prewarm()
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("pre-start" in r.getMessage() for r in warnings)

    def test_prewarm_without_a_platform_adapter_is_a_noop(self, monkeypatch):
        monkeypatch.setattr(embedder_child, "_adapter_override", None)
        embedder_child._reset_for_tests()
        thread_count = threading.active_count()
        embedder_child.prewarm_embedder()
        assert threading.active_count() == thread_count


# ====================== Platform selection ======================


class TestQueryInterleavesIngestion:
    def test_a_query_interleaves_between_ingestion_batches(self, adapter, http):
        """The request lock is held PER BATCH, FIFO: a search query asked while
        a file's batches stream waits for at most one batch, never the file."""
        order = []
        a1_entered = threading.Event()
        a1_release = threading.Event()
        batch = embedder_child.EMBED_BATCH_SIZE

        def handler(url, json):
            inputs = json["input"]
            if len(inputs) == batch and inputs[0] == "A0":
                order.append("A1")
                a1_entered.set()
                assert a1_release.wait(5.0)
            elif len(inputs) == batch:
                order.append("A2")
            elif inputs != ["ping"]:  # the readiness probe stays out of it
                order.append("B")
            return FakeResponse(
                200,
                {
                    "data": [
                        {"object": "embedding", "index": i, "embedding": [0.5]}
                        for i in range(len(inputs))
                    ]
                },
            )

        http.post_handler = handler
        ingestion = threading.Thread(
            target=lambda: embedder_child.embed_texts([f"A{i}" for i in range(batch * 2)])
        )
        ingestion.start()
        assert a1_entered.wait(5.0)
        query = threading.Thread(target=lambda: embedder_child.embed_texts(["une question"]))
        query.start()
        # The query is enqueued on the FIFO lock behind the batch in flight.
        deadline = time.monotonic() + 5.0
        while not embedder_child._request_lock._waiters and time.monotonic() < deadline:
            time.sleep(0.005)
        assert embedder_child._request_lock._waiters
        a1_release.set()
        ingestion.join(5.0)
        query.join(5.0)
        assert order == ["A1", "B", "A2"]


class TestShutdownSemantics:
    def test_ensure_refuses_after_shutdown(self, adapter, http):
        embedder_child.shutdown_embedder()
        with pytest.raises(EngineException):
            embedder_child.ensure_embedder_running()
        assert adapter.spawns == []

    def test_shutdown_during_a_probe_is_prompt_and_terminates_the_pending_child(
        self, adapter, http, monkeypatch
    ):
        monkeypatch.setattr(embedder_child, "PROBE_TIMEOUT_S", 10.0)
        monkeypatch.setattr(embedder_child, "PROBE_POLL_INTERVAL_S", 0.02)
        probing = threading.Event()

        def slow_health(url, timeout=None, headers=None):
            probing.set()
            return FakeResponse(status_code=503)

        http.get = slow_health
        outcome = []

        def run():
            try:
                embedder_child.ensure_embedder_running()
                outcome.append("ok")
            except Exception as exc:
                outcome.append(exc)

        prober = threading.Thread(target=run)
        prober.start()
        assert probing.wait(5.0)
        start = time.monotonic()
        embedder_child.shutdown_embedder()
        # Never waits out the probe timeout.
        assert time.monotonic() - start < 2.0
        prober.join(10.0)
        assert outcome and isinstance(outcome[0], EngineException)
        assert len(adapter.terminated) >= 1
        assert embedder_child._handle is None
        with pytest.raises(EngineException):
            embedder_child.ensure_embedder_running()


class TestReleaseNow:
    def test_release_now_frees_an_idle_unleased_child(self, adapter, http):
        embedder_child.ensure_embedder_running()
        assert embedder_child.release_now(reason="memory pressure")
        assert embedder_child._handle is None
        assert len(adapter.terminated) == 1

    def test_release_now_refuses_under_a_lease(self, adapter, http):
        with embedder_child.ingestion_lease():
            embedder_child.ensure_embedder_running()
            assert not embedder_child.release_now(reason="memory pressure")
            assert embedder_child._handle is not None

    def test_release_now_refuses_while_a_request_is_in_flight(self, adapter, http):
        embedder_child.ensure_embedder_running()
        embedder_child._request_lock.acquire()
        try:
            assert not embedder_child.release_now(reason="memory pressure")
            assert embedder_child._handle is not None
        finally:
            embedder_child._request_lock.release()

    def test_release_now_without_a_child_is_a_noop(self, adapter, http):
        assert not embedder_child.release_now(reason="memory pressure")


class TestAtexitPerHandle:
    def test_the_atexit_handler_follows_its_handle(self, adapter, http, monkeypatch):
        registered, unregistered = [], []
        fake_atexit = SimpleNamespace(
            register=lambda f: registered.append(f),
            unregister=lambda f: unregistered.append(f),
        )
        monkeypatch.setattr(embedder_child, "atexit", fake_atexit)
        first = embedder_child.ensure_embedder_running()
        first["proc"].alive = False
        embedder_child.ensure_embedder_running()
        assert len(registered) == 2
        # Dropping the dead handle unregistered ITS handler, not the new one.
        assert unregistered == [registered[0]]
        embedder_child.shutdown_embedder()
        assert unregistered == [registered[0], registered[1]]


class TestPrewarmTuning:
    def test_prewarm_skips_a_live_child_and_keeps_its_idle_clock(self, adapter, http):
        embedder_child.ensure_embedder_running()
        embedder_child._last_used -= 100.0
        before = embedder_child._last_used
        thread_count = threading.active_count()
        embedder_child.prewarm_embedder()
        assert threading.active_count() == thread_count
        assert embedder_child._last_used == before
        assert len(adapter.spawns) == 1

    def test_prewarm_backs_off_after_a_failure(self, adapter, http):
        def spawn(**kwargs):
            raise OSError("fork refused")

        adapter.spawn = spawn
        embedder_child._prewarm()
        thread_count = threading.active_count()
        embedder_child.prewarm_embedder()  # within the backoff window
        assert threading.active_count() == thread_count

    def test_prewarm_missing_model_logs_once_per_state(self, adapter, http, caplog):
        def resolve_model():
            raise EmbeddingModelMissingException()

        adapter.resolve_model = resolve_model
        with caplog.at_level("INFO"):
            embedder_child._prewarm()
            embedder_child._prewarm()
        infos = [r for r in caplog.records if "pre-start" in r.getMessage()]
        assert len(infos) == 1


class TestParentWatch:
    def test_the_watch_exits_when_the_parent_changes(self):
        from src.engines import _mlx_vlm_server_runner as runner

        ppids = iter([100, 100, 1])
        exits = []
        runner._parent_watch_loop(
            100,
            getppid=lambda: next(ppids),
            hard_exit=exits.append,
            sleep=lambda s: None,
        )
        assert exits == [0]

    def test_the_mlx_adapter_spawns_with_the_parent_watch(self, tmp_path, monkeypatch):
        from src.ingestion import mlx_embedder

        captured = {}

        class FakeProcess:
            def __init__(self, target=None, args=None, daemon=None):
                captured["target"] = target
                captured["args"] = args
                self.pid = 777

            def start(self):
                return None

        monkeypatch.setattr(mlx_embedder.mp, "Process", FakeProcess)
        mlx_embedder.spawn(
            port=27400,
            api_key="k" * 64,
            model_path=tmp_path,
            log_path=str(tmp_path / "embedder-child-27400.log"),
        )
        from src.engines._mlx_vlm_server_runner import run_mlx_vlm_server

        assert captured["target"] is run_mlx_vlm_server
        argv = captured["args"][0]
        assert "--embedding-model" in argv
        # (argv, log_path, apc_num_blocks, token_queue_timeout_s, watch_parent)
        assert captured["args"][4] is True


class TestPlatformSelection:
    def test_darwin_arm_resolves_the_mlx_adapter(self, monkeypatch):
        monkeypatch.setattr(embedder_child, "_adapter_override", embedder_child._PLATFORM)
        monkeypatch.setattr(embedder_child.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(embedder_child.platform, "machine", lambda: "arm64")
        adapter = embedder_child._get_adapter()
        from src.ingestion import mlx_embedder

        assert adapter is mlx_embedder

"""The persisted inference-backend preference, and the startup engine notice.

Auto-detection stays where it is (``BaseEngine.get_engine()``, before the
migrations run). What this file covers is the second half: after the schema is
at head and the catalog is populated, the lifespan reads
``user_settings.inference_backend`` and, when the user refused the GPU, swaps
``CUDA_Engine`` for ``CPU_Engine`` -- then runs the pre-flight only on the
engine that actually won.

Nothing here touches NVML or a GPU: the engines are real classes, the readings
are stubbed.
"""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from src.core import api
from src.core import config
from src.engines.cpu_engine import CPU_Engine
from src.engines.cuda_engine import CUDA_Engine
from src.engines.mlx_engine import MLX_Engine

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _restore_engine():
    """The selected engine is process-global; put it back after every test."""
    previous = config.LLM_Engine
    yield
    config.LLM_Engine = previous


def _preference(monkeypatch, value):
    monkeypatch.setattr(
        "src.domains.user_settings.repository.read_inference_backend",
        lambda: value,
    )


# ============ The preference decides, and only within its remit ============


class TestInferenceBackendPreference:
    def test_cpu_preference_replaces_the_cuda_engine(self, monkeypatch):
        config.LLM_Engine = CUDA_Engine
        _preference(monkeypatch, "cpu")
        monkeypatch.delenv("ERUDI_FORCE_CPU", raising=False)

        api.apply_inference_backend_preference()

        assert config.LLM_Engine is CPU_Engine

    def test_auto_preference_keeps_the_detected_engine(self, monkeypatch):
        config.LLM_Engine = CUDA_Engine
        _preference(monkeypatch, "auto")
        monkeypatch.delenv("ERUDI_FORCE_CPU", raising=False)

        api.apply_inference_backend_preference()

        assert config.LLM_Engine is CUDA_Engine

    def test_the_preference_never_touches_mlx(self, monkeypatch):
        """Apple Silicon has no CUDA story: the setting is inert there."""
        config.LLM_Engine = MLX_Engine
        _preference(monkeypatch, "cpu")
        monkeypatch.delenv("ERUDI_FORCE_CPU", raising=False)

        api.apply_inference_backend_preference()

        assert config.LLM_Engine is MLX_Engine

    def test_force_cpu_env_wins_over_an_auto_preference(self, monkeypatch):
        """ERUDI_FORCE_CPU already made get_engine() return CPU_Engine; the
        setting must not undo the developer override by putting CUDA back."""
        config.LLM_Engine = CPU_Engine
        _preference(monkeypatch, "auto")
        monkeypatch.setenv("ERUDI_FORCE_CPU", "1")

        api.apply_inference_backend_preference()

        assert config.LLM_Engine is CPU_Engine

    def test_an_unreadable_preference_keeps_the_gpu(self, monkeypatch):
        """The setting exists to REFUSE the GPU, so anything short of an
        explicit refusal leaves the hardware detection alone."""
        config.LLM_Engine = CUDA_Engine

        def _boom():
            raise RuntimeError("database is not ready")

        monkeypatch.setattr("src.domains.user_settings.repository.read_inference_backend", _boom)
        monkeypatch.delenv("ERUDI_FORCE_CPU", raising=False)

        with pytest.raises(RuntimeError):
            api.apply_inference_backend_preference()
        assert config.LLM_Engine is CUDA_Engine


class TestSharedFormatTagInvariant:
    def test_cpu_and_cuda_share_a_format_tag(self):
        """Swapping the engine AFTER the catalog was reconciled is only safe
        because both engines consume the same artefact format. Break this and
        the swap silently leaves the user with a catalog of unrunnable models.
        """
        assert CPU_Engine.FORMAT_TAG == CUDA_Engine.FORMAT_TAG == "gguf"

    def test_mlx_does_not_share_it(self):
        """The counter-example that makes the invariant meaningful."""
        assert MLX_Engine.FORMAT_TAG != CPU_Engine.FORMAT_TAG


# ============ The startup notice ============


class TestEngineNotice:
    def _app(self):
        app = FastAPI()
        emitted: list = []
        app.state.emit_event = emitted.append
        return app, emitted

    def test_emits_the_preflight_verdict_on_cuda(self, monkeypatch):
        config.LLM_Engine = CUDA_Engine
        notice = {"event": "engine_notice", "code": "CUDA_DRIVER_TOO_OLD"}
        monkeypatch.setattr(api, "cuda_preflight_notice", lambda: notice)
        app, emitted = self._app()

        api.emit_engine_notice(app)

        assert emitted == [notice]

    def test_emits_nothing_when_the_machine_checks_out(self, monkeypatch):
        config.LLM_Engine = CUDA_Engine
        monkeypatch.setattr(api, "cuda_preflight_notice", lambda: None)
        app, emitted = self._app()

        api.emit_engine_notice(app)

        assert emitted == []

    def test_a_user_who_already_chose_cpu_is_not_nagged(self, monkeypatch):
        """The pre-flight runs AFTER the preference is applied, so a machine
        pinned to CPU never reads NVML and never sees the notice again."""
        config.LLM_Engine = CPU_Engine

        def _must_not_run():
            raise AssertionError("the pre-flight ran on a non-CUDA engine")

        monkeypatch.setattr(api, "cuda_preflight_notice", _must_not_run)
        app, emitted = self._app()

        api.emit_engine_notice(app)

        assert emitted == []

    def test_works_without_a_hook_on_app_state(self, monkeypatch):
        """Plain uvicorn in dev: no injected emitter, the shared stdout one."""
        config.LLM_Engine = CUDA_Engine
        notice = {"event": "engine_notice", "code": "CUDA_ERROR"}
        monkeypatch.setattr(api, "cuda_preflight_notice", lambda: notice)
        emitted: list = []
        monkeypatch.setattr(api, "emit_event", emitted.append)

        api.emit_engine_notice(FastAPI())

        assert emitted == [notice]


# ============ Both, through the real lifespan ============


class _FakeCheckpointerCM:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, *exc):
        return False


def _stub_startup(monkeypatch):
    handle = SimpleNamespace(
        sqlalchemy_url="postgresql+psycopg://x/erudi",
        psycopg_url="postgresql://x/erudi",
    )
    monkeypatch.setattr(api, "start_postgres", lambda *_: handle)
    monkeypatch.setattr(api, "stop_postgres", lambda *_: None)
    monkeypatch.setattr(api, "init_database", lambda *_: None)
    monkeypatch.setattr(api, "run_migrations", lambda *_: None)
    monkeypatch.setattr(api, "init_kb_store", lambda *_: None)
    monkeypatch.setattr(api, "close_kb_store", lambda *_a, **_k: None)
    monkeypatch.setattr(api, "open_checkpointer", lambda *_: _FakeCheckpointerCM())

    async def _fake_populate():
        return {}

    monkeypatch.setattr(api, "startup_populate_database", _fake_populate)


@pytest.mark.unit
async def test_lifespan_emits_the_engine_notice_after_the_catalog(monkeypatch):
    """End to end through the real lifespan: the notice reaches the hook, and
    it lands after `loading_catalog` -- i.e. after the migrations that create
    the column the preference lives in."""
    _stub_startup(monkeypatch)
    monkeypatch.setattr(api, "BaseEngine", SimpleNamespace(get_engine=lambda: CUDA_Engine))
    monkeypatch.setattr(config, "LLM_Engine", None, raising=False)
    monkeypatch.setattr(CUDA_Engine, "start_cleanup_task", classmethod(lambda cls: None))
    monkeypatch.setattr(CUDA_Engine, "stop_cleanup_task", classmethod(lambda cls: None))
    monkeypatch.setattr(CUDA_Engine, "cleanup", classmethod(lambda cls: None))
    monkeypatch.setattr(
        "src.domains.user_settings.repository.read_inference_backend", lambda: "auto"
    )
    monkeypatch.delenv("ERUDI_FORCE_CPU", raising=False)
    notice = {"event": "engine_notice", "code": "CUDA_COMPUTE_CAPABILITY_TOO_LOW"}
    monkeypatch.setattr(api, "cuda_preflight_notice", lambda: notice)

    timeline: list = []
    app = FastAPI()
    app.state.emit_phase = lambda phase: timeline.append(phase)
    app.state.emit_event = timeline.append

    async with api.lifespan(app):
        pass

    assert timeline == [
        "preparing_database",
        "running_migrations",
        "loading_catalog",
        notice,
    ]


@pytest.mark.unit
async def test_lifespan_applies_the_cpu_preference_and_stays_quiet(monkeypatch):
    """The user opted into CPU: the engine is swapped and no notice is emitted."""
    _stub_startup(monkeypatch)
    monkeypatch.setattr(api, "BaseEngine", SimpleNamespace(get_engine=lambda: CUDA_Engine))
    monkeypatch.setattr(config, "LLM_Engine", None, raising=False)
    monkeypatch.setattr(CPU_Engine, "start_cleanup_task", classmethod(lambda cls: None))
    monkeypatch.setattr(CPU_Engine, "stop_cleanup_task", classmethod(lambda cls: None))
    monkeypatch.setattr(CPU_Engine, "cleanup", classmethod(lambda cls: None))
    monkeypatch.setattr(
        "src.domains.user_settings.repository.read_inference_backend", lambda: "cpu"
    )
    monkeypatch.delenv("ERUDI_FORCE_CPU", raising=False)
    monkeypatch.setattr(
        api,
        "cuda_preflight_notice",
        lambda: {"event": "engine_notice", "code": "CUDA_DRIVER_TOO_OLD"},
    )

    events: list = []
    app = FastAPI()
    app.state.emit_event = events.append

    async with api.lifespan(app):
        assert config.LLM_Engine is CPU_Engine

    assert events == []

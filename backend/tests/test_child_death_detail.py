"""Details names the inference process's death, not the transport symptom (#690.1).

When the inference child dies during an answer (``kill -9`` on
``mlx_vlm.server``, ``llama-server.exe`` ended on Windows), the stream breaks
with an HTTP transport error: "peer closed connection without sending
complete message body" on macOS, a bare ``ReadError`` on Windows. The engine
knows the real cause -- the child is dead, with its pid, port and exit code --
and that is the leaf the failed turn's Details shows.
"""

import logging
from typing import ClassVar

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from src.agents import runner as runner_module
from src.agents.runner import ERROR_MESSAGE, AgentRunner, GenParams
from src.core import config
from src.engines.base_chat_server_engine import ChildDeath
from src.engines.base_engine import BaseEngine
from src.engines.cpu_engine import CPU_Engine
from src.engines.cuda_engine import CUDA_Engine
from src.engines.mlx_engine import MLX_Engine
from tests._helpers import ToolableFakeChatModel

pytestmark = pytest.mark.unit


class _Llm:
    id = 7
    link = "/fake/path"
    name = "Test 7B"
    param_size = 7.0


_PARAMS = GenParams(temperature=0.5, top_p=0.9, max_tokens=64)


class _RaisingModel(ToolableFakeChatModel):
    exc: ClassVar[BaseException] = RuntimeError("unset")

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        raise self.exc
        yield  # pragma: no cover  (makes this an async generator function)


def _raising(exc):
    return type("_M", (_RaisingModel,), {"exc": exc})(messages=iter([]))


class _DeadMpProcess:
    """An ``mp.Process`` after ``kill -9``: what MLX_Engine holds."""

    pid = 61112
    exitcode = -9

    def is_alive(self):
        return False


class _PopenDyingLate:
    """A ``subprocess.Popen`` for ``llama-server.exe``, ended from Task
    Manager: the connection reset reaches the client a moment before the
    process handle reports the exit (``poll()`` still says running)."""

    pid = 4120

    def __init__(self, alive_polls: int, code: int = 1):
        self._alive_polls = alive_polls
        self._code = code
        self.returncode = None

    def poll(self):
        if self._alive_polls > 0:
            self._alive_polls -= 1
            return None
        self.returncode = self._code
        return self._code


class _LivePopen:
    pid = 77
    returncode = None

    def poll(self):
        return None


def _transport_error(message: str) -> BaseException:
    """What the openai client raises mid-stream: the httpx error, chained
    from httpcore's (here a plain stand-in with the same message)."""
    request = httpx.Request("POST", "http://127.0.0.1:27300/v1/chat/completions")
    if message:
        exc = httpx.RemoteProtocolError(message, request=request)
    else:
        exc = httpx.ReadError("", request=request)
    try:
        raise exc from OSError(message)
    except BaseException as caught:
        return caught


@pytest.fixture
def engine_with(monkeypatch):
    """The turn's engine reports the child of a real ``engine_cls`` holding
    ``proc``: the liveness check is that engine's own (``mp.Process`` for
    MLX, ``Popen.poll`` for llama.cpp), while the rest of the turn runs on a
    bare engine (no memory brake, no prefix cache talking to a child)."""

    def _install(engine_cls, proc, port):
        loaded = type(f"_Loaded{engine_cls.__name__}", (engine_cls,), {})
        loaded._model = {"proc": proc, "port": port}
        monkeypatch.setattr(loaded, "_read_child_output", classmethod(lambda cls, p: "tail"))

        class _Turn(BaseEngine):
            @classmethod
            def child_death(cls):
                return loaded.child_death()

        monkeypatch.setattr(config, "LLM_Engine", _Turn)
        return loaded

    yield _install


async def _failure(monkeypatch, exc, thread_id):
    monkeypatch.setattr(runner_module, "build_chat_model", lambda llm, **kw: _raising(exc))
    runner = AgentRunner(checkpointer=InMemorySaver())
    events = [
        e
        async for e in runner.astream_text(
            llm=_Llm(),
            user_message="hi",
            system_prompt="s",
            params=_PARAMS,
            thread_id=thread_id,
        )
    ]
    return [e for e in events if e["t"] == "answer"][-1]


# ===== The engine names the death =====


def test_mlx_names_a_killed_child():
    class _Loaded(MLX_Engine):
        _model = {"proc": _DeadMpProcess(), "port": 27300}

    assert _Loaded.child_death().cause == (
        "mlx_vlm.server child is dead (pid 61112, port 27300, exit code -9)"
    )


@pytest.mark.parametrize("engine_cls", [CPU_Engine, CUDA_Engine])
def test_llama_cpp_names_an_ended_child(engine_cls):
    proc = _PopenDyingLate(alive_polls=0, code=1)

    class _Loaded(engine_cls):
        _model = {"proc": proc, "port": 27200}

    assert _Loaded.child_death().cause == (
        "llama-server child is dead (pid 4120, port 27200, exit code 1)"
    )


def test_a_running_child_or_no_model_names_nothing():
    class _Running(CPU_Engine):
        _model = {"proc": _LivePopen(), "port": 27200}

    class _Unloaded(CPU_Engine):
        _model = None

    assert _Running.child_death() is None
    assert _Unloaded.child_death() is None


def test_the_crash_report_still_carries_the_cause_and_the_tail(monkeypatch):
    class _Loaded(MLX_Engine):
        _model = {"proc": _DeadMpProcess(), "port": 27300}

    monkeypatch.setattr(_Loaded, "_read_child_output", classmethod(lambda cls, p: "last lines"))
    report = _Loaded.child_death().report
    assert report.startswith("mlx_vlm.server child is dead (pid 61112, port 27300, exit code -9)")
    assert report.endswith("last lines")


# ===== The failed turn's Details is that cause =====


async def test_a_killed_mlx_child_is_the_details_leaf(monkeypatch, engine_with):
    engine_with(MLX_Engine, _DeadMpProcess(), 27300)
    exc = _transport_error(
        "peer closed connection without sending complete message body (incomplete chunked read)"
    )

    failure = await _failure(monkeypatch, exc, "d-mlx")

    assert failure["text"] == ERROR_MESSAGE
    assert failure["detail"] == "mlx_vlm.server child is dead (pid 61112, port 27300, exit code -9)"


async def test_an_ended_windows_llama_server_is_the_details_leaf(monkeypatch, engine_with):
    """The bare ``ReadError`` of Windows; the process handle reports the
    exit only after a few polls, which the runner waits for."""
    monkeypatch.setattr(runner_module, "CHILD_DEATH_POLL_S", 0.001)
    engine_with(CUDA_Engine, _PopenDyingLate(alive_polls=3, code=1), 27200)

    failure = await _failure(monkeypatch, _transport_error(""), "d-win")

    assert failure["detail"] == "llama-server child is dead (pid 4120, port 27200, exit code 1)"


async def test_a_transport_error_with_a_running_child_keeps_its_own_leaf(monkeypatch, engine_with):
    """The wait is bounded: a child that stays up is not the cause."""
    monkeypatch.setattr(runner_module, "CHILD_DEATH_GRACE_S", 0.01)
    monkeypatch.setattr(runner_module, "CHILD_DEATH_POLL_S", 0.001)
    engine_with(CPU_Engine, _LivePopen(), 27200)

    failure = await _failure(monkeypatch, _transport_error("connection reset"), "d-live")

    assert failure["detail"] == "connection reset"


async def test_another_failure_does_not_wait_for_the_child(monkeypatch, engine_with):
    """Only a transport error waits for the exit: any other failure with a
    running child reports its own cause at once."""
    polls = []
    proc = _LivePopen()
    proc.poll = lambda: polls.append(1)  # type: ignore[method-assign]
    engine_with(CPU_Engine, proc, 27200)

    failure = await _failure(monkeypatch, RuntimeError("bad sampling value"), "d-other")

    assert failure["detail"] == "bad sampling value"
    assert len(polls) <= 2  # the immediate check and the log's suffix, no wait loop


async def test_an_engine_without_a_child_keeps_the_exception_leaf(monkeypatch):
    class _Bare(BaseEngine):
        pass

    monkeypatch.setattr(config, "LLM_Engine", _Bare)

    failure = await _failure(monkeypatch, _transport_error("connection reset"), "d-bare")

    assert failure["detail"] == "connection reset"


# ===== One record, written first; one observation for the log and Details =====


def _scripted_engine(monkeypatch, deaths, order):
    """The turn's engine answers ``child_death()`` from ``deaths`` in turn
    (None once they run out) and notes each read in ``order``."""
    script = iter(deaths)

    class _Turn(BaseEngine):
        @classmethod
        def child_death(cls):
            order.append("probe")
            return next(script, None)

    monkeypatch.setattr(config, "LLM_Engine", _Turn)


class _Order(logging.Handler):
    def __init__(self, order):
        super().__init__(level=logging.ERROR)
        self.order = order
        self.messages = []

    def emit(self, record):
        if "Agent streaming failed" in record.getMessage():
            self.order.append("log")
            self.messages.append(record.getMessage())


async def test_the_failure_is_recorded_before_waiting_for_the_child(monkeypatch):
    """A client gone during the wait must not lose the record: the ERROR is
    written at once; only the Details leaf waits for the death to show."""
    monkeypatch.setattr(runner_module, "CHILD_DEATH_POLL_S", 0.001)
    order: list = []
    late = ChildDeath(
        cause="llama-server child is dead (pid 9, port 27200, exit code 1)", report="r"
    )
    _scripted_engine(monkeypatch, [None, None, late], order)
    handler = _Order(order)
    logging.getLogger("erudi").addHandler(handler)
    try:
        failure = await _failure(monkeypatch, _transport_error(""), "d-order")
    finally:
        logging.getLogger("erudi").removeHandler(handler)

    assert order[:2] == ["probe", "log"]
    assert order.count("log") == 1
    assert failure["detail"] == late.cause


async def test_the_log_and_details_read_the_same_observation(monkeypatch):
    order: list = []
    first = ChildDeath(
        cause="mlx_vlm.server child is dead (pid 1, port 27300, exit code -9)", report="FIRST tail"
    )
    second = ChildDeath(
        cause="mlx_vlm.server child is dead (pid 2, port 27301, exit code -6)", report="SECOND tail"
    )
    _scripted_engine(monkeypatch, [first, second], order)
    handler = _Order(order)
    logging.getLogger("erudi").addHandler(handler)
    try:
        failure = await _failure(monkeypatch, _transport_error("peer closed"), "d-once")
    finally:
        logging.getLogger("erudi").removeHandler(handler)

    assert order == ["probe", "log"]
    (message,) = handler.messages
    assert message.endswith("; FIRST tail")
    assert failure["detail"] == first.cause

"""Metal out-of-memory detection (#504): the pure function and the runner turn."""

import logging
from typing import ClassVar

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from src.agents import runner as runner_module
from src.agents.curated_turns import CURATED_KINDS, curated_text
from src.agents.runner import ERROR_MESSAGE, AgentRunner, GenParams
from src.core import config
from src.engines.base_chat_server_engine import ChildDeath
from src.engines.base_engine import BaseEngine
from src.engines.metal_oom import find_metal_out_of_memory
from tests._helpers import ToolableFakeChatModel

pytestmark = pytest.mark.unit

METAL_LINE = (
    "[METAL] Command buffer execution failed: Insufficient Memory "
    "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
)


# ===== The pure detection =====


def test_the_real_metal_line_is_found():
    assert find_metal_out_of_memory(METAL_LINE) == METAL_LINE


def test_the_line_is_found_inside_a_child_log_tail():
    tail = f"INFO: Uvicorn running\nprefill 4096 tokens\n{METAL_LINE}\nAbort trap: 6\n"
    assert find_metal_out_of_memory(tail) == METAL_LINE


def test_the_line_is_found_inside_an_exception_message():
    text = f"Error in model: RuntimeError: {METAL_LINE}"
    assert find_metal_out_of_memory("connection reset", text) == text


def test_the_gpu_token_alone_is_enough():
    line = "IOGPU: kIOGPUCommandBufferCallbackErrorOutOfMemory"
    assert find_metal_out_of_memory(line) == line


def test_a_metal_insufficient_memory_line_without_the_token_is_found():
    line = "[METAL] Command buffer execution failed: Insufficient Memory"
    assert find_metal_out_of_memory(line) == line


@pytest.mark.parametrize(
    "text",
    [
        "",
        "connection reset",
        # Host RAM, not the GPU's: the sentence about the graphics memory would lie.
        "MemoryError",
        "Cannot allocate memory",
        "Out of memory",
        "Insufficient Memory",
        "RuntimeError: [METAL] Command buffer execution failed: Caused GPU Timeout Error",
        # llama.cpp / CUDA.
        "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 9000 MiB on device 0: "
        "cudaMalloc failed: out of memory",
        "llama_model_load: error loading model: unable to allocate Metal buffer",
        "Request needs 5037 context tokens (5029 prompt + 8 max generation), "
        "but MAX_KV_SIZE is 4096.",
    ],
)
def test_other_failures_are_not_a_metal_out_of_memory(text):
    assert find_metal_out_of_memory(text) is None


def test_none_and_empty_texts_are_ignored():
    assert find_metal_out_of_memory(None, "", METAL_LINE) == METAL_LINE


# ===== The runner =====


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
        yield  # pragma: no cover


def _engine(monkeypatch, *, tail=None, death=None):
    class _Turn(BaseEngine):
        @classmethod
        def child_death(cls):
            return death

        @classmethod
        def child_output_tail(cls, since_mark=None):
            return tail or ""

    monkeypatch.setattr(config, "LLM_Engine", _Turn)


async def _turn(monkeypatch, exc, thread_id):
    model = type("_M", (_RaisingModel,), {"exc": exc})(messages=iter([]))
    monkeypatch.setattr(runner_module, "build_chat_model", lambda llm, **kw: model)
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


def test_the_kind_is_a_failed_curated_turn_with_no_numbers():
    spec = CURATED_KINDS["metal_out_of_memory"]
    assert spec.error and spec.params == ()
    assert spec.template.isascii()


async def test_a_surviving_child_with_the_line_in_its_log_yields_the_curated_turn(monkeypatch):
    _engine(monkeypatch, tail=f"noise\n{METAL_LINE}\n")

    answer = await _turn(monkeypatch, RuntimeError("stream ended"), "m-alive")

    assert answer["text"] == curated_text("metal_out_of_memory")
    assert answer["curated"] is True
    assert answer["curated_error"]["kind"] == "metal_out_of_memory"
    assert answer["curated_error"]["params"] == {}
    assert METAL_LINE in answer["detail"]


async def test_the_line_in_the_exception_is_enough(monkeypatch):
    _engine(monkeypatch)

    answer = await _turn(monkeypatch, RuntimeError(f"Error in model: {METAL_LINE}"), "m-exc")

    assert answer["curated_error"]["kind"] == "metal_out_of_memory"
    assert METAL_LINE in answer["detail"]


async def test_a_dead_child_names_the_process_and_the_metal_cause(monkeypatch):
    death = ChildDeath(
        cause="mlx_vlm.server child is dead (pid 1, port 27300, exit code -6)",
        report=f"mlx_vlm.server child is dead (...). Child output:\n{METAL_LINE}",
    )
    _engine(monkeypatch, tail=METAL_LINE, death=death)

    answer = await _turn(monkeypatch, RuntimeError("peer closed connection"), "m-dead")

    assert answer["curated_error"]["kind"] == "metal_out_of_memory"
    assert answer["detail"].startswith(death.cause)
    assert "kIOGPUCommandBufferCallbackErrorOutOfMemory" in answer["detail"]


async def test_the_record_is_one_error_with_the_cause(monkeypatch, caplog):
    _engine(monkeypatch, tail=METAL_LINE)

    with caplog.at_level(logging.ERROR, logger="erudi"):
        await _turn(monkeypatch, RuntimeError("stream ended"), "m-log")

    records = [r for r in caplog.records if "Metal" in r.getMessage()]
    assert len(records) == 1
    assert records[0].levelno == logging.ERROR
    assert records[0].getMessage().isascii()
    assert "kIOGPUCommandBufferCallbackErrorOutOfMemory" in records[0].getMessage()


async def test_a_generic_failure_stays_generic(monkeypatch):
    _engine(monkeypatch, tail="all quiet")

    answer = await _turn(monkeypatch, RuntimeError("bad sampling value"), "m-generic")

    assert answer["text"] == ERROR_MESSAGE
    assert "curated_error" not in answer
    assert answer["detail"] == "bad sampling value"


def test_the_engine_reads_the_output_of_a_child_that_still_runs(monkeypatch):
    from src.engines.mlx_engine import MLX_Engine

    class _Loaded(MLX_Engine):
        _model = {"proc": object(), "port": 27300}

    monkeypatch.setattr(_Loaded, "_read_child_output", classmethod(lambda cls, p: METAL_LINE))

    assert find_metal_out_of_memory(_Loaded.child_output_tail()) == METAL_LINE


def test_an_engine_with_nothing_loaded_has_no_output():
    from src.engines.mlx_engine import MLX_Engine

    class _Unloaded(MLX_Engine):
        _model = None

    assert _Unloaded.child_output_tail() == ""


async def test_a_python_memory_error_stays_generic(monkeypatch):
    _engine(monkeypatch)

    answer = await _turn(monkeypatch, MemoryError(), "m-pymem")

    assert answer["text"] == ERROR_MESSAGE


# ===== A line written before the turn started is not this turn's =====


def _marked_engine(monkeypatch, lines):
    """An engine whose child log is ``lines`` (a list the test appends to):
    its mark is the log's length, its tail what was written after a mark."""

    class _Turn(BaseEngine):
        @classmethod
        def child_death(cls):
            return None

        @classmethod
        def child_output_mark(cls):
            return len(lines)

        @classmethod
        def child_output_tail(cls, since_mark=None):
            return "\n".join(lines[since_mark or 0 :])

    monkeypatch.setattr(config, "LLM_Engine", _Turn)


async def test_an_old_metal_line_is_not_pinned_on_a_later_generic_failure(monkeypatch):
    # The child survived an earlier out-of-memory: its line is still in the log.
    _marked_engine(monkeypatch, ["old turn", METAL_LINE, "next request served"])

    answer = await _turn(monkeypatch, RuntimeError("bad sampling value"), "m-stale")

    assert answer["text"] == ERROR_MESSAGE
    assert "curated_error" not in answer
    assert answer["detail"] == "bad sampling value"


async def test_a_line_written_during_the_turn_is_found(monkeypatch):
    lines = ["old turn", METAL_LINE]
    _marked_engine(monkeypatch, lines)

    class _Writes(_RaisingModel):
        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
            lines.append(METAL_LINE)  # the child reports it while answering
            raise RuntimeError("stream ended")
            yield  # pragma: no cover

    model = _Writes(messages=iter([]))
    monkeypatch.setattr(runner_module, "build_chat_model", lambda llm, **kw: model)
    runner = AgentRunner(checkpointer=InMemorySaver())
    events = [
        e
        async for e in runner.astream_text(
            llm=_Llm(), user_message="hi", system_prompt="s", params=_PARAMS, thread_id="m-fresh"
        )
    ]
    answer = [e for e in events if e["t"] == "answer"][-1]

    assert answer["curated_error"]["kind"] == "metal_out_of_memory"


def test_the_mlx_engine_reads_only_what_its_log_gained_since_the_mark(tmp_path):
    from src.engines.mlx_engine import MLX_Engine

    log = tmp_path / "mlx-child-27300.log"
    log.write_text(f"{METAL_LINE}\nserved\n", encoding="utf-8")

    class _Proc:
        pass

    proc = _Proc()
    setattr(proc, MLX_Engine._CHILD_LOG_ATTR, str(log))

    class _Loaded(MLX_Engine):
        _model = {"proc": proc, "port": 27300}

    mark = _Loaded.child_output_mark()
    assert find_metal_out_of_memory(_Loaded.child_output_tail(since_mark=mark)) is None
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"{METAL_LINE}\n")
    assert find_metal_out_of_memory(_Loaded.child_output_tail(since_mark=mark)) == METAL_LINE


def test_a_log_rolled_since_the_mark_is_read_whole(tmp_path):
    from src.engines import mlx_child_log

    log = tmp_path / "child.log"
    log.write_text("x" * 500, encoding="utf-8")
    mark = mlx_child_log.log_size(log)
    log.write_text(f"{METAL_LINE}\n", encoding="utf-8")  # truncated by a roll

    assert METAL_LINE in mlx_child_log.read_child_log_from(log, mark, max_chars=4000)
    assert mlx_child_log.log_size(tmp_path / "absent.log") == 0

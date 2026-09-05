"""A CUDA crash on the first message reaches the UI as a typed code.

The path under test, end to end:

    llama-server dies on spawn
      -> `_probe_ready` classifies the captured tail and raises an
         `EngineException` carrying `engine_code`
      -> the runner turns the failed construction into an answer event that
         carries `code` + `raw` alongside the curated sentinel text
      -> the conversation service frames it as the wire `error` event, which is
         where the renderer picks the code up and opens its decision modal.

Nothing above the engine invents a code: an ordinary failure (a missing GGUF, a
port clash) keeps carrying no code at all, which is what keeps it on the plain
error-turn path.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agents import runner as runner_module
from src.core.exceptions import EngineException
from src.engines.base_chat_server_engine import BaseChatServerEngine

pytestmark = pytest.mark.unit


class _Engine(BaseChatServerEngine):
    """Minimal concrete engine: only what `_probe_ready` touches."""

    _server_name = "llama-server"
    FORMAT_TAG = "gguf"
    _probe_timeout_s = 0.2
    _probe_poll_interval_s = 0.01

    @classmethod
    def _spawn_child(cls, **kwargs):  # pragma: no cover - never spawned here
        raise NotImplementedError

    @classmethod
    def _terminate_process(cls, proc):  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def _proc_is_alive(cls, proc):
        return False

    @classmethod
    def _resolve_model_artifact(cls, llm_local_path):  # pragma: no cover
        raise NotImplementedError


# ============ The exception carries the code ============


class TestEngineExceptionCode:
    def test_engine_code_defaults_to_none(self):
        exc = EngineException(message="something went wrong")
        assert exc.engine_code is None
        assert exc.engine_trace is None

    def test_engine_code_and_trace_are_carried(self):
        exc = EngineException(
            message="child exited",
            trace="CUDA error: out of memory",
            engine_code="CUDA_OUT_OF_MEMORY",
        )
        assert exc.engine_code == "CUDA_OUT_OF_MEMORY"
        assert exc.engine_trace == "CUDA error: out of memory"

    def test_the_erudi_code_is_untouched(self):
        """`erudi_code` is the HTTP-facing code every EngineException shares;
        the engine code is a second, narrower axis and must not overwrite it."""
        exc = EngineException(message="boom", engine_code="CUDA_ERROR")
        assert exc.erudi_code == "LLM_ENGINE_FAILURE"


# ============ The probe classifies the tail ============


class TestProbeClassifiesTheChildOutput:
    def _raise_early_crash(self, tail):
        with patch.object(_Engine, "_read_child_output", classmethod(lambda cls, p: tail)):
            with pytest.raises(EngineException) as excinfo:
                _Engine._probe_ready("http://127.0.0.1:19000", proc=MagicMock())
        return excinfo.value

    def test_a_ptx_failure_is_typed(self):
        exc = self._raise_early_crash(
            "CUDA error: the provided PTX was compiled with an unsupported toolchain"
        )
        assert exc.engine_code == "CUDA_DRIVER_TOO_OLD"
        assert "unsupported toolchain" in exc.engine_trace

    def test_a_missing_kernel_image_is_typed(self):
        exc = self._raise_early_crash(
            "CUDA error: no kernel image is available for execution on the device"
        )
        assert exc.engine_code == "CUDA_COMPUTE_CAPABILITY_TOO_LOW"

    def test_a_non_cuda_crash_carries_no_code(self):
        """A missing model file must stay on the generic error path."""
        exc = self._raise_early_crash("error: failed to load model: no such file")
        assert exc.engine_code is None

    def test_the_tail_is_still_in_the_message(self):
        """The curated message keeps showing the tail, code or no code."""
        exc = self._raise_early_crash("CUDA error: out of memory")
        assert "out of memory" in str(exc)


# ============ The runner surfaces it on the turn ============


class TestRunnerConstructionEvent:
    def test_a_typed_engine_failure_carries_code_and_raw(self):
        exc = EngineException(
            message="llama-server child exited before becoming ready (early crash). "
            "CUDA error: no kernel image is available for execution on the device",
            trace="CUDA error: no kernel image is available for execution on the device",
            engine_code="CUDA_COMPUTE_CAPABILITY_TOO_LOW",
        )
        event = runner_module._construction_error_event(exc)

        assert event["t"] == "answer"
        assert event["text"].startswith(runner_module.ERROR_SENTINEL)
        assert event["code"] == "CUDA_COMPUTE_CAPABILITY_TOO_LOW"
        assert "no kernel image" in event["raw"]

    def test_an_untyped_engine_failure_carries_no_code(self):
        event = runner_module._construction_error_event(
            EngineException(message="no .gguf file found in the model folder")
        )
        assert "code" not in event
        assert "raw" not in event
        assert "no .gguf file found" in event["text"]

    def test_a_non_engine_failure_keeps_the_generic_message(self):
        event = runner_module._construction_error_event(RuntimeError("kaboom"))
        assert event["text"] == runner_module.ERROR_MESSAGE
        assert "code" not in event
        # A stringified stack trace must never reach the user.
        assert "kaboom" not in event["text"]


# ============ The wire event the renderer reads ============


class TestConversationWireEvent:
    def _error_event(self, answer_event):
        from src.domains.conversations.services import build_stream_error_event

        return build_stream_error_event(answer_event)

    def test_the_code_and_raw_ride_on_the_error_event(self):
        event = self._error_event(
            {
                "t": "answer",
                "text": f"{runner_module.ERROR_SENTINEL} boom",
                "code": "CUDA_DRIVER_TOO_OLD",
                "raw": "CUDA error: the provided PTX was compiled ...",
            }
        )
        assert event == {
            "t": "error",
            "text": f"{runner_module.ERROR_SENTINEL} boom",
            "code": "CUDA_DRIVER_TOO_OLD",
            "raw": "CUDA error: the provided PTX was compiled ...",
        }

    def test_an_untyped_error_keeps_the_shape_it_always_had(self):
        """Existing clients read `{t, text}`; nothing extra appears for them."""
        event = self._error_event({"t": "answer", "text": f"{runner_module.ERROR_SENTINEL} boom"})
        assert event == {"t": "error", "text": f"{runner_module.ERROR_SENTINEL} boom"}

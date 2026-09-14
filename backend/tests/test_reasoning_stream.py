"""`src.agents.reasoning_stream` — pure extraction of the dedicated reasoning field (#554).

Both local servers extract chain-of-thought server-side and stream it in a
dedicated delta field of the raw chat-completion chunk: llama-server
(``--reasoning-format`` default ``auto``) sends ``delta.reasoning_content``;
mlx_vlm.server sends ``delta.reasoning`` and, on the pinned 0.6.17, mirrors it
into ``delta.reasoning_content``. The chunk shapes below are copied from the
design-phase captures of both engines (Qwen3.5-0.8B, 2026-09-14).

Pure module: no LangChain import required, so these tests run anywhere.
"""

import pytest

from src.agents.reasoning_stream import REASONING_KWARG, extract_reasoning_delta

pytestmark = pytest.mark.unit


def _chunk(delta):
    """A raw chat-completion chunk carrying one choice with the given delta."""
    return {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "finish_reason": None, "delta": delta, "logprobs": None}],
    }


def test_llama_reasoning_content_is_extracted():
    # llama-server `auto` shape: the delta carries ONLY the reasoning field.
    assert extract_reasoning_delta(_chunk({"reasoning_content": "Thinking"})) == "Thinking"


def test_mlx_reasoning_is_extracted():
    # mlx_vlm shape minus the 0.6.17 mirror field: `reasoning` alone must work.
    delta = {"role": "assistant", "content": None, "reasoning": "step one"}
    assert extract_reasoning_delta(_chunk(delta)) == "step one"


def test_both_fields_present_reasoning_content_wins():
    # mlx_vlm 0.6.17 sends both, identical; a hypothetical divergence must
    # resolve deterministically (reasoning_content is the more widespread name).
    delta = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "canonical",
        "reasoning": "mirror",
    }
    assert extract_reasoning_delta(_chunk(delta)) == "canonical"


def test_content_only_delta_has_no_reasoning():
    assert extract_reasoning_delta(_chunk({"content": "The answer"})) is None


def test_none_valued_fields_are_no_reasoning():
    # mlx_vlm sends explicit nulls on content chunks.
    delta = {"role": "assistant", "content": "Answer", "reasoning": None}
    assert extract_reasoning_delta(_chunk(delta)) is None


def test_empty_string_reasoning_is_no_reasoning():
    # "" must behave like None: nothing to emit, never an empty thinking event.
    delta = {"reasoning_content": "", "reasoning": ""}
    assert extract_reasoning_delta(_chunk(delta)) is None


def test_empty_reasoning_content_falls_back_to_reasoning():
    delta = {"reasoning_content": "", "reasoning": "real text"}
    assert extract_reasoning_delta(_chunk(delta)) == "real text"


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "data: [DONE]",
        {},
        {"choices": []},
        {"choices": None},
        {"choices": [{}]},
        {"choices": [{"delta": None}]},  # llama-server can send a null delta
        {"choices": [{"delta": "not-a-dict"}]},
        {"choices": [{"delta": {"reasoning_content": 42}}]},  # non-string value
    ],
)
def test_malformed_chunks_never_raise_and_carry_no_reasoning(raw):
    assert extract_reasoning_delta(raw) is None


def test_the_kwarg_name_matches_the_wire_field_the_runner_reads():
    # The runner reads ``additional_kwargs["reasoning_content"]``; the constant
    # is the single point of truth shared by the client and the tests.
    assert REASONING_KWARG == "reasoning_content"

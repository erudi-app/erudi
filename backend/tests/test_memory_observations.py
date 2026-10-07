"""What the MLX child really used, turn by turn, recorded next to the prior.

Every conversation and Arena turn opens a measurement window on the child
(the footprint interval, reset through the macOS SPI) and closes it at the end
of the turn. The calls made inside the window push their server-reported
usage; the observation records ``n = max over calls of prompt + completion``,
the peak above the base footprint, and what the prior predicted for ``n``.
A compaction restarts the window, so the summary call and the hops before it
never count. When the peak exceeds the prediction, ONE WARNING per child says
so; nothing is ever corrected automatically. Observations are persisted in
``memory_calibration.json`` (merged under a file lock, bounded).
"""

from __future__ import annotations

import json
import logging
import threading

import pytest

from src.engines import memory_observations, process_footprint
from src.engines.memory_budget import MemoryBudget
from src.engines.mlx_engine import MLX_Engine

pytestmark = pytest.mark.unit

GIB = 1024**3
KV = 114_688
VOCAB = 151_936


# ===================== engine-level: the measurement window =====================


class _Child:
    """Scripted footprint readings for one pid."""

    def __init__(self, *, spi=True, base=2 * GIB):
        self.spi = spi
        self.base = base
        self.current = base
        self.peak = base
        self.windows = 0

    def install(self, monkeypatch):
        monkeypatch.setattr(process_footprint, "footprint", lambda pid: self.current)
        monkeypatch.setattr(process_footprint, "begin_peak_window", self._begin)
        monkeypatch.setattr(process_footprint, "peak_since", lambda pid, started: self.peak)
        monkeypatch.setattr(process_footprint, "pressure_level", lambda: 1)
        monkeypatch.setattr(process_footprint, "swapouts", lambda: 100)

    def _begin(self, pid):
        self.windows += 1
        self.peak = self.current
        return self.spi


@pytest.fixture
def child(monkeypatch):
    scripted = _Child()
    scripted.install(monkeypatch)
    handle = {
        "pid": 4242,
        "model_path": "/models/qwen",
        "base_footprint_bytes": scripted.base,
        "memory_facts": {
            "weights_bytes": GIB,
            "kv_token_bytes": KV,
            "vocab_size": VOCAB,
            "prefill_step": 2048,
            "working_set_bytes": 12 * GIB,
            "model_type": "qwen3",
        },
    }
    monkeypatch.setattr(MLX_Engine, "_model", handle)
    monkeypatch.setattr(MLX_Engine, "_reset_prefix_cache", classmethod(lambda cls, h, **k: True))
    yield scripted, handle


def _predict(handle, n):
    return MemoryBudget.from_handle(MLX_Engine, handle).predict(n)


def _recorded(path):
    data = json.loads(path.read_text())
    (entry,) = data["entries"].values()
    return entry


def test_n_is_the_largest_single_call_not_a_mix_of_calls(child, _memory_observations_in_tmp):
    scripted, handle = child
    token = MLX_Engine.begin_memory_window()
    MLX_Engine.note_call_usage(handle, 300, 6000)
    MLX_Engine.note_call_usage(handle, 6500, 50)
    scripted.peak = scripted.base + 3 * GIB

    observation = MLX_Engine.end_memory_window(token)

    assert (observation["n"], observation["n_in"], observation["n_out"]) == (6550, 6500, 50)
    assert observation["predicted_bytes"] == _predict(handle, 6550)
    assert observation["y_bytes"] == 3 * GIB
    assert _recorded(_memory_observations_in_tmp)["observations"][-1]["n"] == 6550


def test_a_short_prompt_with_a_long_answer_is_recorded_at_prompt_plus_answer(child):
    scripted, handle = child
    token = MLX_Engine.begin_memory_window()
    MLX_Engine.note_call_usage(handle, 256, 8000)

    observation = MLX_Engine.end_memory_window(token)

    assert observation["n"] == 8256


def test_calls_before_a_compaction_restart_are_excluded(child):
    scripted, handle = child
    token = MLX_Engine.begin_memory_window()
    MLX_Engine.note_call_usage(handle, 9000, 10)  # a pre-compaction hop
    scripted.peak = scripted.base + 6 * GIB  # ...and the summary call's peak
    MLX_Engine.on_history_rewritten()
    MLX_Engine.note_call_usage(handle, 2000, 300)
    scripted.peak = scripted.base + GIB

    observation = MLX_Engine.end_memory_window(token)

    assert scripted.windows == 2, "the window restarted at the history rewrite"
    assert observation["n"] == 2300
    assert observation["y_bytes"] == GIB


def test_nothing_is_recorded_without_the_spi(child, _memory_observations_in_tmp):
    scripted, handle = child
    scripted.spi = False
    token = MLX_Engine.begin_memory_window()
    MLX_Engine.note_call_usage(handle, 1000, 10)

    assert MLX_Engine.end_memory_window(token) is None
    assert not _memory_observations_in_tmp.exists()


def test_nothing_is_recorded_without_usage(child, _memory_observations_in_tmp):
    token = MLX_Engine.begin_memory_window()

    assert MLX_Engine.end_memory_window(token) is None
    assert not _memory_observations_in_tmp.exists()


def test_usage_pushed_with_no_open_window_is_ignored(child):
    _scripted, handle = child
    MLX_Engine.note_call_usage(handle, 1000, 10)  # a title, before any turn
    token = MLX_Engine.begin_memory_window()
    MLX_Engine.note_call_usage(handle, 500, 5)

    assert MLX_Engine.end_memory_window(token)["n"] == 505


def test_the_observation_carries_what_qa_needs(child):
    scripted, handle = child
    scripted.current = scripted.base + GIB // 4  # residue left by the previous turn
    token = MLX_Engine.begin_memory_window()
    MLX_Engine.note_call_usage(handle, 1000, 10, cached_tokens=600, has_images=False)
    scripted.peak = scripted.base + 2 * GIB

    observation = MLX_Engine.end_memory_window(token, abandoned=True)

    assert observation["cache_read"] == 600
    assert observation["has_images"] is False
    assert observation["abandoned"] is True
    assert observation["residue_at_start_bytes"] == GIB // 4
    assert observation["spi_started"] is True
    assert observation["child_died"] is False
    assert observation["pressure_level"] == 1
    assert observation["swapouts"] == 0
    assert isinstance(observation["id"], str) and observation["id"]


def test_an_exceeded_prediction_warns_once_per_child_and_changes_nothing(child, caplog):
    scripted, handle = child
    before = _predict(handle, 1010)
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            token = MLX_Engine.begin_memory_window()
            MLX_Engine.note_call_usage(handle, 1000, 10)
            scripted.peak = scripted.base + 20 * GIB
            MLX_Engine.end_memory_window(token)

    warnings = [r for r in caplog.records if "exceeded" in r.getMessage()]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert message.isascii()
    for fragment in ("model=qwen3", "n_in=1000", "n_out=10", "residue"):
        assert fragment in message
    assert _predict(handle, 1010) == before


def test_an_image_turn_never_warns(child, caplog):
    scripted, handle = child
    with caplog.at_level(logging.WARNING):
        token = MLX_Engine.begin_memory_window()
        MLX_Engine.note_call_usage(handle, 1000, 10, has_images=True)
        scripted.peak = scripted.base + 20 * GIB
        observation = MLX_Engine.end_memory_window(token)

    assert observation["has_images"] is True
    assert not [r for r in caplog.records if "exceeded" in r.getMessage()]


def test_the_prior_scale_seam_forces_the_warning(child, caplog, monkeypatch):
    """The QA seam: a forced low prior produces exactly one WARNING."""
    scripted, handle = child
    monkeypatch.setenv("ERUDI_MEMORY_PRIOR_SCALE", "0.01")
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            token = MLX_Engine.begin_memory_window()
            MLX_Engine.note_call_usage(handle, 1000, 10)
            scripted.peak = scripted.base + GIB
            MLX_Engine.end_memory_window(token)

    assert len([r for r in caplog.records if "exceeded" in r.getMessage()]) == 1


def test_a_window_token_from_another_child_is_ignored(child):
    _scripted, handle = child
    token = MLX_Engine.begin_memory_window()
    MLX_Engine.note_call_usage(handle, 1000, 10)
    stale = ({"pid": 1}, 1, 1)

    assert MLX_Engine.end_memory_window(stale) is None
    assert MLX_Engine.end_memory_window(token) is not None


def test_a_token_from_an_earlier_window_on_the_same_child_is_ignored(child):
    _scripted, handle = child
    earlier = MLX_Engine.begin_memory_window()
    current = MLX_Engine.begin_memory_window()
    MLX_Engine.note_call_usage(handle, 1000, 10)

    assert MLX_Engine.end_memory_window(earlier) is None
    assert MLX_Engine.end_memory_window(current) is not None


def test_a_compaction_restart_keeps_the_turns_token_valid(child):
    _scripted, handle = child
    token = MLX_Engine.begin_memory_window()
    MLX_Engine.on_history_rewritten()
    MLX_Engine.note_call_usage(handle, 1000, 10)

    assert MLX_Engine.end_memory_window(token) is not None


def test_nothing_is_recorded_on_a_fallback_base(child, _memory_observations_in_tmp):
    """The data must hold measured bases only: the weights-on-disk fallback
    would make every observation of that child wrong by its error."""
    _scripted, handle = child
    handle["base_footprint_bytes"] = None
    token = MLX_Engine.begin_memory_window()
    MLX_Engine.note_call_usage(handle, 1000, 10)

    assert MLX_Engine.end_memory_window(token) is None
    assert not _memory_observations_in_tmp.exists()


def test_a_child_that_dies_during_the_window_is_recorded_without_a_peak(
    child, _memory_observations_in_tmp, monkeypatch
):
    """The ground truth of an under-prediction: the child is gone, its peak
    unreadable; the observation says so (``child_died``, ``y = None``)."""
    scripted, handle = child
    token = MLX_Engine.begin_memory_window()
    MLX_Engine.note_call_usage(handle, 9000, 10)
    monkeypatch.setattr(process_footprint, "peak_since", lambda pid, started: None)
    monkeypatch.setattr(MLX_Engine, "_proc_is_alive", classmethod(lambda cls, proc: False))

    observation = MLX_Engine.end_memory_window(token)

    assert observation["child_died"] is True
    assert observation["y_bytes"] is None
    assert observation["n"] == 9010
    assert _recorded(_memory_observations_in_tmp)["observations"][-1]["child_died"] is True


def test_the_base_footprint_is_measured_right_after_the_readiness_probe(monkeypatch):
    monkeypatch.setattr(process_footprint, "footprint", lambda pid: 3 * GIB if pid == 77 else None)
    handle = {"pid": 77}

    MLX_Engine._read_server_properties(handle)

    assert handle["base_footprint_bytes"] == 3 * GIB


def test_the_base_engine_hooks_are_no_ops():
    from src.engines.base_engine import BaseEngine

    assert BaseEngine.begin_memory_window() is None
    assert BaseEngine.end_memory_window(None) is None
    assert BaseEngine.note_call_usage({}, 1, 1) is None
    assert BaseEngine.note_call_start({}, 1) is None


# ===================== an abandoned turn is still measured =====================


def test_an_abandoned_turn_is_recorded_with_the_estimate_of_the_call_in_flight(
    child, _memory_observations_in_tmp, caplog
):
    """The stream was cut before the usage chunk -- often the heaviest turn
    of all. Its peak is real; its size is the client's estimate of the call
    in flight (``n_est``), never a measured ``n``."""
    import logging

    scripted, handle = child
    token = MLX_Engine.begin_memory_window()
    MLX_Engine.note_call_start(handle, 7000)
    scripted.peak = scripted.base + 20 * GIB

    with caplog.at_level(logging.WARNING):
        observation = MLX_Engine.end_memory_window(token, abandoned=True)

    assert observation["abandoned"] is True
    assert observation["n"] is None
    assert observation["n_est"] == 7000
    assert observation["y_bytes"] == 20 * GIB
    assert observation["predicted_bytes"] == _predict(handle, 7000)
    # An estimate is not a measurement: no exceeded-prediction WARNING.
    assert not [r for r in caplog.records if "exceeded" in r.getMessage()]
    assert _recorded(_memory_observations_in_tmp)["observations"][-1]["n_est"] == 7000


def test_a_completed_call_is_no_longer_in_flight(child):
    _scripted, handle = child
    token = MLX_Engine.begin_memory_window()
    MLX_Engine.note_call_start(handle, 7000)
    MLX_Engine.note_call_usage(handle, 6900, 40)

    observation = MLX_Engine.end_memory_window(token)

    assert observation["n"] == 6940
    assert observation["n_est"] is None


def test_a_turn_that_ended_normally_without_usage_still_records_nothing(child):
    _scripted, handle = child
    token = MLX_Engine.begin_memory_window()
    MLX_Engine.note_call_start(handle, 7000)

    assert MLX_Engine.end_memory_window(token) is None


def test_a_call_started_before_a_compaction_restart_is_not_in_flight(child):
    _scripted, handle = child
    token = MLX_Engine.begin_memory_window()
    MLX_Engine.note_call_start(handle, 9000)  # the hop before the rewrite
    MLX_Engine.on_history_rewritten()

    assert MLX_Engine.end_memory_window(token, abandoned=True) is None


async def test_the_client_pushes_its_estimate_when_a_call_starts(monkeypatch):
    from langchain_core.messages import AIMessageChunk, HumanMessage
    from langchain_openai import ChatOpenAI

    from src.agents.chat_model import erudi_chat_openai_class
    from src.agents.output_budget import estimate_prompt

    started = []

    async def _server(self, messages, *args, **kwargs):
        assert started, "the start is pushed before the call reaches the server"
        yield self._convert_chunk_to_generation_chunk(
            {"choices": [{"index": 0, "delta": {"content": "hi"}}]}, AIMessageChunk, {}
        )

    monkeypatch.setattr(ChatOpenAI, "_astream", _server)
    client = erudi_chat_openai_class()(
        base_url="http://127.0.0.1:1/v1",
        api_key="k",
        model="m",
        call_start_hook=started.append,
    )
    messages = [HumanMessage("hello " * 300)]

    _ = [c async for c in client._astream(messages)]

    assert started == [estimate_prompt(messages, digit_tokens=client.digit_tokens).total]


def test_only_the_conversation_client_announces_its_calls(monkeypatch):
    from src.agents.model_factory import build_chat_model
    from src.core import config

    starts = []

    class _Engine:
        @staticmethod
        def get_model_and_tokenizer(llm_id, link):
            return ({"base_url": "http://127.0.0.1:1", "alias": "a"}, {})

        @staticmethod
        def _payload_model_value(handle):
            return "m"

        @staticmethod
        def note_call_start(handle, estimated):
            starts.append(estimated)

    class _Llm:
        id = 1
        link = "/x"
        name = "x"

    monkeypatch.setattr(config, "LLM_Engine", _Engine)
    conversation = build_chat_model(
        _Llm(), temperature=0.1, top_p=0.9, max_tokens=8, record_usage=True
    )
    title = build_chat_model(_Llm(), temperature=0.1, top_p=0.9, max_tokens=8)

    assert title.call_start_hook is None
    conversation.call_start_hook(42)
    assert starts == [42]


# ===================== the client's usage hook =====================


def test_only_the_conversation_client_pushes_usage(monkeypatch):
    from src.agents.model_factory import build_chat_model
    from src.core import config

    pushed = []

    class _Engine:
        @staticmethod
        def get_model_and_tokenizer(llm_id, link):
            return ({"base_url": "http://127.0.0.1:1", "alias": "a"}, {})

        @staticmethod
        def _payload_model_value(handle):
            return "m"

        @staticmethod
        def note_call_usage(handle, n_in, n_out, cached=0, has_images=False):
            pushed.append((n_in, n_out))

    class _Llm:
        id = 1
        link = "/x"
        name = "x"

    monkeypatch.setattr(config, "LLM_Engine", _Engine)
    conversation = build_chat_model(
        _Llm(), temperature=0.1, top_p=0.9, max_tokens=8, record_usage=True
    )
    title = build_chat_model(_Llm(), temperature=0.1, top_p=0.9, max_tokens=8)

    assert conversation.usage_hook is not None
    assert title.usage_hook is None
    conversation.usage_hook(10, 2, 0, False)
    assert pushed == [(10, 2)]


async def test_the_client_pushes_the_usage_chunk_once(monkeypatch):
    from langchain_core.messages import AIMessageChunk, HumanMessage
    from langchain_openai import ChatOpenAI

    from src.agents.chat_model import erudi_chat_openai_class

    pushed = []

    async def _server(self, messages, *args, **kwargs):
        for raw in (
            {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 3,
                    "total_tokens": 123,
                    "prompt_tokens_details": {"cached_tokens": 64},
                },
            },
        ):
            yield self._convert_chunk_to_generation_chunk(raw, AIMessageChunk, {})

    monkeypatch.setattr(ChatOpenAI, "_astream", _server)
    client = erudi_chat_openai_class()(
        base_url="http://127.0.0.1:1/v1",
        api_key="k",
        model="m",
        usage_hook=lambda *args: pushed.append(args),
    )

    _ = [c async for c in client._astream([HumanMessage("hello")])]

    assert pushed == [(120, 3, 64, False)]


# ===================== persistence =====================


def _components(**overrides):
    components = memory_observations.key_components(
        model="Qwen3-0.6B",
        artifact_bytes=500_000_000,
        working_set_bytes=12 * GIB,
        runtime={"apc_block_size": 16, "prefill_step": 2048, "buffer_cache_limit": "default"},
    )
    components.update(overrides)
    return components


def _observation(i, **extra):
    return {"id": f"obs-{i}", "at": 1_000_000 + i, "n": i, **extra}


def test_a_round_trip_keeps_the_base_and_the_observations(_memory_observations_in_tmp):
    memory_observations.record(_components(), 2 * GIB, _observation(1))

    entry = _recorded(_memory_observations_in_tmp)

    assert entry["base_bytes"] == 2 * GIB
    assert [o["id"] for o in entry["observations"]] == ["obs-1"]


def test_the_model_component_is_the_artifacts_model_type(child, _memory_observations_in_tmp):
    """Not the conversation's model name (renameable, free user text): the
    artifact's ``model_type``, with its size already in the key."""
    _scripted, handle = child
    token = MLX_Engine.begin_memory_window()
    MLX_Engine.note_call_usage(handle, 1000, 10)
    MLX_Engine.end_memory_window(token)

    components = _recorded(_memory_observations_in_tmp)["components"]
    assert components["model"] == "qwen3"
    assert components["artifact_bytes"] == GIB


def test_the_key_components_are_in_clear_and_carry_no_app_version():
    components = _components()
    assert components["model"] == "Qwen3-0.6B"
    for key in ("artifact_bytes", "working_set_bytes", "macos", "mlx", "mlx_vlm", "runtime"):
        assert key in components
    assert "app_version" not in components
    assert not any(
        isinstance(v, str) and "/" in v and v.startswith("/") for v in components.values()
    )


def test_the_ring_holds_the_last_64_observations_per_key(_memory_observations_in_tmp):
    for i in range(70):
        memory_observations.record(_components(), GIB, _observation(i))

    entry = _recorded(_memory_observations_in_tmp)

    assert len(entry["observations"]) == 64
    assert entry["observations"][0]["id"] == "obs-6"


def test_an_older_runtime_entry_is_kept_and_labelled(_memory_observations_in_tmp):
    memory_observations.record(_components(mlx="0.31.0"), GIB, _observation(1))
    memory_observations.record(_components(mlx="0.32.2"), GIB, _observation(2))

    data = json.loads(_memory_observations_in_tmp.read_text())

    versions = sorted(entry["components"]["mlx"] for entry in data["entries"].values())
    assert versions == ["0.31.0", "0.32.2"]


def test_the_file_is_bounded_to_512_observations_oldest_first(_memory_observations_in_tmp):
    for i in range(600):
        memory_observations.record(_components(model=f"m{i % 10}"), GIB, _observation(i))

    data = json.loads(_memory_observations_in_tmp.read_text())
    kept = [o for entry in data["entries"].values() for o in entry["observations"]]

    assert len(kept) == 512
    assert min(o["at"] for o in kept) == 1_000_000 + 600 - 512


def test_concurrent_writers_merge_without_duplicates(_memory_observations_in_tmp):
    def _writer(start):
        for i in range(start, start + 25):
            memory_observations.record(_components(), GIB, _observation(i))
            memory_observations.record(_components(), GIB, _observation(i))  # same id again

    threads = [threading.Thread(target=_writer, args=(k * 25,)) for k in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    ids = [o["id"] for o in _recorded(_memory_observations_in_tmp)["observations"]]
    assert sorted(ids) == sorted({f"obs-{i}" for i in range(50)})


def test_malformed_entries_and_observations_are_dropped_never_blocking(
    _memory_observations_in_tmp, caplog
):
    _memory_observations_in_tmp.write_text(
        json.dumps(
            {
                "version": 1,
                "entries": {
                    "broken-entry": "not a dict",
                    "no-ring": {"components": {}, "observations": "nope"},
                    "mixed": {
                        "components": {"model": "x"},
                        "observations": [{"id": "ok-1", "at": 1}, "junk", {"no": "id"}],
                    },
                },
            }
        )
    )

    memory_observations.record(_components(), GIB, _observation(1))

    data = json.loads(_memory_observations_in_tmp.read_text())
    assert set(data["entries"]) == {"mixed", memory_observations.entry_key(_components())}
    assert [o["id"] for o in data["entries"]["mixed"]["observations"]] == ["ok-1"]


def test_a_corrupt_file_is_ignored_once_and_rewritten(_memory_observations_in_tmp, caplog):
    _memory_observations_in_tmp.write_text("{not json")

    with caplog.at_level(logging.WARNING):
        memory_observations.record(_components(), GIB, _observation(1))
        memory_observations.record(_components(), GIB, _observation(2))

    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1
    assert [o["id"] for o in _recorded(_memory_observations_in_tmp)["observations"]] == [
        "obs-1",
        "obs-2",
    ]


def test_the_default_path_lives_in_the_data_folder():
    from src.core import config

    assert memory_observations.default_observations_path() == (
        config.DATA_ROOT / "memory_calibration.json"
    )


# ===================== the runner opens and closes the window =====================


def _windowed_engine(events):
    from src.engines.base_engine import BaseEngine

    class _Engine(BaseEngine):
        @classmethod
        def effective_context_tokens(cls):
            return None

        @classmethod
        def begin_memory_window(cls):
            events.append(("begin",))
            return "token"

        @classmethod
        def end_memory_window(cls, token, *, abandoned=False):
            events.append(("end", token, abandoned))

    return _Engine


def _patch_models(monkeypatch, built):
    from langchain_core.messages import AIMessage

    from src.agents import runner as runner_module
    from tests._helpers import ToolableFakeChatModel

    def _build(llm, **kw):
        built.append(kw)
        return ToolableFakeChatModel(messages=iter([AIMessage(content="answer")]))

    monkeypatch.setattr(runner_module, "build_chat_model", _build)


class _Llm:
    id = 3
    link = "/m"
    name = "Qwen3-0.6B"
    param_size = 0.6


def _params():
    from src.agents.runner import GenParams

    return GenParams(temperature=0.5, top_p=0.9, max_tokens=64)


@pytest.mark.parametrize("stateful", [True, False], ids=["conversation", "arena"])
async def test_every_conversation_and_arena_turn_is_measured(monkeypatch, stateful):
    from langgraph.checkpoint.memory import InMemorySaver

    from src.agents.runner import AgentRunner
    from src.core import config

    events, built = [], []
    monkeypatch.setattr(config, "LLM_Engine", _windowed_engine(events))
    _patch_models(monkeypatch, built)
    runner = AgentRunner(checkpointer=InMemorySaver() if stateful else None)

    texts = [
        e
        async for e in runner.astream_text(
            llm=_Llm(),
            user_message="hi",
            system_prompt="s",
            params=_params(),
            thread_id="t" if stateful else None,
            summarize=stateful,
        )
    ]

    assert "answer" in "".join(texts)
    assert events == [("begin",), ("end", "token", False)]
    assert len([kw for kw in built if kw.get("record_usage")]) == 1


async def test_an_abandoned_turn_closes_its_window_as_abandoned(monkeypatch):
    import contextlib

    from src.agents.runner import AgentRunner
    from src.core import config

    events, built = [], []
    monkeypatch.setattr(config, "LLM_Engine", _windowed_engine(events))
    _patch_models(monkeypatch, built)
    stream = AgentRunner().astream_text(
        llm=_Llm(), user_message="hi", system_prompt="s", params=_params()
    )
    async with contextlib.aclosing(stream) as texts:
        async for _ in texts:
            break

    assert events[-1] == ("end", "token", True)


async def test_a_title_opens_no_window(monkeypatch):
    from src.agents.runner import AgentRunner
    from src.core import config

    events, built = [], []
    monkeypatch.setattr(config, "LLM_Engine", _windowed_engine(events))
    _patch_models(monkeypatch, built)

    _ = [
        t
        async for t in AgentRunner().astream_oneshot(
            llm=_Llm(), prompt_text="title", temperature=0.1, top_p=0.9, max_tokens=12
        )
    ]

    assert events == []
    assert not any(kw.get("record_usage") for kw in built)

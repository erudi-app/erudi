"""Per-artifact reasoning-lever detection (1.1.2).

Which reasoning lever a model honours is a property of the ARTIFACT, not of a
family: a community fine-tune keeps its parent's name and ``llm.type`` while
shipping whatever template its author baked in. So the verdict is read from the
template itself, through two sources.

**The handle caps (llama.cpp engines, free).** The one ``GET /props`` after
spawn stamps ``chat_template_caps`` on the engine handle. llama.cpp builds that
map by executing the template SYMBOLICALLY with ``reasoning_effort`` bound and
checking whether the variable was actually read (``common/jinja/caps.cpp``), so
``supports_reasoning_effort`` is exact for the server that will render the
prompt. The map carries nine booleans (tools, tool calls, system role, parallel
tool calls, preserve reasoning, reasoning effort, string/typed content, object
arguments) and NO ``enable_thinking`` entry -- the toggle therefore falls
through to the probe below.

**The differential probe (every engine).** Mirrors
``engines.system_role_capability``: same ``_load_capability_tokenizer`` seam
(template files only -- never weights, never the network), same ``lru_cache``
keying on (engine, path), same graceful default. It renders the generation
prompt with ``reasoning_effort`` low vs high (llama.cpp and mlx_vlm both bind
the ``reasoning_strength`` alias alongside, so this does too), then with
``enable_thinking`` true vs false, and reads the verdict from what CHANGES. A
template that cannot render at all is never blamed: no lever, no thinker, no
log spam.

``is_thinker`` comes from the same renders -- a template that differs under
``enable_thinking``, reads ``reasoning_effort``, or whose generation prompt
leaves a thinking block OPEN. ``generation_hints.supports_thinking`` stays what
it has always been: a catalog display hint, never the runtime authority.

One ordering nuance, deliberate: the prompt for a turn is composed BEFORE the
child spawns, so the very first turn on a freshly selected model has no handle
to read and answers from the probe alone. From the second turn on, the caps can
only ever upgrade the verdict to ``NATIVE_EFFORT`` -- the case where llama.cpp's
own Jinja reads a template ours could not.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union

from src.agents.reasoning_effort import ReasoningLever
from src.core.logging import logger

_USER_ONLY = [{"role": "user", "content": "hi"}]

# Thinking-block delimiters used by the families that ship one. The probe looks
# for an OPEN one in the generation prompt: several templates emit an already
# CLOSED empty block (Qwen3 with thinking off) and that is the opposite signal.
_THINKING_MARKERS = (
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<|think|>", "<|/think|>"),
    ("[THINK]", "[/THINK]"),
    ("<|channel|>analysis", "<|end|>"),
)


@dataclass(frozen=True)
class LeverVerdict:
    """What this artifact's template honours, and whether it reasons at all."""

    lever: ReasoningLever
    is_thinker: bool


_NO_LEVER = LeverVerdict(lever=ReasoningLever.NONE, is_thinker=False)


def _render(tokenizer: Any, **template_kwargs: Any) -> Optional[str]:
    """One generation-prompt render, or None when the template refuses it."""
    try:
        rendered = tokenizer.apply_chat_template(
            _USER_ONLY, add_generation_prompt=True, tokenize=False, **template_kwargs
        )
    except Exception:
        # Not a signal about reasoning: the template may reject this shape for
        # entirely unrelated reasons (a missing global, a system-role guard).
        return None
    return rendered if isinstance(rendered, str) else None


def _opens_a_thinking_block(rendered: str) -> bool:
    """True when the generation prompt LEAVES a thinking block open."""
    for opener, closer in _THINKING_MARKERS:
        opened = rendered.rfind(opener)
        if opened >= 0 and rendered.rfind(closer) < opened:
            return True
    return False


def tokenizer_reasoning_lever(tokenizer: Any) -> LeverVerdict:
    """The verdict for a tokenizer-shaped object exposing ``apply_chat_template``.

    Pure and differential. Returns the graceful default (no lever, not a
    thinker) for anything it cannot render -- a wrong verdict would either
    silence a reasoning model or teach a plain one a protocol it does not know.
    """
    if tokenizer is None:
        return _NO_LEVER
    baseline = _render(tokenizer)
    if baseline is None:
        return _NO_LEVER

    low = _render(tokenizer, reasoning_effort="low", reasoning_strength="low")
    high = _render(tokenizer, reasoning_effort="high", reasoning_strength="high")
    if low is not None and high is not None and low != high:
        # The template READS the effort: a model that grades its own reasoning
        # is a reasoning model.
        return LeverVerdict(lever=ReasoningLever.NATIVE_EFFORT, is_thinker=True)

    thinking_on = _render(tokenizer, enable_thinking=True)
    thinking_off = _render(tokenizer, enable_thinking=False)
    if thinking_on is not None and thinking_off is not None and thinking_on != thinking_off:
        return LeverVerdict(lever=ReasoningLever.NATIVE_TOGGLE, is_thinker=True)

    return LeverVerdict(lever=ReasoningLever.NONE, is_thinker=_opens_a_thinking_block(baseline))


@lru_cache(maxsize=64)
def _cached(engine_name: str, local_path: str) -> LeverVerdict:
    from src.core import config

    engine = config.LLM_Engine
    try:
        tokenizer = engine._load_capability_tokenizer(local_path)
    except Exception:
        logger.warning(
            f"[{engine_name}] reasoning-lever detection: could not load a "
            f"tokenizer for {local_path}; assuming no lever",
            exc_info=True,
        )
        return _NO_LEVER
    verdict = tokenizer_reasoning_lever(tokenizer)
    logger.info(
        f"[{engine_name}] reasoning lever for {local_path}: "
        f"{verdict.lever.value} (thinker={verdict.is_thinker})"
    )
    return verdict


def reset_lever_cache() -> None:
    """Drop the memoized verdicts (tests; a swapped artifact keeps its path)."""
    _cached.cache_clear()


def model_reasoning_lever(local_path: Union[str, Path, None], llm_id: Any = None) -> LeverVerdict:
    """The lever verdict for the model at ``local_path``.

    ``llm_id`` identifies which model the caller is planning for, so the handle
    caps are only trusted when the child currently up is serving THAT model --
    a turn planned for another one must not inherit its capabilities.

    Cached per (engine, path): the probe reads only template files, and the
    answer is stable for a given artifact, so a turn pays it at most once per
    model per process.
    """
    if not local_path:
        return _NO_LEVER

    from src.core import config

    engine = getattr(config, "LLM_Engine", None)
    engine_name = getattr(engine, "__name__", "engine") if engine else "engine"

    caps_probe = getattr(engine, "chat_template_caps", None) if engine else None
    if callable(caps_probe):
        try:
            caps = caps_probe(llm_id)
        except Exception:
            # A capability read must never sink a turn: fall through to the
            # probe, which answers from the artifact on disk.
            logger.warning(
                f"[{engine_name}] could not read the chat-template capabilities "
                f"of the running child; probing the template instead",
                exc_info=True,
            )
            caps = None
        if isinstance(caps, dict) and caps.get("supports_reasoning_effort") is True:
            return LeverVerdict(lever=ReasoningLever.NATIVE_EFFORT, is_thinker=True)

    return _cached(engine_name, str(local_path))

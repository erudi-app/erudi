"""Chat-template access straight from a GGUF's KV header (#313, #291).

The capability probes (tool-calling, tool-call wire format, system-role support)
only ever need one string: the model's Jinja chat template. Getting it through
``transformers.AutoTokenizer.from_pretrained(gguf_file=...)`` used to pull the
whole ``modeling_auto`` import graph -- scikit-learn, scipy BLAS and their native
DLLs -- and in the frozen Windows build that import DEADLOCKS when it happens off
the main thread: ``LoadLibraryExW`` never returns, zero CPU, forever. Every first
chat turn against a GGUF model hung on it (#313), as did download finalization
(#291).

This reads the template out of the GGUF key-value header instead (``import
gguf`` is 0.16s, the header read is a few seconds on a 9GB artifact and is cached
per model) and renders it with plain ``jinja2``. No transformers, no native
extension loading, no deadlock.

The probes stay DIFFERENTIAL: they render the template twice and compare, rather
than inspecting the template text. A marker/substring heuristic would silently
flip verdicts (Gemma's template renders fine until a system message is present).
So the object returned here quacks like a tokenizer -- ``chat_template`` plus
``apply_chat_template`` -- and every existing probe keeps working unchanged.

Environment parity matters as much as the template itself. ``apply_chat_template``
injects globals that real templates depend on, and a missing one turns a rendering
failure into a WRONG verdict rather than an error: Gemma rejects system messages
by calling ``raise_exception(...)``, so an environment without it fails BOTH
renders, the differential sees "unrenderable" and returns the graceful default,
and the fold silently stops firing for the one model it exists for.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

from src.core.logging import logger

_CHAT_TEMPLATE_KEY = "tokenizer.chat_template"
_TOKENS_KEY = "tokenizer.ggml.tokens"
_BOS_ID_KEY = "tokenizer.ggml.bos_token_id"
_EOS_ID_KEY = "tokenizer.ggml.eos_token_id"


def _raise_exception(message: str) -> None:
    """``raise_exception`` as templates expect it (Gemma's system-role guard)."""
    raise jinja2.exceptions.TemplateError(message)


def _strftime_now(fmt: str) -> str:
    """``strftime_now``: templates that stamp the current date."""
    return datetime.datetime.now().strftime(fmt)


def _tojson(obj: Any, **kwargs: Any) -> str:
    """``tojson`` filter, used by tool-calling templates to embed signatures."""
    kwargs.pop("ensure_ascii", None)
    return json.dumps(obj, ensure_ascii=False, **kwargs)


def _build_environment() -> ImmutableSandboxedEnvironment:
    """A Jinja environment matching what ``apply_chat_template`` provides.

    Sandboxed because chat templates are attacker-influenced content: they ship
    inside a GGUF downloaded from the Hub. ``loopcontrols`` is required, not
    cosmetic -- templates using ``{% break %}`` fail to COMPILE without it, which
    would read as "unrenderable" and quietly return the graceful default.
    """
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols"],
    )
    env.globals["raise_exception"] = _raise_exception
    env.globals["strftime_now"] = _strftime_now
    env.filters["tojson"] = _tojson
    return env


class GgufChatTemplate:
    """Tokenizer-shaped view over a GGUF's chat template.

    Exposes the two attributes the capability probes consume, so
    ``tokenizer_declares_tools``, ``tokenizer_supports_system_role`` and
    ``compute_wire_tools`` all work against it with no changes.
    """

    def __init__(self, chat_template: str, bos_token: str = "", eos_token: str = ""):
        self.chat_template = chat_template
        self.bos_token = bos_token
        self.eos_token = eos_token

    def apply_chat_template(
        self,
        messages: Sequence[dict],
        add_generation_prompt: bool = False,
        tokenize: bool = False,
        tools: Optional[Sequence[dict]] = None,
        **template_kwargs: Any,
    ) -> str:
        """Render the template. Raises whatever the template raises -- that
        signal IS the probe (``TemplateError`` from ``raise_exception`` is how a
        template rejects the system role).

        ``tokenize`` is accepted for signature compatibility and ignored: there
        is no vocabulary here, and every probe calls with ``tokenize=False``.

        Any other keyword is BOUND as a template variable, exactly as
        ``transformers.apply_chat_template`` binds its extra kwargs and as both
        local servers bind theirs (``chat_template_kwargs`` on llama-server,
        ``to_template_kwargs`` on mlx_vlm). The differential reasoning-lever
        probe depends on it: dropping the bound variables would make its two
        renders identical and every GGUF would read as "no lever".
        """
        template = _build_environment().from_string(self.chat_template)
        return template.render(
            messages=list(messages),
            add_generation_prompt=add_generation_prompt,
            tools=list(tools) if tools else None,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            **template_kwargs,
        )


def _token_at(reader: Any, index: Any) -> str:
    """Resolve one vocabulary entry without materialising the whole vocab."""
    field = reader.fields.get(_TOKENS_KEY)
    if field is None or index is None:
        return ""
    try:
        value = field.contents(int(index))
    except Exception:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def load_gguf_chat_template(gguf_path: Union[str, Path]) -> Optional[GgufChatTemplate]:
    """Read the chat template out of ``gguf_path``'s KV header.

    Returns None when the artifact has no template or cannot be read; callers
    treat that as "unknown" and keep their graceful defaults. Never raises.
    """
    try:
        from gguf import GGUFReader

        reader = GGUFReader(str(gguf_path))
        field = reader.fields.get(_CHAT_TEMPLATE_KEY)
        if field is None:
            logger.info(f"GGUF has no embedded chat template: {gguf_path}")
            return None
        template = field.contents()
        if isinstance(template, bytes):
            template = template.decode("utf-8", errors="replace")
        if not template or not isinstance(template, str):
            return None

        bos_id = reader.fields.get(_BOS_ID_KEY)
        eos_id = reader.fields.get(_EOS_ID_KEY)
        return GgufChatTemplate(
            chat_template=template,
            bos_token=_token_at(reader, bos_id.contents() if bos_id else None),
            eos_token=_token_at(reader, eos_id.contents() if eos_id else None),
        )
    except Exception:
        logger.warning(f"Could not read the chat template from {gguf_path}", exc_info=True)
        return None

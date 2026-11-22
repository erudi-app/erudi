"""Log-formatting helpers shared across domains, engines, agents and ingestion.

Erudi is a fully local app: logging CONTENT (prompts, queries, filenames,
previews) is deliberate policy — nothing ever leaves the machine. What we do
bound is LINE SIZE, so one giant prompt cannot flood a log file. Use:

- user question / RAG query        → ``truncate_for_log(value, 2000)``
- model response preview           → ``truncate_for_log(value, 500)``
- misc previews (tool args, files) → ``truncate_for_log(value, 200)``

NEVER pass base64/image bytes here — log their byte length instead.
"""

from __future__ import annotations

import logging
import re

DEFAULT_LIMIT = 500

_TRUNCATION_SUFFIX = "… [+{n} chars]"

# `data:<mime>` then optional `;name=value` parameters, then `;base64,<payload>`.
_DATA_URL = re.compile(r"(data:[\w.+\-/]+(?:;[\w.+\-]+=[\w.+\-%]*)*;base64,)([A-Za-z0-9+/=_\-]+)")


def reduce_data_urls(text: str) -> str:
    """Replace the payload of every ``data:<mime>;base64,`` URL by its length.

    A failed image decode quotes the whole URL back in its exception text, so
    this is the net under every sink that can carry one (the log handlers, the
    Details line of a failed turn).
    """
    if "base64," not in text:
        return text
    return _DATA_URL.sub(lambda m: f"{m.group(1)}<{len(m.group(2))} chars>", text)


_TRACEBACK_FORMATTER = logging.Formatter()


def reduce_record(record: logging.LogRecord) -> None:
    """Reduce the data URLs of a record's message and traceback, in place.

    The message is rendered once and the traceback text cached on the record
    (``Formatter.format`` reuses ``exc_text``), so every formatter and handler
    sees the reduced text. Never raises: a malformed log call (``"%s %s" % (1,)``)
    must keep failing, or not, exactly as it would have without this.
    """
    try:
        message = record.getMessage()
        if "base64," in message:
            record.msg, record.args = reduce_data_urls(message), None
        if record.exc_info and not record.exc_text:
            record.exc_text = _TRACEBACK_FORMATTER.formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = reduce_data_urls(record.exc_text)
    except Exception:
        # Redaction is a net, not a precondition: a record it cannot read goes
        # through untouched and the handler reports the malformed call itself.
        pass


def install_data_url_record_factory() -> None:
    """Reduce data URLs on every record this process creates, whatever handler
    later formats it. For a process that does not own its logging setup (the
    MLX child: mlx-vlm and uvicorn build their handlers after this runs).
    """
    previous = logging.getLogRecordFactory()
    if getattr(previous, "reduces_data_urls", False):
        return

    def factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        reduce_record(record)
        return record

    factory.reduces_data_urls = True
    logging.setLogRecordFactory(factory)


def truncate_for_log(value, limit: int = DEFAULT_LIMIT) -> str:
    """Render any value as a stripped, size-bounded string for log lines.

    Converts ``value`` to ``str``, strips surrounding whitespace, and — when
    longer than ``limit`` — truncates with an explicit ``… [+N chars]``
    suffix so the reader knows content was elided. Never raises: hostile
    ``__str__``/``__repr__`` degrade to a placeholder, and an invalid
    ``limit`` falls back to the default.
    """
    try:
        text = str(value)
    except Exception:
        try:
            text = repr(value)
        except Exception:
            return "<unloggable>"

    try:
        text = text.strip()
    except Exception:
        return "<unloggable>"

    try:
        limit = int(limit)
    except Exception:
        limit = DEFAULT_LIMIT
    if limit <= 0:
        limit = DEFAULT_LIMIT

    if len(text) <= limit:
        return text
    return text[:limit] + _TRUNCATION_SUFFIX.format(n=len(text) - limit)

"""Keep a spawned child's output pipe drained, and keep its last words.

A `subprocess.Popen` pipe is a fixed-size kernel buffer (64 KiB measured on
macOS; Windows pipes created with `nSize=0` are typically smaller). Once it is
full and nothing reads it, the child's next `write()` blocks -- the process
stays alive, stops making progress, and reports no error. `llama-server` writes
its startup banner, the GGUF metadata dump, and a line per request, so it
reaches that ceiling on its own during an ordinary session (#361).

Draining it in a reader thread removes the blocking condition by construction.
Since we have to read the pipe anyway, we keep a bounded tail of what came out:
that is what the crash report quotes when the child dies before becoming ready,
instead of pointing the user at logs nothing ever wrote.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import IO, Optional

from src.core.logging import logger


class ChildOutputDrainer:
    """Reads a child's merged stdout/stderr to EOF on a daemon thread.

    Each line is logged at DEBUG (so `llama-server`'s own diagnostics land in
    `backend/logs/app.log` for the first time) and appended to a bounded tail.

    The thread is a daemon: a wedged or unkillable child must never hold up
    interpreter shutdown. A read error is logged at WARNING and never raised --
    this object exists to keep a pipe empty, and a drainer that can raise into
    the engine's spawn path would be worse than the problem it solves.
    """

    def __init__(self, stream: Optional[IO[str]], *, name: str, tail_lines: int = 200) -> None:
        self._stream = stream
        self._name = name
        self._tail: deque[str] = deque(maxlen=tail_lines)
        self._lock = threading.Lock()
        self._first_line = threading.Event()
        self._thread = threading.Thread(
            target=self._pump,
            name=f"{name}-output-drainer",
            daemon=True,
        )
        self._thread.start()

    def _pump(self) -> None:
        stream = self._stream
        if stream is None:
            return
        try:
            # `iter(readline, "")` rather than `for line in stream`: iterating a
            # text stream read-aheads in blocks, which would hold lines back
            # until a block fills. readline returns as each line arrives.
            for line in iter(stream.readline, ""):
                line = line.rstrip("\r\n")
                with self._lock:
                    self._tail.append(line)
                self._first_line.set()
                logger.debug(f"[{self._name}] {line}")
        except Exception as exc:
            # EOF is not an exception: readline returns "" and the loop ends.
            # Raising here means the pipe broke under us -- a decode error on
            # a byte the child wrote, a stream closed from another thread. The
            # drainer is gone from then on and the child can wedge once its
            # pipe fills (#361), so this is worth a maintainer's attention.
            logger.warning(f"[{self._name}] output drainer stopped: {type(exc).__name__}: {exc}")
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def tail(self, max_chars: int = 2000) -> str:
        """The child's last lines, newest last, truncated to `max_chars`.

        Safe to call at any time, from any thread, including while the child is
        still running.
        """
        with self._lock:
            text = "\n".join(self._tail)
        return text[-max_chars:] if len(text) > max_chars else text

    def wait_for_output(self, timeout: float) -> bool:
        """Block until the child has produced at least one line, or time out."""
        return self._first_line.wait(timeout)

    def is_running(self) -> bool:
        return self._thread.is_alive()

    def join(self, timeout: Optional[float] = None) -> None:
        """Wait for the reader thread to finish (i.e. for the pipe to hit EOF)."""
        self._thread.join(timeout)

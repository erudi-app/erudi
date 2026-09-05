"""Shared newline-delimited JSON lifecycle-event emitter.

Both the cross-platform launcher (``backend/run.py``) and the FastAPI
``lifespan`` run in the same process and share stdout. The Electron main
process (``frontend/src/main.js``) parses each stdout line as JSON and
tolerates interleaved non-JSON log lines. Keeping a single emitter here means
the launcher and the app agree on the wire format, and startup *progress*
(phase) events can originate from inside the lifespan.

Event shapes:
    - {"event": "starting", ...}                     (run.py)
    - {"event": "phase", "phase": "<name>"}          (lifespan, via emit_phase)
    - {"event": "ready", "port": N}                  (run.py)
    - {"event": "shutdown"}                           (run.py)
    - {"event": "startup_error", "code": "...", ...}  (run.py)
    - {"event": "engine_notice", "code": "...", "gpu_name": "...",
       "compute_capability": "...", "driver_cuda_version": "...",
       "required_cuda_version": "...", "raw": "..."}  (lifespan)

``engine_notice`` reports a GPU the bundled CUDA build cannot drive. It is NOT a
startup failure: the backend is healthy and serving, and the frontend holds the
event until after ``ready`` to show it as a decision (run on the processor
instead, or update the driver) rather than an error screen.

Every emitted event additionally carries a ``ts`` field (UTC ISO-8601 with
milliseconds and a ``Z`` suffix, e.g. ``2026-07-02T09:15:32.123Z``) so the
Electron log (``new Date().toISOString()``) and the backend log can be
correlated on a single timeline.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone


def _utc_timestamp() -> str:
    """UTC ISO-8601 with milliseconds and Z suffix (matches the backend log)."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def emit_event(payload: dict) -> None:
    """Print a structured JSON lifecycle event (newline-delimited) to stdout.

    A ``ts`` field (UTC ISO-8601 ms, Z) is stamped on every event; the
    caller's payload is not mutated.
    """
    print(json.dumps({**payload, "ts": _utc_timestamp()}), flush=True)


def emit_phase(phase: str) -> None:
    """Emit a startup-progress phase the frontend can surface on the loader.

    Phases are informational only — the frontend still gates readiness on the
    ``ready`` event / a confirming health check, never on a phase.
    """
    emit_event({"event": "phase", "phase": phase})

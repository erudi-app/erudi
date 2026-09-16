"""Minimal Chrome DevTools Protocol client for the Electron renderer.

UI selectors (verified against the frontend source, not against a running app):
- composer: the <textarea> rendered by frontend/src/components/QuestionInput.jsx, found as the
  enabled, visible textarea whose container also holds the send button (svg.lucide-arrow-right);
- submit: Enter keydown (QuestionInput.handleKeyDown: Enter without Shift calls handleSend),
  with a click on that send button as fallback when the text is still in the box.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from typing import Any

PERF_KEYS = ("JSHeapUsedSize", "JSHeapTotalSize", "Nodes", "Documents", "JSEventListeners", "LayoutCount", "RecalcStyleCount")

COMPOSER_JS = r"""
(() => {
  const all = Array.from(document.querySelectorAll('textarea'));
  const visible = all.filter(t => t.offsetParent !== null);
  const composer = visible.find(t => {
    let el = t.parentElement;
    for (let i = 0; i < 4 && el; i++, el = el.parentElement) {
      if (el.querySelector('svg.lucide-arrow-right')) return true;
    }
    return false;
  });
  return composer || null;
})()
"""


def http_json(url: str, timeout: float = 3.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def pick_page_target(targets: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The app window: a `page` target that is not DevTools; prefer the one on the app's hash routes."""
    pages = [t for t in targets if t.get("type") == "page" and not str(t.get("url", "")).startswith("devtools://")]
    app_pages = [t for t in pages if "#/erudi" in str(t.get("url", "")) or str(t.get("url", "")).startswith("file://")]
    chosen = (app_pages or pages or [None])[0]
    return chosen if chosen and chosen.get("webSocketDebuggerUrl") else None


class CdpError(RuntimeError):
    pass


class CdpClient:
    def __init__(self, port: int = 9222, host: str = "127.0.0.1"):
        self.base = f"http://{host}:{port}"
        self.ws = None
        self._id = 0
        self.target: dict[str, Any] | None = None
        self._lock = threading.RLock()  # the renderer sampler thread shares the socket

    def version(self) -> dict[str, Any] | None:
        try:
            return http_json(self.base + "/json/version")
        except (OSError, ValueError):
            return None

    def connect(self, timeout: float = 10.0) -> None:
        import websocket  # websocket-client

        target = pick_page_target(http_json(self.base + "/json/list"))
        if not target:
            raise CdpError("no app page target")
        self.target = target
        # suppress_origin: Chromium rejects DevTools websockets that carry a foreign Origin header.
        self.ws = websocket.create_connection(target["webSocketDebuggerUrl"], timeout=timeout, suppress_origin=True)
        self.call("Performance.enable")

    def close(self) -> None:
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:  # noqa: BLE001 - closing a dead socket is fine
                pass
            self.ws = None

    @property
    def connected(self) -> bool:
        return self.ws is not None

    def call(self, method: str, params: dict[str, Any] | None = None, timeout: float = 15.0) -> dict[str, Any]:
        with self._lock:
            return self._call(method, params, timeout)

    def _call(self, method: str, params: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
        if self.ws is None:
            raise CdpError("not connected")
        self._id += 1
        msg_id = self._id
        self.ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.ws.settimeout(max(0.1, deadline - time.monotonic()))
            raw = self.ws.recv()
            msg = json.loads(raw)
            if msg.get("id") != msg_id:
                continue  # an event or a stale reply
            if "error" in msg:
                raise CdpError(f"{method}: {msg['error']}")
            return msg.get("result", {})
        raise CdpError(f"{method}: timeout")

    def evaluate(self, expression: str, await_promise: bool = False, timeout: float = 15.0) -> Any:
        res = self.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": await_promise},
            timeout=timeout,
        )
        if "exceptionDetails" in res:
            raise CdpError(f"evaluate: {res['exceptionDetails'].get('text')}")
        return res.get("result", {}).get("value")

    # --- metrics -------------------------------------------------------------------------

    def renderer_metrics(self) -> dict[str, Any]:
        perf = self.call("Performance.getMetrics").get("metrics", [])
        out: dict[str, Any] = {m["name"]: m["value"] for m in perf if m.get("name") in PERF_KEYS}
        heap = self.call("Runtime.getHeapUsage")
        out["heap_used"] = heap.get("usedSize")
        out["heap_total"] = heap.get("totalSize")
        out["location_hash"] = self.evaluate("location.hash")
        return out

    def load_event_epoch_ms(self) -> float | None:
        return self.evaluate(
            "(() => { const n = performance.getEntriesByType('navigation')[0];"
            " return n && n.loadEventEnd > 0 ? performance.timeOrigin + n.loadEventEnd : null; })()"
        )

    # --- UI driving ----------------------------------------------------------------------

    def navigate_hash(self, hash_route: str) -> None:
        self.evaluate(f"location.hash = {json.dumps(hash_route)}")

    def composer_state(self) -> dict[str, Any] | None:
        return self.evaluate(f"(() => {{ const t = {COMPOSER_JS}; return t ? {{disabled: t.disabled, value: t.value}} : null; }})()")

    def type_into_composer(self, text: str) -> None:
        if not self.evaluate(f"(() => {{ const t = {COMPOSER_JS}; if (!t || t.disabled) return false; t.focus(); return document.activeElement === t; }})()"):
            raise CdpError("composer textarea not found or disabled")
        self.call("Input.insertText", {"text": text})

    def press_enter(self) -> None:
        base = {"key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13}
        self.call("Input.dispatchKeyEvent", {"type": "rawKeyDown", **base})
        self.call("Input.dispatchKeyEvent", {"type": "keyUp", **base})

    def click_send_button(self) -> bool:
        return bool(
            self.evaluate(
                f"(() => {{ const t = {COMPOSER_JS}; if (!t) return false; let el = t.parentElement;"
                " for (let i = 0; i < 4 && el; i++, el = el.parentElement) {"
                "  const svg = el.querySelector('svg.lucide-arrow-right');"
                "  if (svg) { const b = svg.closest('button'); if (b && !b.disabled) { b.click(); return true; } return false; } }"
                " return false; })()"
            )
        )

    def reload(self) -> None:
        self.call("Page.reload", {"ignoreCache": False})

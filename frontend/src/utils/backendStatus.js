// Pure helpers for interpreting backend lifecycle events.
//
// The backend (`backend/run.py`) emits newline-JSON lifecycle events that the
// Electron main process forwards to the renderer over the "backend-event" IPC
// channel: `{event: "starting"|"ready"|"shutdown"|"startup_error", code?, message?}`.
// `main.js` adds a few of its own codes when the spawn itself fails.
//
// The interpretation stays a plain function (no React) so it unit-tests
// cleanly and the renderer wiring (preload bridge + error screen) stays thin.
// The user-facing copy lives in the `errors` catalog (#385): each code maps to
// a `errors:backend.codes.<key>` subtree and is resolved through `i18n.t` at
// call time so the descriptor speaks the active language.

import i18n from "../i18n";

// Backend / main-process error code -> `errors:backend.codes.*` key.
const MESSAGE_KEYS = {
  DATA_DIR_IN_USE: "dataDirInUse",
  PORT_IN_USE: "portInUse",
  NO_PORT_AVAILABLE: "noPortAvailable",
  PORT_TIMEOUT: "portTimeout",
  CRASH_BEFORE_READY: "crashBeforeReady",
  IMPORT_ERROR: "importError",
  DATA_PREP_ERROR: "dataPrepError",
  DATABASE_ERROR: "databaseError",
  CUDA_RUNTIME_ERROR: "cudaRuntimeError",
  MISSING_DEPENDENCY: "missingDependency",
  BACKEND_NOT_FOUND: "backendNotFound",
  BACKEND_STARTUP_FAILED: "backendStartupFailed",
  BACKEND_EXIT_ERROR: "backendExitError",
  BACKEND_SPAWN_FAILED: "backendSpawnFailed",
  UNEXPECTED_ERROR: "unexpectedError",
  POLLING_ERROR: "pollingError",
  // Renderer-side: the /health poll never succeeded within the wait window. This
  // is the catch-all for a silent hang (no startup_error event arrived).
  BACKEND_UNREACHABLE: "backendUnreachable",
};

const FALLBACK_KEY = "unknown";

/**
 * Turn a backend lifecycle event (or a synthetic {code}) into a user-facing
 * descriptor: {code, title, detail, hint, raw}. Unknown/missing codes fall back
 * to a generic descriptor. `raw` carries the backend's own message when present.
 */
export function describeBackendError(evt) {
  const code = (evt && evt.code) || "UNKNOWN";
  const key = MESSAGE_KEYS[code] || FALLBACK_KEY;
  const message = evt && typeof evt.message === "string" ? evt.message.trim() : "";
  return {
    code,
    title: i18n.t(`errors:backend.codes.${key}.title`),
    detail: i18n.t(`errors:backend.codes.${key}.detail`),
    hint: i18n.t(`errors:backend.codes.${key}.hint`),
    raw: message || null,
  };
}

/** True when the event signals the backend failed to start. */
export function isStartupError(evt) {
  return !!evt && evt.event === "startup_error";
}

/** True when the event signals the backend is up. */
export function isBackendReady(evt) {
  return !!evt && (evt.event === "ready" || evt.event === "backend_ready");
}

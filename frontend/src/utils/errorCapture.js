// Catching what nothing else catches: uncaught renderer exceptions and
// unhandled promise rejections.
//
// Without this, an exception that escapes React reaches no log at all — the
// window goes blank and the two log files say nothing about why. With it, the
// error lands in `erudi-backend.log` through the existing `renderer-log`
// bridge, and in an in-memory buffer the Diagnostics page reads.
//
// The hard part is not catching; it is not making things worse:
//
//   - **No recursion.** Everything the handler does runs inside a try/catch
//     that swallows. A logger that throws while reporting an error would
//     otherwise raise a second error, which reaches this same handler.
//   - **No flood.** A render loop or a failing poll produces the same error
//     hundreds of times a second. Identical errors are counted, not logged
//     again, so one defect costs one line in the log file rather than a
//     megabyte of it.
//   - **No bridge, no problem.** `window.logAPI` is absent in a browser, in
//     tests and before the preload script has run. That is a no-op, never a
//     throw.

import { createLogger } from "./logger";

const log = createLogger("renderer:uncaught");

/** Entries kept in memory for the panel. Old entries fall off the front. */
export const SESSION_ERROR_CAP = 50;

/** Characters kept of a message and of a stack. */
const MAX_MESSAGE_CHARS = 1000;
const MAX_STACK_CHARS = 2000;

/** In-memory buffer, oldest first. Never persisted, never sent anywhere. */
let sessionErrors = [];

/** Signature -> entry, so a repeat is found in constant time. */
let bySignature = new Map();

/** The uninstaller of the single active installation, or null. */
let activeUninstall = null;

/** Truncate and stringify anything a handler was handed. */
function asText(value, limit) {
  try {
    if (value === null || value === undefined) return "";
    const text = typeof value === "string" ? value : String(value);
    return text.length > limit ? `${text.slice(0, limit)}… [+${text.length - limit}]` : text;
  } catch {
    return "";
  }
}

/**
 * Record one uncaught error, deduplicated by origin + message + stack.
 *
 * Never throws, whatever the bridge or the arguments do.
 *
 * @param {{origin: string, message: *, stack?: *}} error - What happened and where from.
 * @returns {void}
 */
export function recordSessionError(error) {
  try {
    const origin = asText(error?.origin, 60) || "unknown";
    const message = asText(error?.message, MAX_MESSAGE_CHARS) || "Unknown error";
    const stack = asText(error?.stack, MAX_STACK_CHARS);
    const signature = `${origin}::${message}::${stack.slice(0, 200)}`;

    const seen = bySignature.get(signature);
    if (seen) {
      // A repeat. Count it and say nothing: this is the flood guard.
      seen.count += 1;
      seen.timestamp = new Date().toISOString();
      return;
    }

    const entry = {
      timestamp: new Date().toISOString(),
      origin,
      message,
      stack,
      count: 1,
    };
    sessionErrors.push(entry);
    bySignature.set(signature, entry);

    while (sessionErrors.length > SESSION_ERROR_CAP) {
      const dropped = sessionErrors.shift();
      for (const [key, value] of bySignature) {
        if (value === dropped) {
          bySignature.delete(key);
          break;
        }
      }
    }

    // Logging comes last and inside the same swallow: if the bridge is gone
    // the entry is still in the buffer the panel reads.
    log.error(`${origin}: ${message}`, stack || undefined);
  } catch {
    // A capture mechanism that throws is worse than no capture mechanism.
  }
}

/** The buffer, oldest first. A copy, so a caller cannot mutate it. */
export function getSessionErrors() {
  return sessionErrors.map((entry) => ({ ...entry }));
}

/** Empty the buffer. Used by tests and by the panel's "clear" affordance. */
export function resetSessionErrors() {
  sessionErrors = [];
  bySignature = new Map();
}

/**
 * Attach the `error` and `unhandledrejection` listeners.
 *
 * Idempotent: a second call attaches nothing and hands back the *same*
 * uninstaller, so a double mount (React StrictMode, a hot reload) cannot
 * double-log and cannot strand the listeners either.
 *
 * @returns {() => void} Removes the listeners of the single installation.
 */
export function installGlobalErrorCapture() {
  if (typeof window === "undefined") return () => {};
  if (activeUninstall) return activeUninstall;

  const onError = (event) => {
    recordSessionError({
      origin: "window.onerror",
      message: event?.message || event?.error?.message,
      stack: event?.error?.stack,
    });
  };

  const onRejection = (event) => {
    const reason = event?.reason;
    recordSessionError({
      origin: "unhandledrejection",
      message: reason?.message ?? reason,
      stack: reason?.stack,
    });
  };

  window.addEventListener("error", onError);
  window.addEventListener("unhandledrejection", onRejection);

  activeUninstall = () => {
    window.removeEventListener("error", onError);
    window.removeEventListener("unhandledrejection", onRejection);
    activeUninstall = null;
  };
  return activeUninstall;
}

// Deciding which of the Diagnostics page's own error records describe the
// ENVIRONMENT (no network, a remote service unreachable, the user's disk or
// port, not Erudi being wrong) rather than a real defect (#485, broadened
// per the follow-up product decision below).
//
// `isEnvironmental` is used two ways, both reading the exact same predicate
// so the badge, the "Recent errors" list and the copied report can never
// disagree:
//   - `diagnosticsReport.js`'s `mergeRecentErrors` calls `isHiddenFromDiagnostics`
//     to drop an environmental ERROR/CRITICAL from what the page renders and
//     copies (WARNINGs are never touched by this filter -- see that file).
//     The log files on disk keep everything; this is presentation-only.
//   - `shared/hooks/useBugCounter.js` calls `isCountableError`/`countNewErrors`
//     on that same already-filtered list to decide the badge's number.
//
// Every pattern below is grounded in a real raise/log site, cited in its own
// comment (grepped, not guessed), plus one forward-looking pattern for a
// shape the issue names explicitly but that no current code path reaches.
// Deliberately NOT environmental, despite superficially smelling like it: any
// failure of Erudi's OWN local backend or its inference children (a dead
// llama.cpp/mlx child, a request to 127.0.0.1 that dies, a mid-stream socket
// reset) -- the machine being the user's does not make our process's own
// death legitimate, so none of these patterns match on host/path alone
// without also matching text specific to a genuinely external cause.

/**
 * Lowercased substrings that mark a no-network failure reaching
 * huggingface.co, mirroring the two backend detectors this predicate has to
 * agree with: `backend/src/ingestion/embedding_model.py`'s
 * `_OFFLINE_MESSAGE_MARKERS` and `backend/src/domains/llms/services.py`'s
 * (same name). Kept as one list here since both scopes below draw from it.
 */
const OFFLINE_NETWORK_MARKERS = [
  "getaddrinfo",
  "nodename nor servname",
  "name or service not known",
  "temporary failure in name resolution",
  "network is unreachable",
  "offline mode is enabled",
  "outgoing traffic has been disabled",
  "connection error",
  "max retries exceeded",
  "couldn't connect to 'https://huggingface.co'",
  "cannot reach https://huggingface.co",
  "we couldn't connect to 'https://huggingface.co'",
];

/** Lowercased HTTP-status markers of a HuggingFace-side outage or throttling. */
const HF_SERVICE_TROUBLE_MARKERS = [
  "429",
  "too many requests",
  "500 server error",
  "502",
  "503",
  "504",
  "server error",
  "service unavailable",
  "bad gateway",
  "gateway timeout",
];

/**
 * Environmental patterns, each documented with the real record it exists to
 * match. A record is environmental when it matches ANY of these; everything
 * else -- including every other 5xx, every other HuggingFace API failure, and
 * a request that failed against Erudi's OWN local backend -- counts.
 */
const ENVIRONMENTAL_PATTERNS = [
  {
    name: "hf_download_offline",
    // backend/src/domains/llms/services.py's OFFLINE_DOWNLOAD_MESSAGE
    // ("you appear to be offline - check your connection and retry"),
    // raised as a 503 HuggingFaceAPIException (services.py:811, guarded by
    // `_is_offline_download_error`) when a model download starts with no
    // network. Matches both the backend.log record ("... -> 503
    // HUGGINGFACE_API_ERROR: you appear to be offline - check your
    // connection and retry") and its renderer-side echoes: the api.failure
    // entry `tracedFetch`/`APIClient` log for that response, and
    // DownloadModalContext's "Download job N failed" line, which carries the
    // same `error_message` text.
    test: (message) => message.includes("you appear to be offline"),
  },
  {
    name: "embedding_model_download_offline",
    // backend/src/ingestion/embedding_model.py's `_run_download` (line 233),
    // which logs the RAW exception at ERROR ("Embedding model download
    // failed: <exc>") rather than the friendly OFFLINE_ERROR_MESSAGE (that
    // mapping, `_describe_download_error`, only reaches the download-status
    // API response, not the log line). Scoped to that one log line --
    // requiring the prefix as well as a network marker -- so an unrelated
    // failure whose text happens to share a marker (a local Postgres
    // "connection error", for instance) still counts.
    test: (message) =>
      message.includes("embedding model download failed") &&
      OFFLINE_NETWORK_MARKERS.some((marker) => message.includes(marker)),
  },
  {
    name: "hf_download_gated_requires_auth",
    // backend/src/domains/llms/services.py:798-805: a GatedRepoError or a
    // 401/403 from the Hub during a download is re-raised as
    // UnsupportedPlatformException("<model>", "requires HuggingFace
    // authentication and cannot be downloaded anonymously") -- not offline,
    // not a bug, just a repo this install has no token for. Reaches
    // `_run_download_task`'s ERROR log (endpoints.py) as "... not supported
    // on this platform: requires HuggingFace authentication and cannot be
    // downloaded anonymously", and the same text lands in the download job's
    // `error_message`, so it also matches DownloadModalContext's echo.
    test: (message) =>
      message.includes("requires huggingface authentication and cannot be downloaded anonymously"),
  },
  {
    name: "hf_download_rate_limited_or_service_error",
    // backend/src/domains/llms/services.py:805: an HfHubHTTPError that is
    // NEITHER the gated case (401/403, matched above) NOR an offline one
    // (matched above) is bare re-raised as-is -- a genuine HF-side 429
    // (rate limited) or 5xx (HF is down/degraded). It surfaces through
    // `_run_download_task`'s outer `except Exception` (endpoints.py) as the
    // raw `requests`/huggingface_hub HTTPError text, which names the
    // huggingface.co URL and the status ("429 Client Error: Too Many
    // Requests for url: https://huggingface.co/..." / "503 Server Error:
    // Service Unavailable for url: https://huggingface.co/..."). The same
    // pacing+retry-then-bare-raise shape backs `_call_with_429_retry`
    // (core/config.py) for search/metadata calls, but those stay WARNING
    // (see the predicate's own tests and this file's report for what is
    // already excluded by level).
    test: (message) =>
      message.includes("huggingface.co") &&
      HF_SERVICE_TROUBLE_MARKERS.some((marker) => message.includes(marker)),
  },
  {
    name: "disk_full",
    // No dedicated exception -- ENOSPC bubbles up as a plain OSError through
    // the two generic task-boundary handlers that log it at ERROR with the
    // OS's own wording: backend/src/domains/llms/endpoints.py's
    // `_run_download_task` ("Download job N failed for LLM ...: [Errno 28]
    // No space left on device") and
    // backend/src/domains/knowledge_base/services.py's `_ingest_one_file`
    // ("KB N: ingestion failed for <file>: [Errno 28] No space left on
    // device") when `add_kb_chunks` or the temp-to-final move runs out of
    // room. POSIX and Windows phrase it differently, so both are matched.
    test: (message) =>
      message.includes("no space left on device") ||
      message.includes("not enough space on the disk") ||
      message.includes("disk quota exceeded"),
  },
  {
    name: "port_in_use",
    // backend/run.py: every scanned port (27182-27199) is already held by
    // something else (most often a leftover process, sometimes another
    // app), so `find_available_port` and the same-window `kill_port_process`
    // reclaim attempt both fail. `log_startup_failure` writes "startup_error
    // NO_PORT_AVAILABLE: Ports 27182-27199 all busy, failed to free <port>"
    // at ERROR to backend.log AND to stderr, which main.js echoes into
    // erudi-backend.log -- the one copy still readable when the backend
    // never came up at all. The sibling startup_error codes (IMPORT_ERROR,
    // DATA_PREP_ERROR, CRASH_BEFORE_READY, UNEXPECTED_ERROR, POLLING_ERROR)
    // are internal failures, not the machine's fault, and stay counted.
    test: (message) => message.includes("no_port_available"),
  },
  {
    name: "hub_or_forge_unreachable",
    // No renderer code path fetches huggingface.co or github.com directly
    // today: HuggingFace downloads and metadata are proxied through the
    // backend (matched above), and the updater talks to GitHub releases from
    // the Electron main process and logs its failures at WARNING, already
    // excluded by level before this predicate ever runs. Kept as defense in
    // depth for the shape the issue calls out explicitly -- a "Failed to
    // fetch" / timeout toward one of those hosts -- and covered by a
    // synthetic-record test rather than a live one.
    test: (message) =>
      (message.includes("failed to fetch") || message.includes("timed out")) &&
      (message.includes("huggingface.co") || message.includes("github.com")),
  },
];

/**
 * True when `entry` describes the environment (no network, a remote service
 * unreachable) rather than a defect in the app itself.
 * @param {{message?: string}} entry - A merged Diagnostics record.
 * @returns {boolean}
 */
export function isEnvironmental(entry) {
  const message = String(entry?.message ?? "").toLowerCase();
  if (!message) return false;
  return ENVIRONMENTAL_PATTERNS.some((pattern) => pattern.test(message));
}

/** Levels the environmental filter applies to. CRITICAL is strictly worse than ERROR. */
const COUNTABLE_LEVELS = new Set(["ERROR", "CRITICAL"]);

/**
 * True when `entry` should be hidden from the Diagnostics page entirely --
 * the "Recent errors" list, the copied report, and by construction the badge
 * that counts from the same list. Only ERROR/CRITICAL records are ever
 * hidden this way: a WARNING is never environmental-filtered, it is simply
 * never counted by `isCountableError` below.
 * @param {{level?: string, message?: string}} entry
 * @returns {boolean}
 */
export function isHiddenFromDiagnostics(entry) {
  return !!entry && COUNTABLE_LEVELS.has(entry.level) && isEnvironmental(entry);
}

/**
 * True when `entry` is a real defect the bug counter should include: an
 * ERROR/CRITICAL record that is not environmental.
 * @param {{level?: string, message?: string}} entry
 * @returns {boolean}
 */
export function isCountableError(entry) {
  return !!entry && COUNTABLE_LEVELS.has(entry.level) && !isEnvironmental(entry);
}

/**
 * How many of `entries` are countable errors newer than `since`.
 * @param {object} args
 * @param {Array<{level?: string, message?: string, timestamp?: string}>} [args.entries] -
 *   The merged timeline from `mergeRecentErrors` (already collapsed).
 * @param {number} [args.since] - Epoch ms; entries at or before this do not count.
 * @returns {number}
 */
export function countNewErrors({ entries = [], since = 0 } = {}) {
  let count = 0;
  for (const entry of entries) {
    if (!isCountableError(entry)) continue;
    const ts = Date.parse(entry?.timestamp ?? "");
    if (Number.isFinite(ts) && ts > since) count += 1;
  }
  return count;
}

/** Above this, the badge reads "9+" instead of the exact number. */
export const BADGE_CAP = 9;

/** "3", "9+" -- the badge's own display text for a count. Empty at zero. */
export function badgeLabel(count) {
  if (!count || count <= 0) return "";
  return count > BADGE_CAP ? `${BADGE_CAP}+` : String(count);
}

/** localStorage key for the last time the Diagnostics page was opened. */
export const LAST_VISIT_STORAGE_KEY = "erudi.diagnostics.lastVisit";

// Notified synchronously by `markDiagnosticsVisited`, so a mounted badge
// clears immediately rather than waiting for its next poll -- the localStorage
// write is for the NEXT launch; this is for the current session.
const visitListeners = new Set();

function localStorageOrNull() {
  try {
    return typeof window !== "undefined" ? window.localStorage : null;
  } catch {
    return null;
  }
}

/**
 * The last time the Diagnostics page was opened, as epoch ms, or 0 when
 * there is no record of one (fresh install, storage unavailable/blocked).
 * @returns {number}
 */
export function getLastDiagnosticsVisit() {
  try {
    const raw = localStorageOrNull()?.getItem(LAST_VISIT_STORAGE_KEY);
    const parsed = raw ? Number(raw) : NaN;
    return Number.isFinite(parsed) ? parsed : 0;
  } catch {
    return 0;
  }
}

/**
 * Record that the Diagnostics page was just opened: persists `now` for the
 * next launch and tells every subscriber immediately, so the badge clears
 * without waiting for a poll.
 * @param {number} [now] - Epoch ms. Defaults to the current time.
 * @returns {number} The value recorded.
 */
export function markDiagnosticsVisited(now = Date.now()) {
  try {
    localStorageOrNull()?.setItem(LAST_VISIT_STORAGE_KEY, String(now));
  } catch {
    // Storage may be unavailable (private mode, blocked site data): the
    // badge still clears for this session via the listeners below; only
    // persistence across a relaunch is lost.
  }
  for (const listener of visitListeners) listener(now);
  return now;
}

/**
 * Watch Diagnostics visits. Called with the new visit timestamp whenever
 * `markDiagnosticsVisited` runs.
 * @param {(now: number) => void} listener
 * @returns {() => void} Unsubscribe.
 */
export function subscribeDiagnosticsVisit(listener) {
  visitListeners.add(listener);
  return () => visitListeners.delete(listener);
}
